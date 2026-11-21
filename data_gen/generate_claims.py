"""Generate synthetic claims for Jul-Dec 2026 with an Oct 1 ICD-10 code-set switch.

Design (see README):
- Real ICD-10-CM codes (CMS FY2026 -> FY2027); simulated volumes (see code_sets.py)
- 50 tenants (health systems) with Zipf-distributed volume -> whale tenants
- 2 women's-and-children's tenants (T04, T11) with heavy obstetric case mix,
  where most FY2027 new codes land
- 8 payers with different timely-filing limits; some have *edit lag*: they deny
  more of the new FY2027 codes in Oct-Nov until their claim edits catch up
- Denial labels come from a logistic model of payer, tenant, code, procedure,
  missing prior auth, late filing, and billed amount

Usage:
    python -m data_gen.generate_claims --rows 5000000 --seed 7 --out data
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data_gen.code_sets import N_CHAPTERS, CodeSets, load_cms_code_sets, write_code_sets

MONTHS = pd.date_range("2026-07-01", "2026-12-01", freq="MS")
CUTOVER = pd.Timestamp("2026-10-01")
N_TENANTS = 50
ZIPF_S = 1.3
BASE_LOGIT = -2.3
EDIT_LAG_BOOST = {10: 1.5, 11: 1.0, 12: 0.4}  # month -> logit boost on new codes
OB_TENANTS = (3, 10)  # T04, T11
OB_CHAPTERS = ("15", "16")  # pregnancy, perinatal
OB_MULTIPLIER = 8.0

PAYERS = pd.DataFrame(
    {
        "payer_id": [f"P{i:02d}" for i in range(1, 9)],
        "share": [0.22, 0.16, 0.12, 0.14, 0.08, 0.12, 0.06, 0.10],
        "timely_filing_days": [90, 180, 90, 365, 365, 365, 180, 120],
        "edit_lag": [True, False, True, True, False, True, False, True],
    }
)

PROCS = pd.DataFrame(
    {
        "proc_category": [
            "em_visit", "imaging_advanced", "imaging_basic", "lab",
            "surgery_outpatient", "surgery_inpatient", "dme", "pt_ot",
            "behavioral", "oncology_infusion", "cardiology_diag", "emergency",
        ],
        "share": [0.26, 0.06, 0.09, 0.20, 0.06, 0.03, 0.03, 0.06, 0.05, 0.03, 0.05, 0.08],
        "pa_prob": [0.0, 0.85, 0.05, 0.0, 0.40, 0.90, 0.70, 0.30, 0.50, 0.90, 0.20, 0.0],
        "amount_median": [180, 1400, 250, 60, 3200, 18000, 900, 140, 160, 6500, 700, 1100],
    }
)


@dataclass(frozen=True)
class World:
    """Fixed structure shared by every month: who exists and how they behave."""

    codes: CodeSets
    tenant_share: np.ndarray
    tenant_effect: np.ndarray
    tenant_pa_rate: np.ndarray
    tenant_submit_speed: np.ndarray
    working_ids: np.ndarray    # code_ids billable before Oct 1 (weight > 0)
    tenant_code_p: np.ndarray  # (tenants, working codes) case-mix probabilities
    payer_effect: np.ndarray
    proc_effect: np.ndarray


def build_world(seed: int) -> World:
    rng = np.random.default_rng(seed)
    cs = load_cms_code_sets(rng)
    base = cs.codes.loc[cs.codes.weight > 0]

    share = 1.0 / np.arange(1, N_TENANTS + 1) ** ZIPF_S
    share /= share.sum()

    chapter_idx = base.chapter.astype(int).to_numpy() - 1
    chapter_mult = rng.gamma(4.0, 0.25, size=(N_TENANTS, N_CHAPTERS))
    for t in OB_TENANTS:
        for ch in OB_CHAPTERS:
            chapter_mult[t, int(ch) - 1] = OB_MULTIPLIER  # others average 1.0
    code_p = base.weight.to_numpy()[None, :] * chapter_mult[:, chapter_idx]
    code_p /= code_p.sum(axis=1, keepdims=True)

    return World(
        codes=cs,
        working_ids=base.code_id.to_numpy(),
        tenant_share=share,
        tenant_effect=rng.normal(0.0, 0.25, N_TENANTS),
        tenant_pa_rate=rng.uniform(0.80, 0.96, N_TENANTS),
        tenant_submit_speed=rng.lognormal(0.0, 0.3, N_TENANTS),
        tenant_code_p=code_p,
        payer_effect=rng.normal(0.0, 0.3, len(PAYERS)),
        proc_effect=rng.normal(0.0, 0.3, len(PROCS)),
    )


def generate_month(world: World, month: pd.Timestamp, n: int, seed: int, id_offset: int) -> pd.DataFrame:
    rng = np.random.default_rng([seed, month.month])
    codes = world.codes.codes

    tenant = rng.choice(N_TENANTS, size=n, p=world.tenant_share)
    code_id = np.empty(n, dtype=np.int32)
    counts = np.bincount(tenant, minlength=N_TENANTS)
    for t in range(N_TENANTS):
        if counts[t]:
            code_id[tenant == t] = world.working_ids[
                rng.choice(len(world.working_ids), size=counts[t], p=world.tenant_code_p[t])
            ]

    # Oct 1 switch: deleted codes move to their replacements; codes that gained
    # new siblings hand part of their volume to the new, more specific codes.
    if month >= CUTOVER:
        for src, (targets, target_p, share) in world.codes.recode.items():
            mask = code_id == src
            if share < 1.0:
                mask &= rng.random(n) < share
            if mask.any():
                code_id[mask] = rng.choice(targets, size=int(mask.sum()), p=target_p)

    payer = rng.choice(len(PAYERS), size=n, p=PAYERS.share.to_numpy())
    proc = rng.choice(len(PROCS), size=n, p=PROCS.share.to_numpy())

    days_in_month = month.days_in_month
    service_date = month + pd.to_timedelta(rng.integers(0, days_in_month, n), unit="D")
    days_to_submit = np.clip(
        np.round(rng.lognormal(np.log(10), 1.0, n) * world.tenant_submit_speed[tenant]), 0, 400
    ).astype(np.int16)
    submit_date = service_date + pd.to_timedelta(days_to_submit, unit="D")
    remit_date = submit_date + pd.to_timedelta(rng.integers(14, 46, n), unit="D")

    pa_required = rng.random(n) < PROCS.pa_prob.to_numpy()[proc]
    pa_obtained = pa_required & (rng.random(n) < world.tenant_pa_rate[tenant])
    late = days_to_submit > PAYERS.timely_filing_days.to_numpy()[payer]

    median = PROCS.amount_median.to_numpy()[proc]
    billed = np.round(rng.lognormal(np.log(median), 0.6), 2)

    is_new = codes.is_new.to_numpy()[code_id]
    lag = PAYERS.edit_lag.to_numpy()[payer]
    logit = (
        BASE_LOGIT
        + world.payer_effect[payer]
        + world.tenant_effect[tenant]
        + codes.effect.to_numpy()[code_id]
        + world.proc_effect[proc]
        + 2.0 * (pa_required & ~pa_obtained)
        + 2.5 * late
        + 0.15 * np.log(billed / median)
        + EDIT_LAG_BOOST.get(month.month, 0.0) * (is_new & lag)
    )
    denied = rng.random(n) < 1.0 / (1.0 + np.exp(-logit))

    return pd.DataFrame(
        {
            "claim_id": np.arange(id_offset, id_offset + n, dtype=np.int64),
            "tenant_id": np.char.add("T", np.char.zfill((tenant + 1).astype(str), 2)),
            "payer_id": PAYERS.payer_id.to_numpy()[payer],
            "service_date": service_date.date,
            "submit_date": submit_date.date,
            "remit_date": remit_date.date,
            "icd_code": codes.code.to_numpy()[code_id],
            "icd_category": codes.category.to_numpy()[code_id],
            "icd_chapter": codes.chapter.to_numpy()[code_id],
            "code_set_version": "FY2027" if month >= CUTOVER else "FY2026",
            "proc_category": PROCS.proc_category.to_numpy()[proc],
            "billed_amount": billed,
            "prior_auth_required": pa_required,
            "prior_auth_obtained": pa_obtained,
            "days_to_submit": days_to_submit,
            "denied": denied.astype(np.int8),
            "_is_new_code": is_new,  # ground truth for the manifest only, dropped before writing
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()

    t0 = time.time()
    world = build_world(args.seed)
    write_code_sets(world.codes, args.out / "reference")

    days = np.array([m.days_in_month for m in MONTHS])
    rows_per_month = np.floor(args.rows * days / days.sum()).astype(int)
    rows_per_month[-1] += args.rows - rows_per_month.sum()

    claims_dir = args.out / "raw" / "claims"
    stats = {"rows": int(args.rows), "seed": args.seed, "months": {}}
    offset = 0
    tenant_counts = np.zeros(N_TENANTS, dtype=np.int64)
    for month, n in zip(MONTHS, rows_per_month):
        df = generate_month(world, month, int(n), args.seed, offset)
        offset += n
        tenant_counts += df.tenant_id.value_counts().reindex(
            [f"T{i:02d}" for i in range(1, N_TENANTS + 1)], fill_value=0
        ).to_numpy()
        new = df["_is_new_code"]
        stats["months"][month.strftime("%Y-%m")] = {
            "rows": int(n),
            "denial_rate": round(float(df.denied.mean()), 4),
            "new_code_share": round(float(new.mean()), 4),
            "new_code_denial_rate": round(float(df.denied[new].mean()), 4) if new.any() else None,
        }
        if month == CUTOVER:
            by_tenant = new.groupby(df.tenant_id).mean()
            stats["oct_new_code_share_by_tenant"] = {
                t: round(float(by_tenant[t]), 4) for t in ("T01", "T02", "T03", "T04", "T11")
            }
        part = claims_dir / f"service_month={month.strftime('%Y-%m')}"
        part.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df.drop(columns="_is_new_code"), preserve_index=False)
        pq.write_table(table, part / "part-00000.parquet", compression="snappy")
        print(f"{month:%Y-%m}: {n:>9,} rows  denial={stats['months'][month.strftime('%Y-%m')]['denial_rate']:.3f}")

    share = tenant_counts / tenant_counts.sum()
    stats["top3_tenant_share"] = round(float(share[:3].sum()), 4)
    stats["n_new_codes"] = int(world.codes.codes.is_new.sum())
    stats["n_deleted_codes"] = int(world.codes.codes.is_deleted.sum())
    stats["seconds"] = round(time.time() - t0, 1)
    (args.out / "raw" / "_manifest.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "months"}, indent=2))


if __name__ == "__main__":
    main()
