"""Real ICD-10-CM code sets (CMS FY2026 and FY2027) and the Oct 1, 2026 update.

Inputs, committed under reference/cms/, come from the CMS "Code Descriptions in
Tabular Order" downloads:
    icd10cm_codes_2026.txt.gz, icd10cm_codes_2027.txt.gz

What is real: every code, every description, and the Oct 1 changes (new codes,
deleted codes). What is simulated: how often each code is billed. Claim volumes
are not public, so code popularity is a seeded prior, and the codes touched by
the update are oversampled so the change is visible at laptop scale.

How each change is modeled after Oct 1 (sources found by longest shared prefix):
    expansion        deleted code -> its new, more specific child codes (100% of volume)
    remap            deleted code with no new child -> closest valid sibling,
                     preferring the "unspecified" (9) code, as a coder would
    new_specificity  existing code -> new sibling codes (ADOPT_SHARE of volume moves)
"""

from __future__ import annotations

import gzip
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CMS_DIR = Path(__file__).resolve().parents[1] / "reference" / "cms"

# ICD-10-CM chapters by category range (first 3 characters).
CHAPTERS = [
    ("01", "A00", "B99"), ("02", "C00", "D49"), ("03", "D50", "D89"), ("04", "E00", "E89"),
    ("05", "F01", "F99"), ("06", "G00", "G99"), ("07", "H00", "H59"), ("08", "H60", "H95"),
    ("09", "I00", "I99"), ("10", "J00", "J99"), ("11", "K00", "K95"), ("12", "L00", "L99"),
    ("13", "M00", "M99"), ("14", "N00", "N99"), ("15", "O00", "O9A"), ("16", "P00", "P96"),
    ("17", "Q00", "Q99"), ("18", "R00", "R99"), ("19", "S00", "T88"), ("20", "V00", "Y99"),
    ("21", "Z00", "Z99"), ("22", "U00", "U85"),
]
N_CHAPTERS = len(CHAPTERS)

# Simulation prior: rough share of claims by chapter as the principal diagnosis.
# Chapter 20 (external causes) is never a principal diagnosis, so it is excluded.
CHAPTER_PRIOR = {
    "01": 1.5, "02": 3.0, "03": 1.0, "04": 6.0, "05": 5.0, "06": 3.0, "07": 2.0, "08": 1.5,
    "09": 8.0, "10": 6.0, "11": 5.0, "12": 2.5, "13": 9.0, "14": 4.0, "15": 3.0, "16": 0.7,
    "17": 0.5, "18": 7.0, "19": 6.0, "20": 0.0, "21": 12.0, "22": 0.3,
}

WORKING_SET = 3000      # commonly billed codes sampled from the FY2026 set
AFFECTED_SHARE = 0.08   # share of pre-Oct volume on codes the update touches
ADOPT_SHARE = 0.60      # share of a donor code's volume that moves to new siblings
COMMON_SHARE = 0.15     # share of volume on well-known, frequently billed codes
COMMON_CODES = [
    "I10", "E119", "E785", "Z0000", "Z0001", "Z23", "J069", "M5450", "R059", "E669",
    "F411", "F32A", "K219", "N390", "R109", "M25561", "J029", "E039", "Z1231", "R519",
    "Z794", "I2510", "J45909", "G4733", "M1711",
]


def chapter_of(category: str) -> str:
    for ch, lo, hi in CHAPTERS:
        if lo <= category <= hi:
            return ch
    # New categories outside the published ranges (e.g. QA1) fall back by letter.
    return [ch for ch, lo, _ in CHAPTERS if lo[0] == category[0]][-1]


def dotted(code: str) -> str:
    return f"{code[:3]}.{code[3:]}" if len(code) > 3 else code


def read_cms_codes(path: Path) -> pd.DataFrame:
    """Parse a CMS icd10cm_codes_YYYY file: code in columns 1-7, then description."""
    rows = []
    with gzip.open(path, "rt", encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.strip():
                rows.append((line[:8].strip(), line[8:].strip()))
    df = pd.DataFrame(rows, columns=["raw", "description"])
    df["category"] = df.raw.str[:3]
    df["chapter"] = df.category.map(chapter_of)
    df["code"] = df.raw.map(dotted)
    return df


def _lcp(a: str, b: str) -> int:
    n, i = min(len(a), len(b)), 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _best_match(code: str, candidates: list[str]) -> str:
    """Longest shared prefix, then same 7th and 6th characters, then prefer '9'."""

    def key(c: str):
        n = _lcp(code, c)
        return (-n, c[-1] != code[-1], c[-2:-1] != code[-2:-1], c[n : n + 1] != "9", c)

    return min(candidates, key=key)


def _candidates(code: str, by_prefix: dict[str, list[str]]) -> list[str]:
    for k in (3, 2, 1):
        if by_prefix.get(code[:k]):
            return by_prefix[code[:k]]
    raise ValueError(f"no candidate for {code}")


@dataclass(frozen=True)
class CodeSets:
    """codes: one row per code used in simulation, with columns
        code_id, raw, code, description, category, chapter, weight, effect,
        is_new, is_deleted
    recode: FY2026 code_id -> (target code_ids, target probabilities, share moved)
    changes: old_code, new_code, change_type (for docs and the blog)
    fy2026 / fy2027: full official code tables
    """

    codes: pd.DataFrame
    recode: dict[int, tuple[np.ndarray, np.ndarray, float]]
    changes: pd.DataFrame
    fy2026: pd.DataFrame
    fy2027: pd.DataFrame


def load_cms_code_sets(rng: np.random.Generator, cms_dir: Path = CMS_DIR) -> CodeSets:
    fy26 = read_cms_codes(cms_dir / "icd10cm_codes_2026.txt.gz")
    fy27 = read_cms_codes(cms_dir / "icd10cm_codes_2027.txt.gz")
    old, new = set(fy26.raw), set(fy27.raw)
    added, deleted = sorted(new - old), sorted(old - new)

    def index(codes):
        idx = defaultdict(list)
        for c in codes:
            for k in (1, 2, 3):
                idx[c[:k]].append(c)
        return idx

    by26, by27_valid = index(sorted(old)), index(sorted(new - set(added)))

    # 1. Each new code gets a source FY2026 code by longest shared prefix.
    children: dict[str, list[str]] = defaultdict(list)
    change_type: dict[tuple[str, str], str] = {}
    for n in added:
        src = _best_match(n, _candidates(n, by26))
        children[src].append(n)
        change_type[(src, n)] = "expansion" if src in deleted else "new_specificity"
    # 2. Deleted codes with no new child are remapped to the closest valid sibling.
    for d in deleted:
        if d not in children:
            tgt = _best_match(d, _candidates(d, by27_valid))
            children[d].append(tgt)
            change_type[(d, tgt)] = "remap"

    affected = set(children)
    remap_targets = {t for (s, t), kind in change_type.items() if kind == "remap"}

    # 3. Working set of commonly billed FY2026 codes, plus every affected code.
    pool = fy26[(fy26.chapter != "20") & ~fy26.raw.isin(deleted)].copy()
    cat_w = {c: rng.lognormal(0.0, 1.0) for c in pool.category.unique()}
    rank = pool.assign(r=rng.random(len(pool))).groupby("category").r.rank(method="first")
    pool["w"] = pool.chapter.map(CHAPTER_PRIOR) * pool.category.map(cat_w) / rank**1.1
    # Real claims favor less specific codes and initial encounters over follow-ups.
    pool["w"] *= 0.6 ** (pool.raw.str.len() - 3)
    seventh = pool.raw.str[6:7]
    pool["w"] *= np.where(seventh == "D", 0.3, np.where(seventh == "S", 0.1, 1.0))
    picked = set(rng.choice(pool.raw.to_numpy(), size=WORKING_SET, replace=False, p=(pool.w / pool.w.sum()).to_numpy()))
    common = set(COMMON_CODES) & set(pool.raw)
    working = picked | affected | remap_targets | common

    codes = fy26[fy26.raw.isin(working)].copy()
    codes["is_new"] = False
    codes["is_deleted"] = codes.raw.isin(deleted)
    new_rows = fy27[fy27.raw.isin(added)].copy()
    new_rows["is_new"], new_rows["is_deleted"] = True, False
    codes = pd.concat([codes, new_rows], ignore_index=True)

    # Popularity: unaffected codes share (1 - AFFECTED_SHARE). Affected codes share
    # the rest, split across chapters in proportion to the chapter's new-code count,
    # so volume follows the size of the real update (obstetrics: 44 of 190).
    base_w = pool.set_index("raw").w
    codes["weight"] = codes.raw.map(base_w).fillna(0.0)
    is_common = codes.raw.isin(common)
    codes.loc[is_common, "weight"] = rng.lognormal(0.0, 0.5, int(is_common.sum()))
    codes.loc[is_common, "weight"] *= COMMON_SHARE / codes.loc[is_common, "weight"].sum()
    plain = ~codes.raw.isin(affected) & ~codes.is_new & ~is_common
    codes.loc[plain, "weight"] *= (1 - AFFECTED_SHARE - COMMON_SHARE) / codes.loc[plain, "weight"].sum()
    new_per_chapter = codes.loc[codes.is_new].chapter.value_counts()
    aff = codes.loc[codes.raw.isin(affected) & ~codes.is_new]
    for ch, grp in aff.groupby("chapter"):
        w = rng.lognormal(0.0, 0.7, len(grp))
        codes.loc[grp.index, "weight"] = w / w.sum() * new_per_chapter.get(ch, 1)
    is_aff = codes.index.isin(aff.index)
    codes.loc[is_aff, "weight"] *= AFFECTED_SHARE / codes.loc[is_aff, "weight"].sum()

    # Denial effects: category effect + code noise; new codes inherit from their source.
    cat_eff = {c: rng.normal(0.0, 0.5) for c in codes.category.unique()}
    codes["effect"] = codes.category.map(cat_eff) + rng.normal(0.0, 0.25, len(codes))
    eff = codes.set_index("raw").effect
    for (src, tgt), kind in change_type.items():
        if kind != "remap":
            eff[tgt] = eff[src] + rng.normal(0.0, 0.2)
    codes["effect"] = codes.raw.map(eff)

    codes.insert(0, "code_id", np.arange(len(codes), dtype=np.int32))
    cid = dict(zip(codes.raw, codes.code_id))
    recode = {}
    for src, tgts in children.items():
        share = 1.0 if src in deleted else ADOPT_SHARE
        recode[cid[src]] = (np.array([cid[t] for t in tgts]), rng.dirichlet(np.full(len(tgts), 2.0)), share)

    changes = pd.DataFrame(
        [(dotted(s), dotted(t), k) for (s, t), k in sorted(change_type.items())],
        columns=["old_code", "new_code", "change_type"],
    )
    cols = ["code", "description", "category", "chapter"]
    return CodeSets(codes=codes, recode=recode, changes=changes, fy2026=fy26[cols], fy2027=fy27[cols])


def write_code_sets(cs: CodeSets, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cs.fy2026.to_parquet(out_dir / "icd10cm_fy2026.parquet", index=False)
    cs.fy2027.to_parquet(out_dir / "icd10cm_fy2027.parquet", index=False)
    cs.changes.to_parquet(out_dir / "icd10cm_fy2027_changes.parquet", index=False)
