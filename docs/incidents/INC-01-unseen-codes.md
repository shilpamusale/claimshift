# INC-01: Denial model under-predicts claims with new FY2027 ICD-10 codes

| | |
|---|---|
| **Severity** | High: wrong scores on ~6% of claims, no errors raised |
| **Period** | Oct 1 – Dec 1, 2026 (simulated timeline) |
| **Detected by** | Data contract, unseen-code check (`pipeline/contracts.py`) |
| **Status** | Resolved: retrained model v2 |

> Simulated incident. The ICD-10-CM codes and the Oct 1 changes are the official CMS FY2026/FY2027 files; claims, payers and denials are simulated.

## Summary

On Oct 1, 2026, the FY2027 ICD-10-CM code set took effect: 190 new codes, 30 deleted. Claims serviced in October carried codes the denial model (v1) had never seen. Payers were also slow to update their claim edits, so they denied claims with new codes at 28.7%, about 2.5× the normal rate. v1 predicted 12.0% for those claims. Nothing errored, and overall AUC barely moved. The data contract flagged the drift on October's claims, and a retrained model (v2) closed the gap.

## Impact

- **51,220 October claims (6.1%)** carried new FY2027 codes.
- On those claims, v1 predicted a **12.0%** denial rate against an actual **28.7%**: a 2.4× under-prediction. Denial specialists would not have been routed to the claims most likely to be denied.
- Women's-and-children's tenants (T04, T11) were hit hardest: 11.1% of their October claims had unseen codes, vs 5–6% at the largest tenants.
- Overall October AUC fell only from 0.691 (September) to 0.689, which is why aggregate monitoring would not have caught this.

## Timeline

| When | What happened |
|---|---|
| Oct 1 | FY2027 code set effective. Deleted codes (e.g. I42.0) split into child codes (I42.00, I42.01, I42.09); new codes such as the ectopic pregnancy codes O00.131, O00.31 and O00.512 appear |
| October | **Data contract on October claims: FAIL → retrain.** Unseen-code rate 6.14% overall, 50 of 50 tenants over the 2% limit, T04 and T11 worst at 11.1%. Chapter PSI: **0.0000** |
| Nov 1 | Monthly feature refresh picks up October's first denial results for the new codes |
| Nov–Dec | v1's calibration on new-code claims recovers (20.6% predicted vs 19.3% actual) |
| Dec 1 | **v2 retrained** on Aug–Oct claims with labels known by Dec 1 (2.07M rows) |

## Root cause

1. **Unseen categories.** The model encodes each ICD code by its historical denial rate. New codes have no history.
2. **Payer edit lag.** Payers deny new codes at elevated rates until their edits catch up: 28.7% of new-code claims in October, 19.3% across November–December. No feature derived from past data can know this; only fresh labels can.
3. **Coarse monitoring would have missed it.** New codes stay in the same chapters as the codes they replace, so the ICD chapter mix didn't change (PSI 0.0000).

## Resolution

| Change | Effect (October new-code claims unless noted) |
|---|---|
| Hierarchical code rates: unseen code → its category → its chapter (`pipeline/features.py`) | AUC 0.594 → **0.630**; calibration barely improves (11.0% → 12.0% vs 28.7% actual) |
| Monthly feature refresh with newly arrived labels | By Nov–Dec, v1 is calibrated on new codes (20.6% vs 19.3%) |
| Retrain v2 on labels known by Dec 1 | Nov–Dec AUC 0.6940 → **0.7143**; new-code AUC 0.6451 → **0.6649** |

## What went well

- The code-level unseen-code check caught the drift on the first month's claims and named the worst tenants and the specific new codes, with their CMS descriptions.
- Separating "block" (bad data) from "retrain" (changed data) meant the pipeline triggered a retrain instead of stopping.
- Point-in-time labels made it possible to measure what each model would really have known on its training date.

## What didn't go well

- **Monthly granularity.** The contract runs on a full month. In October, the riskiest month, it should run daily.
- **The hierarchy fixes ranking, not level.** Falling back to the parent code helps order new-code claims, but it can't anticipate payer edit lag.
- **Retraining waits on labels.** Denial results arrive 3–8 weeks after service, so a retrain on October's labels couldn't happen until late November.

## Action items

| # | Action | Type |
|---|---|---|
| 1 | Before each Oct 1, load the CMS addenda and seed each new code's rate from the code it replaces (CMS conversion table), with a payer-lag prior from past years | Prevent |
| 2 | Run the data contract daily from Oct 1 to Nov 15 | Detect |
| 3 | Monitor predicted vs actual denial rate on new-code claims as labels arrive, not just overall AUC | Detect |
| 4 | Retrain on the first 2–3 weeks of new-code labels instead of waiting for a full month | Mitigate |
| 5 | Alert per tenant: specialty-heavy tenants saw about twice the impact | Detect |

## Lesson

The dangerous failure was calibration, not ranking: AUC on new-code claims moved a few points while the predicted denial rate was off by 2.4×. Watch the predicted-vs-actual rate on the segment that changed.

## Evidence

- Drift reports: `make contract-demo` → `data/reports/drift_report_2026-10.json`
- Model impact: `make train` → `docs/results/training.json`
- Code changes: `reference/cms/`, `data/reference/icd10cm_fy2027_changes.parquet`