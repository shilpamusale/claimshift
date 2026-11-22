# INC-02: Whale tenant creates a straggler in the retraining feature build

| | |
|---|---|
| **Severity** | Medium: slower retraining builds; grows with data volume |
| **Found** | 2026-11-21, while building features for the v2 retrain |
| **Detected by** | Spark event log: rows and run time per task (`pipeline/skew_bench.py`) |
| **Status** | Resolved: broadcast join by default; salting documented as fallback |

## Summary

The feature build for October's claims joins every claim to its tenant's profile. Three "whale" health systems hold about 57% of claims, and the largest, T01, holds 34%. Spark shuffled the claims by `tenant_id`, so all of T01's claims landed in one task. That task processed 34.5% of October's rows and ran 6.7× longer than the median task. Spark's Adaptive Query Execution (AQE), on by default, did not fix it. An explicit broadcast join did.

## Impact

- In the join stage, **one task of 16 processed 290,320 of 842,391 rows (34.5%)** and took 140 ms against a 21 ms median.
- At 5M claims the whole build took 1.74 s, so the cost was small. At production scale the stage waits on the straggler, and retraining time grows with the largest tenant rather than with the cluster size.

## Timeline (2026-11-21)

| Step | Finding |
|---|---|
| Benchmark the October build | Join stage: biggest task 290,320 rows (34.5%), 140 ms vs 21 ms median |
| Read the physical plan | Tenant join is a **sort-merge join**: claims are shuffled by `tenant_id` |
| Enable AQE (Spark's default) | Biggest task **still 290,320 rows**. AQE merged 16 partitions into 5, which made max/median look fine, while the straggler stayed the same |
| AQE with skew-join split tuned to laptop scale | **Still 290,320 rows.** No split happened |
| Broadcast hint on the tenant profile | No shuffle; wall time 1.74 s → **1.21 s (−30%)** |
| Salting, 16 buckets | Biggest task 290,320 → **102,330 rows**; 140 → 77 ms; wall time −13% |

## Root cause

1. **Skewed key.** Tenant volume follows a Zipf distribution, so a shuffle on `tenant_id` sends a third of the rows to one task.
2. **The planner couldn't see the table was small.** The tenant profile is an aggregate computed in the same job. Without statistics, Spark estimated it from its input (millions of rows) and chose a sort-merge join instead of a broadcast.
3. **Why AQE didn't help:**
   - It *coalesced* the small partitions, which hides skew in max/median ratios without shrinking the straggler.
   - It couldn't *broadcast* at runtime: the partially aggregated tenant table was 13 MB (the median sketches are large), over the 10 MB limit.
   - It couldn't *split* the skewed partition: AQE's skew-join split needs both sides of the join to be raw shuffles, and here an aggregate sits between the shuffle and the join.

## Resolution

- **Broadcast the tenant profile explicitly** (`F.broadcast`, now the default in `pipeline/features.py`). It is 50 rows, so copying it to every task removes the shuffle.
- **Salting** (`strategy="salted"`) kept as the fallback for when the small side is too big to broadcast.
- **A test proves all three join strategies produce identical features** (`tests/test_features.py`).

## What went well

- The event log gave exact per-task row counts, which are deterministic, so the diagnosis didn't depend on noisy timings. The row counts matched on two different machines.
- Reading the physical plan, not just the setting, exposed that AQE kept the sort-merge join.

## What didn't go well

- The first benchmark reported a max/median *ratio* that AQE's partition merging made look healthy. Switching the headline metric to the biggest task's **share** of rows fixed that.
- The skew would have stayed invisible in wall-clock time at 5M rows; it was found by looking for it.

## Action items

| # | Action | Type |
|---|---|---|
| 1 | Add a plan test: fail CI if any join on a known-small dimension table is not a broadcast | Prevent |
| 2 | Log the biggest task's share of rows per stage for every feature build; alert above 20% | Detect |
| 3 | Materialize the tenant profile as a feature-store table with statistics, so the planner can size it | Prevent |
| 4 | Re-run the benchmark at 50–100M rows (EMR Serverless) to size the effect at production scale | Verify |

## Lesson

Turning on AQE is not the same as fixing skew. Check the physical plan and the per-task row counts: merged partitions can make the numbers look balanced while one task still does a third of the work.

## Evidence

- Benchmark: `make skew-bench` → `docs/results/skew_bench.json`
- Code: `pipeline/features.py` (`join_tenant`), `pipeline/skew_bench.py`