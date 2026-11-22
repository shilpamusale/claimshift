# INC-03: Silent scoring bug in the v2 serving build, and a race in the canary check

| | |
|---|---|
| **Severity** | High: wrong scores with no errors; caught before full rollout |
| **Found** | 2026-11-21, canary rollouts on kind with Argo Rollouts |
| **Detected by** | Golden-set parity check in the canary analysis (`rollout/canary_check.py`) |
| **Status** | Resolved: fixed build v2.1 promoted; routing guard added to the check |

## Summary

The v2 serving build (v2-bad) built each model input from a hand-maintained feature list in which `payer_rate` and `payer_code_rate` were swapped. Every request succeeded, latency was normal, and the average score barely moved, but individual claims were mis-scored by up to 0.11. The canary's golden-set parity check caught it and Argo Rollouts rolled it back automatically.

The first kind run also exposed a bug in the canary check itself: it could start before Kubernetes routed the canary Service to the canary pods. One check passed v2-bad by scoring stable pods.

## Impact

- **v2-bad:** at most 1 of 4 pods (25% of traffic) for **12.9 s** before the automatic rollback. On the golden set, the average score moved from 12.25% to 11.88% and the flagged rate from 14.6% to 13.0%, both normal-looking, while individual scores were off by up to 0.11.
- **Before the routing fix:** v2-bad passed its 25% check and reached 50% of traffic (2 of 4 pods) before the second check rolled it back, 19 s after the deploy. The correct build, v2.1, then failed falsely.

## Timeline (2026-11-21)

| Step | What happened |
|---|---|
| Deploy v1 | 4 stable pods, healthy |
| Roll out v2-bad (run 1) | **25% check passed (false pass).** 50% check failed on parity 0.1096. Rolled back 19 s after the deploy |
| Roll out v2.1 (run 1) | **Failed falsely:** the check's golden set came from a v1 pod (`build v1`) and some scores from v2.1 pods, giving parity 0.1265 |
| Diagnose | `kubectl argo rollouts get` showed the 25% analysis `Successful` and the 50% one `Failed`; the analysis log named a v1 pod as the "canary" |
| Fix | Routing guard: responses carry their pod name; Argo passes the canary's pod-template hash; the check waits for canary-only answers and rejects any other pod |
| Roll out v2-bad (run 2) | Check answered by canary pod `denial-scorer-998dcb66-…`, 0 misrouted. **Parity 0.1096 → fail.** Rolled back **12.9 s** after the deploy |
| Roll out v2.1 (run 2) | Passed at 25% and 50%. **Promoted to 100% in 27.7 s.** 200/200 requests answered by v2.1 |

## Canary check results (run 2, v2-bad)

| Check | Value | Result |
|---|---|---|
| Misrouted responses | 0 | pass |
| **Parity: max \|online − offline\|** | **0.1096** (limit 0.000001) | **fail** |
| Error rate | 0% | pass |
| p95 latency | 32 ms | pass |
| Flagged-rate delta vs stable | 1.5 pts (limit 8) | pass |

Only parity caught it.

## Root cause

1. **A second copy of the feature order.** Serving code kept its own feature list instead of reading the order from the model's manifest. The list held the right names, so a "same features?" check passed; only the order was wrong.
2. **The swapped features look alike.** Both are denial rates with similar ranges, so the model accepted them without complaint and averages stayed plausible.
3. **(Check bug) Analysis raced the Service switch.** Argo starts the analysis as soon as the canary pod is ready. The canary Service's endpoints switch a moment later, so the check's first requests reached stable pods.

## Resolution

- **v2.1** builds every input row by name, in the manifest's order (`training/model_io.prepare_matrix`).
- **Routing guard** in the canary check: every answer must come from a pod with the canary's hash, passed in by Argo with `podTemplateHashValue: Latest`. The check waits up to 90 s for that and fails otherwise.

## What went well

- Parity compares the canary with its *own* offline scores, so it caught a bug that canary-vs-stable comparisons, error rates and latency all missed.
- The rollback was automatic: no human decision, at most 25% exposure.
- The rollout history and analysis logs made the race diagnosable in minutes.

## What didn't go well

- **The safety check had a blind spot,** and it was found only by watching it pass something it shouldn't have.
- **The bug reached the cluster at all.** The same parity comparison could have run in CI before the image was built.
- **A 25% first step is large.** With real traffic, the first step should be 5% or less, which needs a traffic router (ALB or service mesh).

## Action items

| # | Action | Type |
|---|---|---|
| 1 | Remove all hardcoded feature lists from serving; `prepare_matrix` is the only way to build inputs (done in v2.1) | Prevent |
| 2 | Run golden-set parity in CI against every serving build, before the image is pushed | Prevent |
| 3 | Keep the routing guard and its unit test; any canary check must prove which pods answered | Detect |
| 4 | Add a traffic router and make the first canary step 5% | Mitigate |
| 5 | In production, run parity continuously on mirrored traffic, not only at rollout | Detect |

## Lesson

A check you haven't watched fail is a check you can't fully trust. The parity check was right about the model; it was wrong about which pods it was talking to.

## Evidence

- Rollout and check: `make rollout-v2-bad`, `make analysis-logs`, `make rollout-v2.1`, `make traffic`
- Recording: `docs/canary.gif`
- Code: `serving/features.py`, `serving/app.py`, `rollout/canary_check.py`, `rollout/rollout.yaml`, `rollout/analysis-template.yaml`