"""The golden set: fixed claims every model version is scored on.

Used twice:
    offline     at training time, each version records its scores on these claims
    online      in the canary (Step 5), the served model must reproduce those scores
                exactly; any difference means serving and training disagree

Mostly a random sample, plus enough new-code and OB-tenant claims that a scoring 
bug on those paths can't hide.
"""

from __future__ import annotations

import pandas as pd


def select_golden_set(
    df: pd.DataFrame,
    size:  int, 
    new_codes: set[str], 
    ob_tenants: list[str],
    new_code_claims: int, 
    ob_tenant_claims: int,
    seed: int
) -> pd.DataFrame:
    new = df[df.icd_code.isin(new_codes)]
    ob = df[df.tenant_id.isin(ob_tenants) & ~df.icd_code.isin(new_codes)]
    picked = pd.concat([
        new.sample(min(new_code_claims, len(new)), random_state=seed),
        ob.sample(min(ob_tenant_claims, len(ob)), random_state=seed),
    ])
    rest = df[~df.claim_id.isin(picked.claim_id)].sample(size - len(picked), random_state=seed)
    return pd.concat([picked, rest]).sort_values("claim_id").reset_index(drop=True)