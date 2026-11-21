import numpy as np
import pandas as pd
import pytest

from data_gen.generate_claims import CUTOVER, build_world, generate_month


@pytest.fixture(scope="module")
def world():
    return build_world(seed=7)


@pytest.fixture(scope="module")
def sep(world):
    return generate_month(world, pd.Timestamp("2026-09-01"), 40_000, seed=7, id_offset=0)


@pytest.fixture(scope="module")
def octo(world):
    return generate_month(world, pd.Timestamp("2026-10-01"), 40_000, seed=7, id_offset=40_000)


def test_no_new_codes_before_cutover(world, sep):
    new_codes = set(world.codes.codes.loc[world.codes.codes.is_new, "code"])
    assert len(new_codes) == 190  # official FY2027 count
    assert not set(sep.icd_code) & new_codes


def test_deleted_codes_gone_after_cutover(world, octo):
    deleted = set(world.codes.codes.loc[world.codes.codes.is_deleted, "code"])
    assert len(deleted) == 30  # official FY2027 count
    assert not set(octo.icd_code) & deleted
    assert octo["_is_new_code"].mean() > 0.03


def test_whale_tenants_dominate(sep):
    top3 = sep.tenant_id.value_counts(normalize=True).iloc[:3].sum()
    assert 0.5 < top3 < 0.65


def test_denial_rate_plausible(sep):
    assert 0.06 < sep.denied.mean() < 0.16


def test_payer_edit_lag_raises_new_code_denials(octo):
    new = octo["_is_new_code"]
    assert octo.denied[new].mean() > octo.denied[~new].mean()


def test_dates_ordered_and_codes_well_formed(world, octo):
    assert (octo.submit_date >= octo.service_date).all()
    assert (octo.remit_date > octo.submit_date).all()
    assert (pd.to_datetime(octo.service_date) >= CUTOVER).all()
    assert octo.icd_code.str.match(r"^[A-Z][0-9A-Z]{2}(\.[0-9A-Z]{1,4})?$").all()
    assert set(octo.icd_code) <= set(world.codes.fy2027.code)  # every code valid on its date
    assert (octo.icd_code.str[:3] == octo.icd_category).all()
    assert np.array_equal(octo.claim_id.to_numpy(), np.arange(40_000, 80_000))


def test_ob_tenants_see_more_new_codes(octo):
    by_tenant = octo["_is_new_code"].groupby(octo.tenant_id).mean()
    assert min(by_tenant["T04"], by_tenant["T11"]) > 1.5 * by_tenant["T01"]
