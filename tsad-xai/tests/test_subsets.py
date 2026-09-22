"""BRIEF_A-followup F7: subset selection and group counting (F4)."""
import pytest

from src.data.subsets import n_groups, select, summarise

CFG = {
    "all": {},
    "non_medical": {"domain_goswami_not_in": ["ABP", "ECG", "RESP"]},
    "physical": {"domain_goswami_in": ["Acceleration", "Air Temperature", "NASA", "Power Demand"]},
    "plain_only": {"variant": "plain"},
    "no_sentinel": {"sentinel_in_train_n": 0, "sentinel_in_test_normal_n": 0},
}


def _rows():
    spec = [  # num, domain, variant, name_group, sentinel counts
        (1, "ECG", "plain", "a", 0, 0), (2, "ECG", "DISTORTED", "a", 0, 0),
        (3, "Gait", "plain", "b", 1, 0), (4, "Gait", "NOISE", "b", 0, 0),
        (5, "NASA", "plain", "c", 0, 0), (6, "NASA", "DISTORTED", "c", 0, 2),
        (7, "Power Demand", "plain", "d", 0, 0),
    ]
    return [{"num": n, "domain_goswami": d, "variant": v, "name_group": g,
             "sentinel_in_train_n": s1, "sentinel_in_test_normal_n": s2} for n, d, v, g, s1, s2 in spec]


def test_single_subsets():
    rows = _rows()
    assert [r["num"] for r in select(rows, ["non_medical"], CFG)] == [3, 4, 5, 6, 7]
    assert [r["num"] for r in select(rows, ["physical"], CFG)] == [5, 6, 7]
    assert [r["num"] for r in select(rows, ["plain_only"], CFG)] == [1, 3, 5, 7]
    assert [r["num"] for r in select(rows, ["no_sentinel"], CFG)] == [1, 2, 4, 5, 7]
    assert len(select(rows, ["all"], CFG)) == 7


def test_intersection_and_group_count():
    rows = _rows()
    sel = select(rows, ["physical", "plain_only"], CFG)
    assert [r["num"] for r in sel] == [5, 7]
    assert n_groups(sel, "name_group") == 2
    assert n_groups(select(rows, ["non_medical"], CFG), "name_group") == 3
    assert n_groups(select(rows, ["physical", "no_sentinel"], CFG), "name_group") == 2


def test_summarise_marks_small_subsets_descriptive():
    d = summarise(_rows(), ["physical", "plain_only"], unit="name_group", min_groups=3, subsets_cfg=CFG)
    assert d == {"subset": "physical ∩ plain_only", "n_series": 2, "n_groups": 2,
                 "unit": "name_group", "descriptive_only": True}


def test_unknown_subset_or_column_raises():
    with pytest.raises(KeyError):
        select(_rows(), ["nope"], CFG)
    with pytest.raises(KeyError):
        select(_rows(), ["x"], {"x": {"missing_col": 1}})
    with pytest.raises(ValueError):
        select(_rows(), ["x"], {"x": {"variant": ["plain"]}})   # list needs _in
