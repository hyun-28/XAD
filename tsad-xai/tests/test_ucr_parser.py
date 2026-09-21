"""BRIEF B6: tests/test_ucr_parser.py — underscore-in-name and rejection cases."""
import pytest

from src.data.ucr import UcrFilenameError, parse_filename


def test_documented_example():
    m = parse_filename("012_UCR_Anomaly_tiltAPB1_100000_114283_114350.txt")
    assert (m.num, m.name, m.train_end_raw, m.begin_raw, m.end_raw) == (12, "tiltAPB1", 100000, 114283, 114350)
    assert m.variant == "plain" and not m.is_distorted and m.base_name == "tiltAPB1"
    assert m.entity == "012_UCR_Anomaly_tiltAPB1"


def test_name_with_underscores_is_kept_whole():
    m = parse_filename("999_UCR_Anomaly_mit14046long_term_ecg_81214_143000_143300.txt")
    assert m.name == "mit14046long_term_ecg"
    assert (m.train_end_raw, m.begin_raw, m.end_raw) == (81214, 143000, 143300)


def test_name_ending_in_digits_does_not_steal_fields():
    m = parse_filename("001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.txt")
    assert m.name == "DISTORTED1sddb40" and m.is_distorted and m.base_name == "1sddb40"
    assert m.variant == "DISTORTED"
    n = parse_filename("102_UCR_Anomaly_NOISEMesoplodonDensirostris_10000_19280_19440.txt")
    assert n.variant == "NOISE" and not n.is_distorted and n.base_name == "MesoplodonDensirostris"


def test_full_path_is_accepted():
    m = parse_filename("/some/dir/250_UCR_Anomaly_weallwalk_2951_7290_7296.txt")
    assert m.num == 250 and m.filename == "250_UCR_Anomaly_weallwalk_2951_7290_7296.txt"


@pytest.mark.parametrize("bad", [
    "001_NAB_id_1_Facility_tr_1007_1st_2014.csv",          # TSB-AD format
    "12_UCR_Anomaly_tiltAPB1_100000_114283_114350.txt",     # num not 3 digits
    "012_UCR_Anomaly_tiltAPB1_100000_114283.txt",           # only two fields
    "012_UCR_Anomaly_tiltAPB1_100000_114283_114350.csv",    # wrong extension
    "012_UCR_Anomaly__100000_114283_114350.txt",            # empty name
    "UCR_Anomaly_tiltAPB1_100000_114283_114350.txt",        # no num
])
def test_rejects_malformed(bad):
    with pytest.raises(UcrFilenameError):
        parse_filename(bad)
