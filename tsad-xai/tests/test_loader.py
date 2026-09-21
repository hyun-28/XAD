"""BRIEF B6: tests/test_loader.py — the format variants found in A3."""
import numpy as np
import pytest

from src.data.ucr import file_layout, load_series, scan_archive


def _write(tmp_path, name, text, newline=""):
    p = tmp_path / name
    p.write_text(text, newline=newline)
    return p


def test_one_value_per_line_crlf(tmp_path):
    p = _write(tmp_path, "a.txt", "   1.0\r\n  -2.5e+02\r\n 3\r\n")
    x = load_series(p)
    assert x.dtype == np.float64 and x.shape == (3,)
    assert x.tolist() == [1.0, -250.0, 3.0]
    assert file_layout(p)["has_crlf"] and not file_layout(p)["single_line"]


def test_single_line_space_separated(tmp_path):
    p = _write(tmp_path, "b.txt", "1.0 2.0   3.0\r\n")
    assert load_series(p).tolist() == [1.0, 2.0, 3.0]
    assert file_layout(p)["single_line"]


def test_single_line_tab_separated(tmp_path):
    p = _write(tmp_path, "c.txt", "1.0\t2.0\t3.0")
    assert load_series(p).tolist() == [1.0, 2.0, 3.0]
    assert file_layout(p)["has_tab"]


def test_empty_file_raises(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        load_series(_write(tmp_path, "d.txt", "\r\n\r\n"))


def test_non_numeric_raises(tmp_path):
    with pytest.raises(ValueError, match="non-numeric"):
        load_series(_write(tmp_path, "e.txt", "1.0\nabc\n"))


def test_scan_archive_count_and_numbering(tmp_path):
    for i in (1, 2):
        _write(tmp_path, f"{i:03d}_UCR_Anomaly_x_10_20_30.txt", "1\n")
    with pytest.raises(RuntimeError, match="expected 3"):
        scan_archive(tmp_path, 3)
    assert [m.num for m in scan_archive(tmp_path, 2)] == [1, 2]
    _write(tmp_path, "003_bad_name.txt", "1\n")
    with pytest.raises(RuntimeError, match="do not parse"):
        scan_archive(tmp_path, 3)


def test_scan_archive_rejects_gaps(tmp_path):
    _write(tmp_path, "001_UCR_Anomaly_x_10_20_30.txt", "1\n")
    _write(tmp_path, "003_UCR_Anomaly_x_10_20_30.txt", "1\n")
    with pytest.raises(RuntimeError, match="numbering"):
        scan_archive(tmp_path, 2)


# --- against the real archive (skipped if 01_download.py has not run) -------

def test_real_single_line_files_load(official_fulldata_dir):
    # Slide 7 of UCR_AnomalyDataSets.pptx lists these nine as "formatted differently".
    metas = scan_archive(official_fulldata_dir, 250)
    single = [m for m in metas if file_layout(official_fulldata_dir / m.filename)["single_line"]]
    assert sorted(m.num for m in single) == [204, 205, 206, 207, 208, 225, 226, 242, 243]
    for m in single:
        x = load_series(official_fulldata_dir / m.filename)
        assert x.ndim == 1 and x.dtype == np.float64 and np.isfinite(x).all()
        assert len(x) > m.end_raw - 1
