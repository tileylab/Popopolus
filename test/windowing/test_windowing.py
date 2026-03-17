import pandas as pd
import pytest

from popopolus.windowing.windowing import parse_interval_spec
from popopolus.windowing.windowing import build_windows


def test_parse_interval_spec():
    assert parse_interval_spec("0") is None
    assert parse_interval_spec(0) is None
    assert parse_interval_spec("1000:250") == (1000, 250)


@pytest.mark.parametrize("interval", ["10", "10:-1", "0:10", "foo:bar"])
def test_parse_interval_spec_invalid(interval):
    with pytest.raises((ValueError, TypeError)):
        parse_interval_spec(interval)


def test_build_windows_from_site_coordinates():
    site_df = pd.DataFrame(
        {
            "site_index": [0, 1, 2, 3],
            "chromosome": ["chr1", "chr1", "chr1", "chr2"],
            "position": [100, 150, 320, 50],
        }
    )

    windows_df = build_windows(site_df, "100:100", include_empty=False)

    assert windows_df.shape[0] == 3
    assert windows_df.loc[0, "site_indices"] == [0, 1]
    assert windows_df.loc[1, "site_indices"] == [2]
    assert windows_df.loc[2, "site_indices"] == [3]


def test_build_windows_include_empty():
    site_df = pd.DataFrame(
        {
            "site_index": [0, 1, 2],
            "chromosome": ["chr1", "chr1", "chr1"],
            "position": [100, 150, 320],
        }
    )

    windows_df = build_windows(site_df, "100:100", include_empty=True)
    assert windows_df.shape[0] == 3
    assert windows_df.loc[1, "n_sites"] == 0


def test_build_windows_genome_wide_mode():
    site_df = pd.DataFrame(
        {
            "site_index": [0, 1],
            "chromosome": ["chr1", "chr2"],
            "position": [10, 20],
        }
    )

    windows_df = build_windows(site_df, "0")
    assert windows_df.shape[0] == 1
    assert windows_df.loc[0, "chromosome"] == "all"
    assert windows_df.loc[0, "n_sites"] == 2
