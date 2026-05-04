import pandas as pd


def parse_interval_spec(interval):
    """
    Parse an interval specification string into (window_size, step_size).

    Accepted values:
        - "0" or 0: genome-wide (no windowing), returns None
        - "<window>:<step>": returns tuple of positive ints
    """
    if interval in (0, "0", None):
        return None

    if not isinstance(interval, str) or ":" not in interval:
        raise ValueError("interval must be '0' or '<window>:<step>'")

    window_str, step_str = interval.split(":", 1)
    window_size = int(window_str)
    step_size = int(step_str)

    if window_size <= 0 or step_size <= 0:
        raise ValueError("window and step must both be > 0")

    return window_size, step_size


def build_windows(site_df, interval, include_empty=False):
    """
    Build per-chromosome sliding windows and assign site indices.

    Parameters:
        site_df (pd.DataFrame): must contain columns chromosome, position, site_index
        interval (str|int): "0" for genome-wide, else "window:step"
        include_empty (bool): include windows with zero sites

    Returns:
        windows_df (pd.DataFrame): columns include window_id, chromosome, start, end, n_sites, site_indices
    """
    parsed = parse_interval_spec(interval)
    if parsed is None:
        return pd.DataFrame(
            [
                {
                    "window_id": 1,
                    "chromosome": "all",
                    "start": None,
                    "end": None,
                    "n_sites": int(site_df.shape[0]),
                    "site_indices": site_df["site_index"].astype(int).tolist(),
                }
            ]
        )

    required_cols = {"chromosome", "position", "site_index"}
    missing = required_cols.difference(site_df.columns)
    if missing:
        raise ValueError(f"site_df is missing required columns: {sorted(missing)}")

    window_size, step_size = parsed
    windows = []
    window_id = 1

    for chromosome, chrom_df in site_df.groupby("chromosome"):
        chrom_df = chrom_df.sort_values("position")
        chrom_start = int(chrom_df["position"].min())
        chrom_end = int(chrom_df["position"].max())

        start = chrom_start
        while start <= chrom_end:
            end = start + window_size - 1
            in_window = chrom_df[(chrom_df["position"] >= start) & (chrom_df["position"] <= end)]
            site_indices = in_window["site_index"].astype(int).tolist()

            if include_empty or len(site_indices) > 0:
                windows.append(
                    {
                        "window_id": window_id,
                        "chromosome": chromosome,
                        "start": start,
                        "end": end,
                        "n_sites": len(site_indices),
                        "site_indices": site_indices,
                    }
                )
                window_id += 1

            start += step_size

    return pd.DataFrame(windows)


def subset_genotype_by_sites(genotype_dat, site_indices):
    """Return a genotype array subset containing only the provided site indices."""
    return genotype_dat[:, site_indices, :]
