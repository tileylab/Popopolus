import numpy as np
import pandas as pd

from popopolus.utils import assign_populations


def _population_chromosome_counts(ind_map):
    """Return total sampled chromosomes per population based on ploidy metadata."""
    populations = assign_populations(ind_map)
    chromosome_counts = {}
    for pop, individuals in populations.items():
        chromosome_counts[pop] = int(sum(int(ind_map[ind]["ploidy"]) for ind in individuals))
    return chromosome_counts


def _sample_population_subset(individuals, ind_map, target_chromosomes, rng, require_exact=True):
    """
    Sample a subset of individuals with chromosome total equal to target if possible.

    If require_exact is False and exact matching is impossible, the largest total <= target is used.
    """
    shuffled = list(individuals)
    rng.shuffle(shuffled)

    reachable = {0: []}
    for ind in shuffled:
        ploidy = int(ind_map[ind]["ploidy"])
        next_reachable = dict(reachable)
        for current_sum, selected in reachable.items():
            new_sum = current_sum + ploidy
            if new_sum <= target_chromosomes and new_sum not in next_reachable:
                next_reachable[new_sum] = selected + [ind]
        reachable = next_reachable

    if target_chromosomes in reachable:
        chosen_sum = target_chromosomes
    elif require_exact:
        raise ValueError(
            f"Could not find an exact chromosome subset of {target_chromosomes} "
            f"from ploidies {[int(ind_map[ind]['ploidy']) for ind in individuals]}"
        )
    else:
        chosen_sum = max(reachable.keys())

    return reachable[chosen_sum], chosen_sum


def rarefy_genotype_dataset(
    genotype_dat,
    tax_list,
    ind_map,
    n_replicates=1,
    target_chromosomes=None,
    seed=None,
    require_exact=True,
):
    """
    Rarefy a genotype dataset by population so each population has the same chromosome count.

    Parameters:
        genotype_dat (np.ndarray): data array with shape (n_layers, n_sites, n_taxa)
        tax_list (list): sample identifiers aligned to genotype_dat columns
        ind_map (dict): individual metadata with population and ploidy keys
        n_replicates (int): number of independent rarefaction replicates
        target_chromosomes (int|None): chromosome target per population; if None, uses the minimum across populations
        seed (int|None): random seed for reproducible sampling
        require_exact (bool): require exact target chromosome count per population

    Returns:
        replicate_datasets (list[dict]): one entry per replicate with tax_list, ind_map, genotype_dat
        rarefaction_df (pd.DataFrame): per-population sampling details for each replicate
    """
    if n_replicates < 1:
        raise ValueError("n_replicates must be >= 1")

    tax_to_index = {tax: idx for idx, tax in enumerate(tax_list)}
    populations = assign_populations(ind_map)
    chromosome_counts = _population_chromosome_counts(ind_map)

    if target_chromosomes is None:
        target_chromosomes = min(chromosome_counts.values())

    if target_chromosomes <= 0:
        raise ValueError("target_chromosomes must be > 0")

    if require_exact and any(count < target_chromosomes for count in chromosome_counts.values()):
        raise ValueError("target_chromosomes cannot exceed the chromosome count in any population")

    rng = np.random.default_rng(seed)
    replicate_datasets = []
    rarefaction_rows = []

    for replicate in range(1, n_replicates + 1):
        selected_individuals = []

        for population, individuals in populations.items():
            chosen, chosen_chromosomes = _sample_population_subset(
                individuals=individuals,
                ind_map=ind_map,
                target_chromosomes=target_chromosomes,
                rng=rng,
                require_exact=require_exact,
            )
            selected_individuals.extend(chosen)
            rarefaction_rows.append(
                {
                    "replicate": replicate,
                    "population": population,
                    "target_chromosomes": target_chromosomes,
                    "sampled_chromosomes": chosen_chromosomes,
                    "sampled_individuals": len(chosen),
                    "individual_ids": ",".join(chosen),
                }
            )

        selected_taxa = [tax for tax in tax_list if tax in selected_individuals]
        selected_indices = [tax_to_index[tax] for tax in selected_taxa]
        genotype_subset = genotype_dat[:, :, selected_indices]
        ind_map_subset = {tax: ind_map[tax] for tax in selected_taxa}

        replicate_datasets.append(
            {
                "replicate": replicate,
                "tax_list": selected_taxa,
                "ind_map": ind_map_subset,
                "genotype_dat": genotype_subset,
            }
        )

    rarefaction_df = pd.DataFrame(rarefaction_rows)
    return replicate_datasets, rarefaction_df


def bootstrap_genotype_dataset(genotype_dat, n_bootstraps=100, seed=None):
    """
    Create bootstrap replicates by resampling sites with replacement.

    Parameters:
        genotype_dat (np.ndarray): data array with shape (n_layers, n_sites, n_taxa)
        n_bootstraps (int): number of bootstrap replicates
        seed (int|None): random seed for reproducible sampling

    Returns:
        bootstrap_datasets (list[dict]): one entry per bootstrap replicate
    """
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be >= 1")

    n_sites = genotype_dat.shape[1]
    if n_sites < 1:
        raise ValueError("genotype_dat must contain at least one site")

    rng = np.random.default_rng(seed)
    bootstrap_datasets = []

    for bootstrap in range(1, n_bootstraps + 1):
        site_indices = rng.integers(0, n_sites, size=n_sites)
        genotype_bootstrap = genotype_dat[:, site_indices, :]
        bootstrap_datasets.append(
            {
                "bootstrap": bootstrap,
                "site_indices": site_indices,
                "genotype_dat": genotype_bootstrap,
            }
        )

    return bootstrap_datasets


def summarize_bootstrap_theta(theta_bootstrap_df):
    """
    Summarize bootstrap theta estimates by replicate and population.

    Returns mean, SD, and percentile 95% confidence intervals.
    """
    required_cols = {
        "replicate",
        "population",
        "n_individuals",
        "n_chromosomes",
        "theta_wattersons",
        "theta_pi",
        "tajima_D",
    }
    missing = required_cols.difference(theta_bootstrap_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for bootstrap summary: {sorted(missing)}")

    group_cols = ["replicate", "population"]
    optional_group_cols = ["window_id", "chromosome", "start", "end"]
    for col in optional_group_cols:
        if col in theta_bootstrap_df.columns:
            group_cols.append(col)

    grouped = theta_bootstrap_df.groupby(group_cols, as_index=False)
    summary = grouped.agg(
        n_bootstraps=("bootstrap", "nunique"),
        n_individuals_mean=("n_individuals", "mean"),
        n_chromosomes_mean=("n_chromosomes", "mean"),
        theta_wattersons_mean=("theta_wattersons", "mean"),
        theta_wattersons_sd=("theta_wattersons", "std"),
        theta_pi_mean=("theta_pi", "mean"),
        theta_pi_sd=("theta_pi", "std"),
        tajima_D_mean=("tajima_D", "mean"),
        tajima_D_sd=("tajima_D", "std"),
    )

    ci_rows = []
    for keys, df_sub in theta_bootstrap_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(
            {
                "theta_wattersons_ci_lower": df_sub["theta_wattersons"].quantile(0.025),
                "theta_wattersons_ci_upper": df_sub["theta_wattersons"].quantile(0.975),
                "theta_pi_ci_lower": df_sub["theta_pi"].quantile(0.025),
                "theta_pi_ci_upper": df_sub["theta_pi"].quantile(0.975),
                "tajima_D_ci_lower": df_sub["tajima_D"].quantile(0.025),
                "tajima_D_ci_upper": df_sub["tajima_D"].quantile(0.975),
            }
        )
        ci_rows.append(row)

    ci_df = pd.DataFrame(ci_rows)
    summary = summary.merge(ci_df, on=group_cols, how="left")
    return summary
