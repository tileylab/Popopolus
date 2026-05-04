import numpy as np

def apply_missing_imputation(
    genotype_dat,
    method="skip",
    tax_list=None,
    ind_map=None,
    mean_strategy="site_mean",
    random_strategy="within_ploidy",
    fill_all_missing=0,
    in_place=False,
    seed=None,
    site_df=None,
    drop_if="any",
    return_removed=False,
):
    """
    Dispatch missing-value imputation by keyword.

    Supported methods:
        - "skip": no imputation (returns data unchanged)
        - "mean": average_missing using mean_strategy
        - "popmean": alias for ploidy-aware mean imputation
        - "learn": learn_impute_missing (currently falls back to mean)
        - "random": randomly_impute_missing using random_strategy
        - "zeros": zero_impute_missing
        - "remove": remove_missing
    """
    method_key = str(method).strip().lower()

    if method_key == "skip":
        return genotype_dat

    if method_key in ("mean", "popmean"):
        strategy = mean_strategy
        if method_key == "popmean" and mean_strategy == "site_mean":
            strategy = "within_ploidy"
        return average_missing(
            genotype_dat,
            tax_list=tax_list,
            ind_map=ind_map,
            strategy=strategy,
            fill_all_missing=fill_all_missing,
            in_place=in_place
        )

    if method_key == "random":
        return randomly_impute_missing(
            genotype_dat,
            tax_list=tax_list,
            ind_map=ind_map,
            strategy=random_strategy,
            fill_all_missing=fill_all_missing,
            in_place=in_place,
            seed=seed
        )
        
    if method_key == "learn":
        return learn_impute_missing(
            genotype_dat,
            tax_list=tax_list,
            ind_map=ind_map,
            fallback_strategy=mean_strategy,
            fill_all_missing=fill_all_missing,
            in_place=in_place
        )
    
    if method_key == "zeros":
        return zero_impute_missing(
            genotype_dat,
            in_place=in_place
        )

    if method_key == "remove":
        return remove_missing(
            genotype_dat,
            site_df=site_df,
            missing_value=-1,
            drop_if=drop_if,
            in_place=in_place,
            return_removed=return_removed
        )

    raise ValueError(f"Unsupported imputation method: {method}")

def average_missing(
    genotype_dat,
    tax_list=None,
    ind_map=None,
    strategy="site_mean",
    fill_all_missing=0,
    in_place=False,
):
    """
    Impute missing genotype dosage states using per-site mean dosage.

    Parameters:
        genotype_dat (np.ndarray): layered genotype array from get_ind_genotypes
            with shape (n_layers, n_sites, n_taxa); layer 0 stores genotype dosage
            values where missing states are coded as -1.
        tax_list (list|None): sample order corresponding to genotype_dat columns.
            Required for ploidy-aware strategies.
        ind_map (dict|None): metadata dict containing per-sample ploidy values.
            Required for ploidy-aware strategies.
        strategy (str): imputation strategy.
            - "site_mean": per-site mean dosage across all samples.
            - "within_ploidy": per-site mean dosage within each ploidy class,
              with global scaled fallback when a ploidy class has no observed values.
            - "scaled": estimate per-site allele frequency across all observed
              samples and scale expected dosage by each sample's ploidy.
        fill_all_missing (int): fill value for sites where all individuals are missing.
            Must be >= 0. Default is 0.
        in_place (bool): if True, modify genotype_dat in place. Otherwise, return
            an imputed copy.

    Returns:
        np.ndarray: genotype_dat with imputed layer-0 missing dosage states.
    """
    if fill_all_missing < 0:
        raise ValueError("fill_all_missing must be >= 0")

    if strategy not in ("site_mean", "within_ploidy", "scaled"):
        raise ValueError("strategy must be one of: site_mean, within_ploidy, scaled")

    if genotype_dat.ndim != 3 or genotype_dat.shape[0] < 1:
        raise ValueError("genotype_dat must have shape (n_layers, n_sites, n_taxa)")

    dat = genotype_dat if in_place else genotype_dat.copy()

    genotype_layer = dat[0]
    missing_mask = genotype_layer == -1
    if not np.any(missing_mask):
        return dat

    ploidies = None
    if strategy in ("within_ploidy", "scaled"):
        if tax_list is None or ind_map is None:
            raise ValueError("tax_list and ind_map are required for ploidy-aware imputation")
        if len(tax_list) != genotype_layer.shape[1]:
            raise ValueError("tax_list length must match genotype_dat n_taxa dimension")
        ploidies = np.array([int(ind_map[tax]["ploidy"]) for tax in tax_list], dtype=np.int16)
        if np.any(ploidies <= 0):
            raise ValueError("All ploidy values must be positive integers")

    # Compute site-wise mean dosage from observed states only.
    if strategy == "site_mean":
        observed = np.where(missing_mask, 0.0, genotype_layer.astype(np.float64))
        observed_counts = np.sum(~missing_mask, axis=1)
        observed_sums = np.sum(observed, axis=1)
        max_dosage = int(np.max(genotype_layer[~missing_mask])) if np.any(~missing_mask) else int(fill_all_missing)

        site_means = np.divide(
            observed_sums,
            observed_counts,
            out=np.full(observed_sums.shape, float(fill_all_missing), dtype=np.float64),
            where=observed_counts > 0,
        )

        site_impute_values = np.clip(np.rint(site_means), 0, max_dosage).astype(genotype_layer.dtype)
        genotype_layer = np.where(missing_mask, site_impute_values[:, None], genotype_layer)

    elif strategy == "scaled":
        out_layer = genotype_layer.astype(np.int32, copy=True)
        for site_idx in range(out_layer.shape[0]):
            site_vals = out_layer[site_idx]
            site_missing = site_vals == -1
            if not np.any(site_missing):
                continue

            site_observed = ~site_missing
            if np.any(site_observed):
                p = site_vals[site_observed].sum() / ploidies[site_observed].sum()
            else:
                p = None

            missing_indices = np.where(site_missing)[0]
            for j in missing_indices:
                if p is None:
                    imputed = int(fill_all_missing)
                else:
                    imputed = int(np.rint(p * ploidies[j]))
                out_layer[site_idx, j] = int(np.clip(imputed, 0, ploidies[j]))

        genotype_layer = out_layer.astype(genotype_layer.dtype, copy=False)

    else:  # within_ploidy
        out_layer = genotype_layer.astype(np.int32, copy=True)
        unique_ploidies = np.unique(ploidies)
        for site_idx in range(out_layer.shape[0]):
            site_vals = out_layer[site_idx]
            site_missing = site_vals == -1
            if not np.any(site_missing):
                continue

            site_observed = ~site_missing
            if np.any(site_observed):
                global_p = site_vals[site_observed].sum() / ploidies[site_observed].sum()
            else:
                global_p = None

            for ploidy in unique_ploidies:
                group_idx = np.where(ploidies == ploidy)[0]
                group_missing_idx = group_idx[site_missing[group_idx]]
                if len(group_missing_idx) == 0:
                    continue

                group_observed_idx = group_idx[site_observed[group_idx]]
                if len(group_observed_idx) > 0:
                    group_mean = np.mean(site_vals[group_observed_idx])
                    group_imputed = int(np.rint(group_mean))
                elif global_p is not None:
                    group_imputed = int(np.rint(global_p * ploidy))
                else:
                    group_imputed = int(fill_all_missing)

                group_imputed = int(np.clip(group_imputed, 0, ploidy))
                out_layer[site_idx, group_missing_idx] = group_imputed

        genotype_layer = out_layer.astype(genotype_layer.dtype, copy=False)

    dat[0] = genotype_layer.astype(dat[0].dtype, copy=False)
    return dat

def randomly_impute_missing(
    genotype_dat,
    tax_list=None,
    ind_map=None,
    strategy="within_ploidy",
    fill_all_missing=0,
    in_place=False,
    seed=None,
):
    """
    Impute missing genotype dosage states using proportional random draws.

    Parameters:
        genotype_dat (np.ndarray): layered genotype array from get_ind_genotypes
            with shape (n_layers, n_sites, n_taxa); layer 0 stores genotype dosage
            values where missing states are coded as -1.
        tax_list (list|None): sample order corresponding to genotype_dat columns.
            Required for ploidy-aware strategies.
        ind_map (dict|None): metadata dict containing per-sample ploidy values.
            Required for ploidy-aware strategies.
        strategy (str): imputation strategy.
            - "site_weighted": draw from observed site dosage frequencies across
              all samples.
            - "within_ploidy": draw from observed site dosage frequencies within
              each ploidy class; falls back to scaled if no class observations.
            - "scaled": estimate site allele frequency across all observations,
              then draw dosage from Binomial(ploidy, p) for each missing sample.
        fill_all_missing (int): fill value for sites where all individuals are missing.
            Must be >= 0. Default is 0.
        in_place (bool): if True, modify genotype_dat in place. Otherwise, return
            an imputed copy.
        seed (int|None): seed for reproducible random draws.

    Returns:
        np.ndarray: genotype_dat with imputed layer-0 missing dosage states.
    """
    if fill_all_missing < 0:
        raise ValueError("fill_all_missing must be >= 0")

    if strategy not in ("site_weighted", "within_ploidy", "scaled"):
        raise ValueError("strategy must be one of: site_weighted, within_ploidy, scaled")

    if genotype_dat.ndim != 3 or genotype_dat.shape[0] < 1:
        raise ValueError("genotype_dat must have shape (n_layers, n_sites, n_taxa)")

    dat = genotype_dat if in_place else genotype_dat.copy()

    genotype_layer = dat[0]
    missing_mask = genotype_layer == -1
    if not np.any(missing_mask):
        return dat

    rng = np.random.default_rng(seed)

    ploidies = None
    if strategy in ("within_ploidy", "scaled"):
        if tax_list is None or ind_map is None:
            raise ValueError("tax_list and ind_map are required for ploidy-aware imputation")
        if len(tax_list) != genotype_layer.shape[1]:
            raise ValueError("tax_list length must match genotype_dat n_taxa dimension")
        ploidies = np.array([int(ind_map[tax]["ploidy"]) for tax in tax_list], dtype=np.int16)
        if np.any(ploidies <= 0):
            raise ValueError("All ploidy values must be positive integers")

    out_layer = genotype_layer.astype(np.int32, copy=True)

    for site_idx in range(out_layer.shape[0]):
        site_vals = out_layer[site_idx]
        site_missing = site_vals == -1
        if not np.any(site_missing):
            continue

        site_observed = ~site_missing
        observed_vals = site_vals[site_observed]

        if strategy == "site_weighted":
            if observed_vals.size == 0:
                out_layer[site_idx, site_missing] = int(fill_all_missing)
                continue

            uniq, counts = np.unique(observed_vals, return_counts=True)
            probs = counts / counts.sum()
            draws = rng.choice(uniq, size=int(site_missing.sum()), p=probs)
            out_layer[site_idx, site_missing] = draws.astype(out_layer.dtype)
            continue

        # Ploidy-aware strategies below.
        if np.any(site_observed):
            global_p = observed_vals.sum() / ploidies[site_observed].sum()
        else:
            global_p = None

        if strategy == "scaled":
            missing_indices = np.where(site_missing)[0]
            for j in missing_indices:
                if global_p is None:
                    imputed = int(fill_all_missing)
                else:
                    imputed = int(rng.binomial(int(ploidies[j]), float(global_p)))
                out_layer[site_idx, j] = int(np.clip(imputed, 0, int(ploidies[j])))
            continue

        # within_ploidy
        unique_ploidies = np.unique(ploidies)
        for ploidy in unique_ploidies:
            group_idx = np.where(ploidies == ploidy)[0]
            group_missing_idx = group_idx[site_missing[group_idx]]
            if len(group_missing_idx) == 0:
                continue

            group_observed_idx = group_idx[site_observed[group_idx]]
            if len(group_observed_idx) > 0:
                group_vals = site_vals[group_observed_idx]
                uniq, counts = np.unique(group_vals, return_counts=True)
                probs = counts / counts.sum()
                draws = rng.choice(uniq, size=len(group_missing_idx), p=probs)
                draws = np.clip(draws, 0, int(ploidy))
                out_layer[site_idx, group_missing_idx] = draws.astype(out_layer.dtype)
            elif global_p is not None:
                draws = rng.binomial(int(ploidy), float(global_p), size=len(group_missing_idx))
                out_layer[site_idx, group_missing_idx] = draws.astype(out_layer.dtype)
            else:
                out_layer[site_idx, group_missing_idx] = int(fill_all_missing)

    dat[0] = out_layer.astype(dat[0].dtype, copy=False)
    return dat

def zero_impute_missing(
        genotype_dat,
        in_place=False
):
    """
    Impute missing genotype dosage states by simply assigning them a value of 0.

    Parameters:
        genotype_dat (np.ndarray): layered genotype array from get_ind_genotypes
            with shape (n_layers, n_sites, n_taxa); layer 0 stores genotype dosage
            values where missing states are coded as -1.
        in_place (bool): if True, modify genotype_dat in place. Otherwise, return
            an imputed copy.
    Returns:
        np.ndarray: genotype_dat with imputed layer-0 missing dosage states.
    """

    if genotype_dat.ndim != 3 or genotype_dat.shape[0] < 1:
        raise ValueError("genotype_dat must have shape (n_layers, n_sites, n_taxa)")

    dat = genotype_dat if in_place else genotype_dat.copy()

    genotype_layer = dat[0]
    missing_mask = genotype_layer == -1
    if np.any(missing_mask):
        dat[0] = np.where(missing_mask, 0, genotype_layer).astype(dat[0].dtype, copy=False)

    return dat

def learn_impute_missing(
    genotype_dat,
    tax_list=None,
    ind_map=None,
    fallback_strategy="site_mean",
    fill_all_missing=0,
    in_place=False,
):
    """
    Placeholder for machine learning-based imputation.

    The ML model is not implemented yet, so this helper currently delegates to
    average_missing using the provided fallback strategy.
    """
    return average_missing(
        genotype_dat,
        tax_list=tax_list,
        ind_map=ind_map,
        strategy=fallback_strategy,
        fill_all_missing=fill_all_missing,
        in_place=in_place,
    )

def remove_missing(
    genotype_dat,
    site_df=None,
    missing_value=-1,
    drop_if="any",
    in_place=False,
    return_removed=False,
):
    """
    Remove sites containing missing genotype states in layer 0.

    Parameters:
        genotype_dat (np.ndarray): layered genotype array with shape
            (n_layers, n_sites, n_taxa); layer 0 stores genotype dosage states.
        site_df (pd.DataFrame|None): optional site metadata table aligned to
            genotype_dat sites. If provided, it is filtered and site_index is
            reset to 0..n_retained-1.
        missing_value (int): value encoding missing genotype states. Default -1.
        drop_if (str): "any" drops a site if any sample is missing;
            "all" drops only if all samples are missing.
        in_place (bool): if True, update genotype_dat in place by shrinking the
            site axis where possible. If False, return a filtered copy.
        return_removed (bool): if True, also return removed site indices from
            the original coordinate space.

    Returns:
        np.ndarray or tuple:
            - genotype_dat_filtered
            - (genotype_dat_filtered, site_df_filtered) when site_df is provided
            - optionally appends removed_site_indices if return_removed=True
    """
    if drop_if not in ("any", "all"):
        raise ValueError("drop_if must be one of: any, all")

    if genotype_dat.ndim != 3 or genotype_dat.shape[0] < 1:
        raise ValueError("genotype_dat must have shape (n_layers, n_sites, n_taxa)")

    original_n_sites = genotype_dat.shape[1]
    genotype_layer = genotype_dat[0]
    missing_mask = genotype_layer == missing_value

    if drop_if == "any":
        drop_site_mask = np.any(missing_mask, axis=1)
    else:
        drop_site_mask = np.all(missing_mask, axis=1)

    keep_site_mask = ~drop_site_mask
    removed_site_indices = np.where(drop_site_mask)[0].astype(np.int64)

    filtered = genotype_dat[:, keep_site_mask, :]
    if in_place:
        # Replace contents reference for caller-style in-place semantics.
        genotype_dat = filtered
        filtered = genotype_dat

    if site_df is None:
        if return_removed:
            return filtered, removed_site_indices
        return filtered

    if len(site_df) != original_n_sites:
        raise ValueError("site_df row count must match genotype_dat n_sites")

    site_df_filtered = site_df.loc[keep_site_mask].copy().reset_index(drop=True)
    site_df_filtered["site_index"] = np.arange(site_df_filtered.shape[0], dtype=np.int32)

    if return_removed:
        return filtered, site_df_filtered, removed_site_indices
    return filtered, site_df_filtered