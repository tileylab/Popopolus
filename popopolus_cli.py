#!/usr/bin/env python

"""
-------------------------------------------------------------------------------------------------------------------------------------------
MIT License

Copyright (c) 2024 George P. Tiley

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Contact: gptiley@ncsu.edu
-------------------------------------------------------------------------------------------------------------------------------------------

popopolus uses a command-line interface to interact with the popopolus package. A successful installation should allow you to access all functions and their help with:

popopolus --help
"""

import click
import time
import os
from datetime import datetime
import logging
import pandas as pd

from popopolus.utils import check_dir

#--------------------------------#
# CLI ENTRY POINT
#--------------------------------#

@click.group()
def cli():
    current_time = datetime.now()
    time_string = current_time.strftime('%Y-%m-%d-%H-%M-%S')
    logfile = time_string + '_popopolus.log'
    print(f'Start popopolus at {current_time}')
    logging.basicConfig(
       filename = logfile,
       format='%(asctime)s - %(levelname)s - %(message)s',
       filemode = 'w',
       level = logging.INFO
    )
    print(f'Created logfile {logfile}')


#----------------
# Ploidy Classifier Models (Please ignore the estimate_ploidy functions down below)
#----------------
@cli.command(context_settings={'help_option_names': ['-h','--help']})
@click.argument('sample_sheet',type=str) 
@click.option('-v', '--vcf_file', type=str, default='dummy.vcf', required=True,
              help = 'name of the input vcf file. should not be compressed.'
)
@click.option('-o', '--output_dir', type=str, default='dummy', required=False,
              help = 'name of the directory where . will be a matrix of allele frequencies'
)
def classify_ploidy(sample_sheet, vcf_file, output_dir):
    from popopolus.classify_ploidy.logistic_regression import logistic_regression

    start_time = time.process_time()
    logging.info(f'Begin at {start_time}')
    logging.info(f'Clasifying ploidy for individuals in {sample_sheet} that are present in {vcf_file}')
    if (output_dir != 'dummy'):
        check_dir(output_dir)
        logging.info(f'Matrix of allele frequencies for each individual will be written to: {output_dir}')

    ploidy_results = logistic_regression(sample_sheet, vcf_file, output_dir)
    print(ploidy_results)
    end_time = time.process_time()
    logging.info(f'End at {end_time}')
    compute_time = (end_time - start_time) / 60
    logging.info(f'Total compute time was {compute_time} minutes')

#----------------
# Site frequency spectrum and diversity statistics
#----------------
@cli.command(context_settings={'help_option_names': ['-h','--help']})
@click.argument('sample_sheet',type=str)
@click.option('-v', '--vcf_file', type=str, default='dummy.vcf', required=True,
              help = 'name of the input vcf file. should not be compressed.'
)
@click.option('-i', '--imputation_method', type=str, default='skip', required=False,
              help = 'Missing-data handling method. Options: skip, mean, popmean, learn, random, zeros, remove.'
)
@click.option('-d', '--minimum_depth', type=int, default=10, required=False,
              help = 'The minimum depth of a site to be treated as data'
)
@click.option('-c', '--minimum_count', type=int, default=3, required=False,
              help = 'The minimum count of the minor allele for a site to be treated as data'
)
@click.option('-q', '--minimum_quality', type=int, default=20, required=False,
              help = 'The minimum phred-scaled genotype quality score'
)
@click.option('-f', '--pass_flag', type=str, default='PASS', required=False,
              help = 'Does the VCF have an PASS field or something else to condider like a "."'
)
@click.option('-o', '--output_dir', type=str, default='dummy', required=False,
              help = 'name of the directory where . will be a matrix of allele frequencies'
)
def individual_genotypes(sample_sheet, vcf_file, minimum_depth, minimum_count, minimum_quality, imputation_method, pass_flag, output_dir):
    from popopolus.utils import map_individuals
    from popopolus.utils import check_dir
    from popopolus.utils import get_vcf_dimensions
    from popopolus.calculate_frequencies.calculate_frequencies import get_ind_genotypes
    from popopolus.calculate_frequencies.impute import apply_missing_imputation

    start_time = time.process_time()
    logging.info(f'Begin at {start_time}')
    logging.info(f'Checking all individuals in {sample_sheet} are present in {vcf_file}')
    ind_map = map_individuals(sample_sheet)
    logging.info(f'Checking dimensions of VCF')
    n_sites, n_tax = get_vcf_dimensions(vcf_file, pass_flag, ind_map)
    logging.info(f'Calculating individual allele frequencies from {vcf_file}')
    if (output_dir != 'dummy'):
        check_dir(output_dir)
        logging.info(f'Matrix of allele frequencies for each individual will be written to: {output_dir}')

    # Generate genotype layers and apply user-selected missing-data treatment.
    tax_list, genotype_dat = get_ind_genotypes(
        n_sites,
        n_tax,
        ind_map,
        vcf_file,
        minimum_depth,
        minimum_count,
        minimum_quality,
        pass_flag,
        'dummy',
    )
    try:
        genotype_dat = apply_missing_imputation(
            genotype_dat,
            method=imputation_method,
            tax_list=tax_list,
            ind_map=ind_map,
            in_place=True,
        )
    except ValueError:
        click.echo(f'Warning: Imputation method {imputation_method} is not supported. Skipping allele frequencies.')
        end_time = time.process_time()
        logging.info(f'End at {end_time}')
        compute_time = (end_time - start_time) / 60
        logging.info(f'Total compute time was {compute_time} minutes')
        return

    if (output_dir != 'dummy'):
        for tax_index, tax in enumerate(tax_list):
            output_file = f'{output_dir}/{tax}.txt'
            with open(output_file, 'w') as outfile:
                outfile.write('genotype\tdepth\tgenotype_quality\tpass_filters\n')
                for site_index in range(genotype_dat.shape[1]):
                    outstring = (
                        f'{genotype_dat[0, site_index, tax_index]}\t'
                        f'{genotype_dat[1, site_index, tax_index]}\t'
                        f'{genotype_dat[2, site_index, tax_index]}\t'
                        f'{genotype_dat[3, site_index, tax_index]}\n'
                    )
                    outfile.write(outstring)
    end_time = time.process_time()
    logging.info(f'End at {end_time}')
    compute_time = (end_time - start_time) / 60
    logging.info(f'Total compute time was {compute_time} minutes')

@cli.command(context_settings={'help_option_names': ['-h','--help']})
@click.argument('sample_sheet',type=str)
@click.option('-v', '--vcf_file', type=str, default='dummy.vcf', required=True,
              help = 'name of the input vcf file. should not be compressed.'
)
@click.option('-i', '--imputation_method', type=str, default='skip', required=False,
              help = 'Missing-data handling method. Options: skip, mean, popmean, learn, random, zeros, remove.'
)
@click.option('-d', '--minimum_depth', type=int, default=10, required=False,
              help = 'The minimum depth of a site to be treated as data'
)
@click.option('-c', '--minimum_count', type=int, default=3, required=False,
              help = 'The minimum count of the minor allele for a site to be treated as data'
)
@click.option('-q', '--minimum_quality', type=int, default=20, required=False,
              help = 'The minimum phred-scaled genotype quality score'
)
@click.option('-f', '--pass_flag', type=str, default='PASS', required=False,
              help = 'Does the VCF have an PASS field or something else to condider like a "."'
)
@click.option('-l', '--interval', type=str, default='0', required=False,
              help = 'The window size and step size for calculating theta in the format window:step e.g., 10000:5000. The default of 0 will calculate genome-wide theta.'
)
@click.option('-u', '--folded', type=bool, default=True, required=False,
              help = 'Whether to calculate folded or unfolded site frequency spectrum'
)
@click.option('--rarefy', is_flag=True, default=False,
              help = 'Downsample each population to a common chromosome count before estimating theta.'
)
@click.option('--rarefy_replicates', type=int, default=1, required=False,
              help = 'Number of independent rarefaction replicates to run when --rarefy is set.'
)
@click.option('--rarefy_seed', type=int, default=None, required=False,
              help = 'Random seed for reproducible rarefaction sampling.'
)
@click.option('--rarefy_target_chromosomes', type=int, default=0, required=False,
              help = 'Target chromosomes per population for rarefaction. Default 0 uses the smallest population.'
)
@click.option('--rarefy_relax_exact', is_flag=True, default=False,
              help = 'Allow <= target chromosomes if an exact subset is not possible for a population.'
)
@click.option('--bootstrap_replicates', type=int, default=0, required=False,
              help = 'Bootstrap replicates per rarefied dataset for mean and 95% CI estimation. 0 disables bootstrapping.'
)
@click.option('--bootstrap_seed', type=int, default=None, required=False,
              help = 'Random seed for bootstrap site resampling.'
)
@click.option('-o', '--output_dir', type=str, default='dummy', required=False,
              help = 'name of the directory where . will be a matrix of allele frequencies'
)
def estimate_theta(sample_sheet, vcf_file, minimum_depth, minimum_count, minimum_quality, imputation_method, pass_flag, interval, folded, rarefy, rarefy_replicates, rarefy_seed, rarefy_target_chromosomes, rarefy_relax_exact, bootstrap_replicates, bootstrap_seed, output_dir):
    from popopolus.utils import map_individuals
    from popopolus.utils import check_dir
    from popopolus.utils import get_vcf_dimensions
    from popopolus.calculate_frequencies.calculate_frequencies import get_ind_genotypes
    from popopolus.calculate_frequencies.impute import apply_missing_imputation
    from popopolus.diversity_statistics.theta import estimate_thetas
    from popopolus.sampling.sampling import rarefy_genotype_dataset
    from popopolus.sampling.sampling import bootstrap_genotype_dataset
    from popopolus.sampling.sampling import summarize_bootstrap_theta
    from popopolus.windowing.windowing import parse_interval_spec
    from popopolus.windowing.windowing import build_windows
    from popopolus.windowing.windowing import subset_genotype_by_sites

    def run_theta_for_dataset(genotype_data_in, tax_list_in, ind_map_in, site_df_in, output_dir_in):
        """Run theta globally (interval=0) or per genomic window (interval=window:step)."""
        parsed_interval = parse_interval_spec(interval)
        if parsed_interval is None:
            theta_df_local = estimate_thetas(genotype_data_in, tax_list_in, ind_map_in, interval, folded, output_dir_in)
            return theta_df_local

        windows_df = build_windows(site_df_in, interval, include_empty=False)
        windows_df.to_csv(f'{output_dir_in}/windows.csv', index=False)

        theta_parts = []
        for _, window in windows_df.iterrows():
            site_indices = window['site_indices']
            if len(site_indices) == 0:
                continue

            window_genotype = subset_genotype_by_sites(genotype_data_in, site_indices)
            window_output_dir = f"{output_dir_in}/window_{int(window['window_id'])}"
            os.makedirs(window_output_dir, exist_ok=True)
            window_theta_df = estimate_thetas(window_genotype, tax_list_in, ind_map_in, '0', folded, window_output_dir)
            window_theta_df.insert(0, 'window_id', int(window['window_id']))
            window_theta_df.insert(1, 'chromosome', window['chromosome'])
            window_theta_df.insert(2, 'start', int(window['start']))
            window_theta_df.insert(3, 'end', int(window['end']))
            window_theta_df.insert(4, 'n_sites_window', int(window['n_sites']))
            theta_parts.append(window_theta_df)

        if len(theta_parts) == 0:
            return pd.DataFrame()
        return pd.concat(theta_parts, ignore_index=True)

    start_time = time.process_time()
    logging.info(f'Begin at {start_time}')
    logging.info(f'Checking all individuals in {sample_sheet} are present in {vcf_file}')
    ind_map = map_individuals(sample_sheet)
    logging.info(f'Checking dimensions of VCF')
    n_sites, n_tax = get_vcf_dimensions(vcf_file, pass_flag, ind_map)
    logging.info(f'Calculating individual allele frequencies from {vcf_file}')
    if (output_dir != 'dummy'):
        check_dir(output_dir)
        logging.info(f'Matrix of allele frequencies for each individual will be written to: {output_dir}')
    else:
        os.makedirs(output_dir, exist_ok=True)

    tax_list, genotype_dat, site_df = get_ind_genotypes(
        n_sites,
        n_tax,
        ind_map,
        vcf_file,
        minimum_depth,
        minimum_count,
        minimum_quality,
        pass_flag,
        output_dir,
        return_site_data=True,
    )

    try:
        imputation_result = apply_missing_imputation(
            genotype_dat,
            method=imputation_method,
            tax_list=tax_list,
            ind_map=ind_map,
            site_df=site_df,
            in_place=True,
        )
        if str(imputation_method).strip().lower() == 'remove':
            genotype_dat, site_df = imputation_result
        else:
            genotype_dat = imputation_result
    except ValueError:
        click.echo(f'Warning: Imputation method {imputation_method} is not supported.')
        end_time = time.process_time()
        logging.info(f'End at {end_time}')
        compute_time = (end_time - start_time) / 60
        logging.info(f'Total compute time was {compute_time} minutes')
        return

    site_df.to_csv(f'{output_dir}/site_coordinates.csv', index=False)
    logging.info(f'Wrote site coordinates for windowing: {output_dir}/site_coordinates.csv')
    if rarefy:
        target = None if rarefy_target_chromosomes == 0 else rarefy_target_chromosomes
        replicate_datasets, rarefaction_df = rarefy_genotype_dataset(
            genotype_dat=genotype_dat,
            tax_list=tax_list,
            ind_map=ind_map,
            n_replicates=rarefy_replicates,
            target_chromosomes=target,
            seed=rarefy_seed,
            require_exact=(not rarefy_relax_exact),
        )

        if bootstrap_replicates > 0:
            theta_bootstrap_rows = []
            for replicate_data in replicate_datasets:
                rep = replicate_data['replicate']
                rep_dir = f'{output_dir}/rarefaction_replicate_{rep}'
                os.makedirs(rep_dir, exist_ok=True)

                rep_bootstrap_seed = None if bootstrap_seed is None else bootstrap_seed + rep
                bootstrap_datasets = bootstrap_genotype_dataset(
                    genotype_dat=replicate_data['genotype_dat'],
                    n_bootstraps=bootstrap_replicates,
                    seed=rep_bootstrap_seed,
                )

                for bootstrap_data in bootstrap_datasets:
                    boot = bootstrap_data['bootstrap']
                    boot_dir = f'{rep_dir}/bootstrap_{boot}'
                    os.makedirs(boot_dir, exist_ok=True)
                    bootstrap_site_df = site_df.iloc[bootstrap_data['site_indices']].copy().reset_index(drop=True)
                    bootstrap_site_df['site_index'] = bootstrap_site_df.index.astype(int)

                    theta_df = run_theta_for_dataset(
                        bootstrap_data['genotype_dat'],
                        replicate_data['tax_list'],
                        replicate_data['ind_map'],
                        bootstrap_site_df,
                        boot_dir,
                    )
                    theta_df.insert(0, 'bootstrap', boot)
                    theta_df.insert(0, 'replicate', rep)
                    theta_bootstrap_rows.append(theta_df)

            theta_bootstrap_df = pd.concat(theta_bootstrap_rows, ignore_index=True)
            theta_bootstrap_df.to_csv(f'{output_dir}/theta_rarefied_bootstrap_replicates.csv', index=False)

            theta_bootstrap_summary_df = summarize_bootstrap_theta(theta_bootstrap_df)
            theta_bootstrap_summary_df.to_csv(f'{output_dir}/theta_rarefied_bootstrap_summary.csv', index=False)
            rarefaction_df.to_csv(f'{output_dir}/rarefaction_samples.csv', index=False)
            print(theta_bootstrap_summary_df)
        else:
            theta_replicates = []
            for replicate_data in replicate_datasets:
                rep = replicate_data['replicate']
                rep_dir = f'{output_dir}/rarefaction_replicate_{rep}'
                os.makedirs(rep_dir, exist_ok=True)
                theta_df = run_theta_for_dataset(
                    replicate_data['genotype_dat'],
                    replicate_data['tax_list'],
                    replicate_data['ind_map'],
                    site_df,
                    rep_dir,
                )
                theta_df.insert(0, 'replicate', rep)
                theta_replicates.append(theta_df)

            theta_replicates_df = pd.concat(theta_replicates, ignore_index=True)
            theta_replicates_df.to_csv(f'{output_dir}/theta_rarefied_replicates.csv', index=False)

            summary_group_cols = ['population']
            for col in ['window_id', 'chromosome', 'start', 'end']:
                if col in theta_replicates_df.columns:
                    summary_group_cols.append(col)

            theta_summary_df = theta_replicates_df.groupby(summary_group_cols, as_index=False).agg(
                n_replicates=('replicate', 'nunique'),
                n_individuals_mean=('n_individuals', 'mean'),
                n_chromosomes_mean=('n_chromosomes', 'mean'),
                theta_wattersons_mean=('theta_wattersons', 'mean'),
                theta_wattersons_sd=('theta_wattersons', 'std'),
                theta_pi_mean=('theta_pi', 'mean'),
                theta_pi_sd=('theta_pi', 'std'),
                tajima_D_mean=('tajima_D', 'mean'),
                tajima_D_sd=('tajima_D', 'std'),
            )
            theta_summary_df.to_csv(f'{output_dir}/theta_rarefied_summary.csv', index=False)
            rarefaction_df.to_csv(f'{output_dir}/rarefaction_samples.csv', index=False)
            print(theta_summary_df)
    else:
        theta_df = run_theta_for_dataset(genotype_dat, tax_list, ind_map, site_df, output_dir)
        theta_df.to_csv(f'{output_dir}/theta.csv', index=False)
        print(theta_df)
    end_time = time.process_time()
    logging.info(f'End at {end_time}')
    compute_time = (end_time - start_time) / 60
    logging.info(f'Total compute time was {compute_time} minutes')

#----------------
# Ploidy estimation
#----------------
@cli.command(context_settings={'help_option_names': ['-h','--help']})
@click.argument('sample_sheet',type=str)
@click.option('-v', '--vcf_file', type=str, default='dummy.vcf', required=True,
              help = 'name of the input vcf file. should not be compressed.'
)
@click.option('-i', '--imputation_method', type=str, default='skip', required=False,
              help = 'Missing-data handling for this command. Only skip is currently supported.'
)
@click.option('-d', '--minimum_depth', type=int, default=10, required=False,
              help = 'The minimum depth of a site to be treated as data'
)
@click.option('-c', '--minimum_count', type=int, default=3, required=False,
              help = 'The minimum count of the minor allele for a site to be treated as data'
)
@click.option('-q', '--minimum_quality', type=int, default=20, required=False,
              help = 'The minimum phred-scaled genotype quality score'
)
@click.option('-f', '--pass_flag', type=str, default='PASS', required=False,
              help = 'Does the VCF have an PASS field or something else to condider like a "."'
)
@click.option('-o', '--output_dir', type=str, default='dummy', required=False,
              help = 'name of the directory where . will be a matrix of allele frequencies'
)
def individual_ab(sample_sheet, vcf_file, minimum_depth, minimum_count, minimum_quality, imputation_method, pass_flag, output_dir):
    from popopolus.utils import map_individuals
    from popopolus.utils import check_dir
    from popopolus.utils import get_vcf_dimensions
    from popopolus.calculate_frequencies.calculate_frequencies import get_ind_ab

    start_time = time.process_time()
    logging.info(f'Begin at {start_time}')
    logging.info(f'Checking all individuals in {sample_sheet} are present in {vcf_file}')
    ind_map = map_individuals(sample_sheet)
    logging.info(f'Checking dimensions of VCF')
    n_sites, n_tax = get_vcf_dimensions(vcf_file, pass_flag, ind_map)
    logging.info(f'Calculating individual allele frequencies from {vcf_file}')
    if str(imputation_method).strip().lower() == 'skip':
        if (output_dir != 'dummy'):
            check_dir(output_dir)
            logging.info(f'Matrix of allele frequencies for each individual will be written to: {output_dir}')
        get_ind_ab(n_sites, n_tax, ind_map, vcf_file, minimum_depth, minimum_count, minimum_quality, pass_flag, output_dir)
        
    else:
        click.echo(f'Warning: Imputation method {imputation_method} is not supported. Skipping allele frequencies.')
    end_time = time.process_time()
    logging.info(f'End at {end_time}')
    compute_time = (end_time - start_time) / 60
    logging.info(f'Total compute time was {compute_time} minutes')



@cli.command(context_settings={'help_option_names': ['-h','--help']})
@click.argument('sample_sheet',type=str)
@click.option('-v', '--vcf_file', type=str, default='dummy.vcf', required=True,
              help = 'name of the input vcf file. should not be compressed.'
)
@click.option('-i', '--imputation_method', type=str, default='skip', required=False,
              help = 'Missing-data handling for this command. Only skip is currently supported.'
)
@click.option('-d', '--minimum_depth', type=int, default=10, required=False,
              help = 'The minimum depth of a site to be treated as data'
)
@click.option('-c', '--minimum_count', type=int, default=3, required=False,
              help = 'The minimum count of the minor allele for a site to be treated as data'
)
@click.option('-q', '--minimum_quality', type=int, default=40, required=False,
              help = 'The minimum phred-scaled genotype quality score'
)
@click.option('-o', '--output_dir', type=str, default='dummy', required=False,
              help = 'name of the directory where . will be a matrix of allele frequencies'
)
@click.option('-m', '--estimation_method', type=str, default='gmm', required=False,
              help = 'Method for fitting a model to allele balance data. Only option currently is gmm'
)
@click.option('-p', '--ploidy_levels', type=str, default='2,4,6', required=False,
              help = 'The ploidies you would like to test. Only values between two and six are valid.'
)
@click.option('-f', '--pate_flag', type=bool, default=False, required=False,
              help = 'Is the VCF a product of the PATE pipeline?'
)
@click.option('-s', '--minimum_sites', type=int, default=100, required=False,
              help = 'What are the minimum number of data points needed to fit a mixture model?'
)
@click.option('-e', '--model_contraints', type=int, default=2, required=False,
              help = 'What parameters should be contrained in the model. 0 is none, 1 is means, and 2 is means and weights.'
)
def estimate_ploidy(sample_sheet, vcf_file, minimum_depth, minimum_count, minimum_quality, imputation_method, estimation_method, ploidy_levels, pate_flag, minimum_sites, model_contraints, output_dir):
    from popopolus.utils import map_individuals
    from popopolus.utils import check_dir
    from popopolus.utils import get_vcf_dimensions
    from popopolus.calculate_frequencies.calculate_frequencies import get_ind_ab
    from popopolus.fit_mixtures.fit_mixtures import est_ploidy

    start_time = time.process_time()
    logging.info(f'Begin at {start_time}')
    logging.info(f'Checking all individuals in {sample_sheet} are present in {vcf_file}')
    ind_map = map_individuals(sample_sheet)
    logging.info(f'Checking dimensions of VCF')
    n_sites, n_tax = get_vcf_dimensions(vcf_file, pate_flag, ind_map)
    logging.info(f'Calculating individual allele frequencies from {vcf_file}')
    if str(imputation_method).strip().lower() == 'skip':
        if (output_dir != 'dummy'):
            check_dir(output_dir)
            logging.info(f'Matrix of allele frequencies for each individual will be written to: {output_dir}')
            tax_list, ab_mat = get_ind_ab(n_sites, n_tax, ind_map, vcf_file, minimum_depth, minimum_count, minimum_quality, pate_flag, output_dir)
            ploidy_df = est_ploidy(tax_list, ab_mat, estimation_method, ploidy_levels, minimum_sites, model_contraints, output_dir)
            logging.info('Ploidy estimates returned based on Gaussian mixture models')
            logging.info(ploidy_df.head())
    else:
        click.echo(f'Warning: Imputation method {imputation_method} is not supported. Skipping allele frequencies.')
    end_time = time.process_time()
    logging.info(f'End at {end_time}')
    compute_time = (end_time - start_time) / 60
    logging.info(f'Total compute time was {compute_time} minutes')

##----------------
## Calculate population-level allele frequencies for fst and genotype-environment association analyses
##----------------
#@click.argument('sample_sheet',type=str)
#@click.option('-v', '--vcf_file', type=str, default='dummy.vcf',
#              help = 'name of the input vcf file. should not be compressed.'
#)
#@click.option('-o', '--output_file', type=str, default='samples.vcf',
#              help = 'name of the output text file. will be a matrix of allele frequencies'
#)
#def population_frequencies(sample_sheet, vcf_file, output_file):
#    pass
