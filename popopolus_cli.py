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
from datetime import datetime
import logging

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
# Site frequency spectrum and diversity statistics
#----------------
@cli.command(context_settings={'help_option_names': ['-h','--help']})
@click.argument('sample_sheet',type=str)
@click.option('-v', '--vcf_file', type=str, default='dummy.vcf', required=True,
              help = 'name of the input vcf file. should not be compressed.'
)
@click.option('-i', '--imputation_method', type=str, default='drop', required=False,
              help = 'decide how to impute missing data if at all. options are: drop, mean, and popmean'
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

    start_time = time.process_time()
    logging.info(f'Begin at {start_time}')
    logging.info(f'Checking all individuals in {sample_sheet} are present in {vcf_file}')
    ind_map = map_individuals(sample_sheet)
    logging.info(f'Checking dimensions of VCF')
    n_sites, n_tax = get_vcf_dimensions(vcf_file, pass_flag, ind_map)
    logging.info(f'Calculating individual allele frequencies from {vcf_file}')
    if (imputation_method == 'drop'):
        if (output_dir != 'dummy'):
            check_dir(output_dir)
            logging.info(f'Matrix of allele frequencies for each individual will be written to: {output_dir}')
        get_ind_genotypes(n_sites, n_tax, ind_map, vcf_file, minimum_depth, minimum_count, minimum_quality, pass_flag, output_dir)
        
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
@click.option('-i', '--imputation_method', type=str, default='drop', required=False,
              help = 'decide how to impute missing data if at all. options are: drop, mean, and popmean'
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
@click.option('-o', '--output_dir', type=str, default='dummy', required=False,
              help = 'name of the directory where . will be a matrix of allele frequencies'
)
def estimate_theta(sample_sheet, vcf_file, minimum_depth, minimum_count, minimum_quality, imputation_method, pass_flag, interval, folded, output_dir):
    from popopolus.utils import map_individuals
    from popopolus.utils import check_dir
    from popopolus.utils import get_vcf_dimensions
    from popopolus.calculate_frequencies.calculate_frequencies import get_ind_genotypes
    from popopolus.diversity_statistics.theta import estimate_thetas

    start_time = time.process_time()
    logging.info(f'Begin at {start_time}')
    logging.info(f'Checking all individuals in {sample_sheet} are present in {vcf_file}')
    ind_map = map_individuals(sample_sheet)
    logging.info(f'Checking dimensions of VCF')
    n_sites, n_tax = get_vcf_dimensions(vcf_file, pass_flag, ind_map)
    logging.info(f'Calculating individual allele frequencies from {vcf_file}')
    if (imputation_method == 'drop'):
        if (output_dir != 'dummy'):
            check_dir(output_dir)
            logging.info(f'Matrix of allele frequencies for each individual will be written to: {output_dir}')
        tax_list, genotype_dat = get_ind_genotypes(n_sites, n_tax, ind_map, vcf_file, minimum_depth, minimum_count, minimum_quality, pass_flag, output_dir)
        theta_df = estimate_thetas(genotype_dat, tax_list, ind_map, interval, folded, output_dir)
        print(theta_df)
    else:
        click.echo(f'Warning: Imputation method {imputation_method} is not supported.')
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
@click.option('-i', '--imputation_method', type=str, default='drop', required=False,
              help = 'decide how to impute missing data if at all. options are: drop, mean, and popmean'
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
    if (imputation_method == 'drop'):
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
@click.option('-i', '--imputation_method', type=str, default='drop', required=False,
              help = 'decide how to impute missing data if at all. options are: drop, mean, and popmean'
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
    if (imputation_method == 'drop'):
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
