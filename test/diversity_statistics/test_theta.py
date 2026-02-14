import numpy as np
import pandas as pd
import tempfile
import os
from popopolus.diversity_statistics.theta import estimate_wattersons
from popopolus.utils import map_individuals
from popopolus.calculate_frequencies.calculate_frequencies import get_ind_genotypes

def test_estimate_wattersons_with_test_data():
    """
    Test that estimate_wattersons correctly calculates Watterson's theta using the example VCF data.
    
    This test uses the real test data files (Lsim.example.vcf and Lsim.example.spmap) to ensure
    the function works correctly with actual VCF data. Watterson's theta is a measure of genetic
    diversity that accounts for the number of segregating sites.
    """
    # Get the path to test data
    test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'calculate_frequencies')
    vcf_file = os.path.join(test_data_dir, 'Lsim.example.vcf')
    spmap_file = os.path.join(test_data_dir, 'Lsim.example.spmap')
    
    # Load the individual to population mapping
    ind_map = map_individuals(spmap_file)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Parse the VCF to get genotype data
        # Use large numbers to process all available data
        # min_depth=10, min_count=1, min_qual=20, pass_flag='PASS'
        tax_list, genotype_dat = get_ind_genotypes(
            n_sites=10000,  # Large number to capture all sites
            n_tax=200,      # Large number to capture all individuals
            ind_map=ind_map,
            vcf_file=vcf_file,
            min_depth=10,
            min_count=1,
            min_qual=20,
            pass_flag='PASS',
            output_dir='dummy'  # Don't write output files during testing
        )
        
        # Define an interval covering all sites (dummy interval for now)
        intervals = [(0, 1000000)]
        
        # Calculate Watterson's theta
        theta_df = estimate_wattersons(
            genotype_dat=genotype_dat,
            tax_list=tax_list,
            ind_map=ind_map,
            intervals=intervals,
            output_dir=temp_dir
        )
        
        # Verify the output structure
        assert isinstance(theta_df, pd.DataFrame), "Should return a DataFrame"
        assert 'population' in theta_df.columns, "DataFrame should have 'population' column"
        assert 'theta' in theta_df.columns, "DataFrame should have 'theta' column"
        
        # Check that we have results for the expected populations
        populations = set(theta_df['population'].values)
        expected_populations = {'Itremo', 'Ankafobe', 'Isalo'}
        assert expected_populations.issubset(populations), f"Expected populations {expected_populations}, but got {populations}"
        
        # Verify that theta values are non-negative
        assert all(theta_df['theta'] >= 0), "All theta values should be non-negative"
        
        # Verify that output file was created
        output_file = os.path.join(temp_dir, 'wattersons_theta.csv')
        assert os.path.exists(output_file), "Output CSV file should be created"
        
        # Verify that theta values are reasonable (not NaN or Inf)
        assert all(np.isfinite(theta_df['theta'])), "All theta values should be finite"
        
        print(f"Test passed! Calculated theta for {len(theta_df)} populations:")
        for _, row in theta_df.iterrows():
            print(f"  {row['population']}: {row['theta']:.4f}")


def test_estimate_wattersons_simple_case():
    """
    Test estimate_wattersons with a simple synthetic dataset where we can verify the calculation.
    
    Watterson's theta per site = (S / a1) / n_sites, where:
    - S is the number of segregating sites
    - a1 = sum(1/i for i in range(1, n)) where n is the number of chromosomes
    - n_sites is the total number of sites
    """
    # Create a simple synthetic dataset
    # Population with 3 diploid individuals = 6 chromosomes
    # We'll create 10 sites with different segregation patterns
    
    n_individuals = 3
    n_sites = 10
    
    # Create genotype data: shape (4, n_sites, n_individuals)
    # The first dimension is [genotype_data, site_depth_data, genotype_quality_data, passing_filter_data]
    genotype_array = np.zeros((n_sites, n_individuals), dtype=np.int8)
    
    # Site 0: all homozygous reference (0/0) - not segregating
    genotype_array[0, :] = [0, 0, 0]
    
    # Site 1: one heterozygote (segregating) - 1 derived allele
    genotype_array[1, :] = [1, 0, 0]
    
    # Site 2: two heterozygotes (segregating) - 2 derived alleles
    genotype_array[2, :] = [1, 1, 0]
    
    # Site 3: one homozygous alt (segregating) - 2 derived alleles
    genotype_array[3, :] = [2, 0, 0]
    
    # Site 4: all homozygous alt (not segregating - fixed at 6 alleles)
    genotype_array[4, :] = [2, 2, 2]
    
    # Sites 5-9: more segregating sites
    genotype_array[5, :] = [1, 0, 1]  # 2 derived
    genotype_array[6, :] = [0, 1, 2]  # 3 derived
    genotype_array[7, :] = [1, 1, 1]  # 3 derived
    genotype_array[8, :] = [0, 0, 1]  # 1 derived
    genotype_array[9, :] = [2, 1, 0]  # 3 derived
    
    # Create the full genotype_dat structure
    genotype_data = np.array([
        genotype_array,
        np.ones((n_sites, n_individuals), dtype=np.uint16) * 50,  # depth
        np.ones((n_sites, n_individuals), dtype=np.uint8) * 30,   # quality
        np.ones((n_sites, n_individuals), dtype=np.bool_)          # pass_filter
    ])
    
    # Create tax_list and ind_map
    tax_list = ['ind1', 'ind2', 'ind3']
    ind_map = {
        'ind1': {'population': 'pop1'},
        'ind2': {'population': 'pop1'},
        'ind3': {'population': 'pop1'}
    }
    
    intervals = [(0, 1000000)]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        theta_df = estimate_wattersons(
            genotype_dat=genotype_data,
            tax_list=tax_list,
            ind_map=ind_map,
            intervals=intervals,
            output_dir=temp_dir
        )
        
        # Verify output structure
        assert isinstance(theta_df, pd.DataFrame), f"Should return a DataFrame, got {type(theta_df)}"
        assert len(theta_df) == 1, "Should have one population"
        assert theta_df['population'].iloc[0] == 'pop1', "Population name should be 'pop1'"
        
        # Calculate expected theta per site manually
        # n_chromosomes = 2 * 3 = 6
        # a1 = sum(1/i for i from 1 to n_chromosomes-1) = 1/1 + 1/2 + 1/3 + 1/4 + 1/5 ≈ 2.283
        n_chromosomes = 2 * n_individuals
        a1 = sum(1/i for i in range(1, n_chromosomes))
        
        # Count segregating sites (sites where not all individuals have the same genotype)
        # Sites 0 and 4 are not segregating (all same)
        # Sites 1, 2, 3, 5, 6, 7, 8, 9 are segregating = 8 sites
        # Expected S = 8
        expected_S = 8
        total_sites = n_sites
        
        # Watterson's theta = S / a1, then per-site = theta / n_sites
        expected_theta = expected_S / a1
        expected_theta_persite = expected_theta / total_sites
        
        calculated_theta_persite = theta_df['theta'].iloc[0]
        
        print(f"Total sites: {total_sites}")
        print(f"Segregating sites (S): {expected_S}")
        print(f"Number of chromosomes: {n_chromosomes}")
        print(f"a1: {a1:.4f}")
        print(f"Theta: {expected_theta:.4f}")
        print(f"Expected theta per site: {expected_theta_persite:.4f}")
        print(f"Calculated theta per site: {calculated_theta_persite:.4f}")
        
        # Allow for small floating point differences
        assert abs(calculated_theta_persite - expected_theta_persite) < 0.01, \
            f"Theta per site calculation mismatch: expected {expected_theta_persite:.4f}, got {calculated_theta_persite:.4f}"


if __name__ == '__main__':
    print("Running test_estimate_wattersons_simple_case...")
    test_estimate_wattersons_simple_case()
    print("\nRunning test_estimate_wattersons_with_test_data...")
    test_estimate_wattersons_with_test_data()
    print("\nAll tests passed!")
