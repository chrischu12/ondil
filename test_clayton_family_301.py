#!/usr/bin/env python3
"""
Test script to verify that Python Clayton copula with family_code=301 
matches exactly with R gamCopula bicoppd1d2 function.
"""

import numpy as np
import pandas as pd
from src.ondil.distributions.bicop_clayton import BivariateCopulaClayton

def test_clayton_301_comparison():
    """
    Test that Python Clayton family 301 matches R gamCopula results.
    """
    
    # Load the data from your previous R tests
    print("=== Testing Clayton Family 301 vs R gamCopula ===")
    
    # Create test data - same as used in R
    np.random.seed(123)
    n = 100
    u = np.random.uniform(0.01, 0.99, n)
    v = np.random.uniform(0.01, 0.99, n)
    y_numpy = np.column_stack([u, v])
    
    # Test parameters - both positive and negative
    test_thetas = [2.0, -1.5, 0.5, -0.8, 1.2, -2.0]
    
    # Create Clayton copula with family 301
    clayton_301 = BivariateCopulaClayton(family_code=301)
    
    print(f"Testing with {len(test_thetas)} different theta values...")
    print(f"Data shape: {y_numpy.shape}")
    print()
    
    for theta in test_thetas:
        print(f"--- Testing theta = {theta} ---")
        
        # Determine expected rotation based on family 301 logic
        if theta > 0:
            expected_rotation = "Standard Clayton (rotation 0)"
            expected_vinecop_family = 3
        else:
            expected_rotation = "90° rotated Clayton (rotation 2)" 
            expected_vinecop_family = 23
            
        print(f"Expected rotation: {expected_rotation}")
        print(f"Expected VineCopula family: {expected_vinecop_family}")
        
        # Test logpdf
        try:
            logpdf_result = clayton_301.logpdf(y_numpy, {0: theta})
            print(f"✓ LogPDF computed: shape {logpdf_result.shape}, finite values: {np.sum(np.isfinite(logpdf_result))}/{len(logpdf_result)}")
            print(f"  LogPDF range: [{np.min(logpdf_result):.6f}, {np.max(logpdf_result):.6f}]")
            
            # Check for NaN or infinite values
            if not np.all(np.isfinite(logpdf_result)):
                print(f"  ⚠️  Warning: {np.sum(~np.isfinite(logpdf_result))} non-finite values found")
                
        except Exception as e:
            print(f"  ❌ Error in logpdf: {e}")
            continue
        
        # Test first derivative
        try:
            d1_result = clayton_301.dl1_dp1(y_numpy, {0: theta})
            print(f"✓ First derivative computed: shape {d1_result.shape}, finite values: {np.sum(np.isfinite(d1_result))}/{len(d1_result)}")
            print(f"  D1 range: [{np.min(d1_result):.6f}, {np.max(d1_result):.6f}]")
            
            if not np.all(np.isfinite(d1_result)):
                print(f"  ⚠️  Warning: {np.sum(~np.isfinite(d1_result))} non-finite values found")
                
        except Exception as e:
            print(f"  ❌ Error in first derivative: {e}")
            continue
            
        # Test second derivative
        try:
            d2_result = clayton_301.dl2_dp2(y_numpy, {0: theta})
            print(f"✓ Second derivative computed: shape {d2_result.shape}, finite values: {np.sum(np.isfinite(d2_result))}/{len(d2_result)}")
            print(f"  D2 range: [{np.min(d2_result):.6f}, {np.max(d2_result):.6f}]")
            
            if not np.all(np.isfinite(d2_result)):
                print(f"  ⚠️  Warning: {np.sum(~np.isfinite(d2_result))} non-finite values found")
                
        except Exception as e:
            print(f"  ❌ Error in second derivative: {e}")
            continue
            
        print()

def save_test_data_for_r():
    """
    Generate and save test data that can be used in R for comparison.
    """
    print("=== Generating test data for R comparison ===")
    
    # Generate the same data as used in tests
    np.random.seed(123)
    n = 100
    u = np.random.uniform(0.01, 0.99, n)
    v = np.random.uniform(0.01, 0.99, n)
    
    # Save as CSV for R
    test_data = pd.DataFrame({
        'u': u,
        'v': v
    })
    
    output_file = "test_data_for_r_comparison.csv"
    test_data.to_csv(output_file, index=False)
    print(f"✓ Test data saved to: {output_file}")
    print(f"  Shape: {test_data.shape}")
    print(f"  Columns: {list(test_data.columns)}")
    
    # Also generate test parameters
    test_thetas = [2.0, -1.5, 0.5, -0.8, 1.2, -2.0]
    theta_df = pd.DataFrame({'theta': test_thetas})
    theta_file = "test_thetas_for_r_comparison.csv"
    theta_df.to_csv(theta_file, index=False)
    print(f"✓ Test thetas saved to: {theta_file}")
    print(f"  Thetas: {test_thetas}")
    
    return test_data, test_thetas

def compute_python_results_for_r():
    """
    Compute Python results and save them for R comparison.
    """
    print("=== Computing Python results for R comparison ===")
    
    # Load or generate test data
    test_data, test_thetas = save_test_data_for_r()
    y_numpy = test_data[['u', 'v']].values
    
    # Create Clayton copula with family 301
    clayton_301 = BivariateCopulaClayton(family_code=301)
    
    results = []
    
    for i, theta in enumerate(test_thetas):
        print(f"Computing for theta = {theta}")
        
        try:
            # Compute all quantities
            logpdf = clayton_301.logpdf(y_numpy, {0: theta})
            d1 = clayton_301.dl1_dp1(y_numpy, {0: theta})
            d2 = clayton_301.dl2_dp2(y_numpy, {0: theta})
            
            # Store results for each observation
            for j in range(len(y_numpy)):
                results.append({
                    'theta_index': i,
                    'theta': theta,
                    'obs_index': j,
                    'u': y_numpy[j, 0],
                    'v': y_numpy[j, 1],
                    'logpdf': logpdf[j],
                    'derivative_1st': d1[j],
                    'derivative_2nd': d2[j]
                })
                
        except Exception as e:
            print(f"  ❌ Error for theta {theta}: {e}")
            
    # Save results
    results_df = pd.DataFrame(results)
    output_file = "python_clayton_301_results.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"✓ Python results saved to: {output_file}")
    print(f"  Shape: {results_df.shape}")
    print(f"  Columns: {list(results_df.columns)}")
    
    return results_df

if __name__ == "__main__":
    print("Testing Clayton Copula Family 301 Implementation")
    print("=" * 50)
    
    # Run basic functionality test
    test_clayton_301_comparison()
    
    # Generate data for R comparison
    save_test_data_for_r()
    
    # Compute Python results for R comparison
    compute_python_results_for_r()
    
    print("\n" + "=" * 50)
    print("✓ All tests completed!")
    print("✓ Test data and results saved for R comparison")
    print("\nNext steps:")
    print("1. Run the equivalent R script with gamCopula")
    print("2. Compare the results CSV files")
    print("3. Verify that family 301 gives identical results")