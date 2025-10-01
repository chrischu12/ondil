#!/usr/bin/env python3
"""
Final verification of Clayton copula rotation derivatives.
This script tests the corrected rotation handling and verifies numerical derivatives.
"""

import sys
import os
import numpy as np
from scipy.optimize import approx_fprime

# Add the source path
ondil_path = r"C:\Users\alvar\Documents\Essen\Copula\rolch\src\ondil"
if ondil_path not in sys.path:
    sys.path.insert(0, ondil_path)

from src.ondil.distributions.bicop_clayton import _clayton_logpdf, _clayton_derivative_1st, _clayton_derivative_2nd

def numerical_derivative_1st(y, theta, rotation, eps=1e-6):
    """Calculate numerical first derivative for verification."""
    
    def loglik_wrapper(th):
        return np.sum(_clayton_logpdf(y, np.full(len(y), th[0])))
    
    # Handle rotation transformations for numerical derivative
    if rotation == 2:  # 90°
        y_transformed = np.column_stack([1 - y[:, 0], y[:, 1]])
        theta_transformed = -theta
    elif rotation == 3:  # 270°
        y_transformed = np.column_stack([y[:, 0], 1 - y[:, 1]])
        theta_transformed = -theta
    elif rotation == 1:  # 180°
        y_transformed = np.column_stack([1 - y[:, 0], 1 - y[:, 1]])
        theta_transformed = theta
    else:  # Standard
        y_transformed = y
        theta_transformed = theta
    
    def loglik_transformed(th):
        return np.sum(_clayton_logpdf(y_transformed, np.full(len(y_transformed), th[0])))
    
    grad = approx_fprime([np.abs(theta_transformed)], loglik_transformed, eps)
    
    # Apply sign correction for rotations 2 and 3
    if rotation in [2, 3]:
        return -grad[0]
    else:
        return grad[0]

def test_derivatives_comprehensive():
    """Comprehensive test of all rotations with derivative verification."""
    
    # Test data
    test_data = np.array([
        [0.93 , 0.533],
        [0.853, 0.699],
        [0.432, 0.813],
        [0.702, 0.454],
        [0.737, 0.654]
    ])
    
    theta = 1.47
    
    print("Clayton Copula Comprehensive Derivative Test")
    print("=" * 60)
    print(f"Test data shape: {test_data.shape}")
    print(f"Theta parameter: {theta}")
    print()
    
    for rotation in [0, 1, 2, 3]:
        print(f"Rotation {rotation} ({'Standard' if rotation == 0 else '180°' if rotation == 1 else '90°' if rotation == 2 else '270°'}):")
        print("-" * 40)
        
        try:
            # Analytical derivatives
            deriv1_analytical = _clayton_derivative_1st(test_data, np.full(len(test_data), theta), rotation)
            deriv2_analytical = _clayton_derivative_2nd(test_data, np.full(len(test_data), theta), rotation)
            
            # Numerical first derivative (sum over all observations)
            deriv1_numerical = numerical_derivative_1st(test_data, theta, rotation)
            deriv1_analytical_sum = np.sum(deriv1_analytical)
            
            print(f"Analytical 1st deriv (sum): {deriv1_analytical_sum:.6f}")
            print(f"Numerical 1st deriv (sum):  {deriv1_numerical:.6f}")
            print(f"1st deriv difference:       {abs(deriv1_analytical_sum - deriv1_numerical):.6f}")
            
            print(f"Analytical 1st deriv (first 3): {deriv1_analytical[:3]}")
            print(f"Analytical 2nd deriv (first 3): {deriv2_analytical[:3]}")
            
            # Check for any problematic values
            if np.any(np.isnan(deriv1_analytical)) or np.any(np.isnan(deriv2_analytical)):
                print("WARNING: NaN values found!")
            elif np.any(np.isinf(deriv1_analytical)) or np.any(np.isinf(deriv2_analytical)):
                print("WARNING: Infinite values found!")
            else:
                print("✓ All values are finite")
                
        except Exception as e:
            print(f"ERROR: {e}")
        
        print()

def test_rotation_consistency():
    """Test that rotations produce expected relationships."""
    
    # Test with a simple case
    test_data = np.array([[0.7, 0.6], [0.3, 0.8]])
    theta = 2.0
    
    print("Rotation Consistency Test")
    print("=" * 40)
    
    # Calculate derivatives for all rotations
    results = {}
    for rotation in [0, 1, 2, 3]:
        try:
            deriv1 = _clayton_derivative_1st(test_data, np.full(len(test_data), theta), rotation)
            deriv2 = _clayton_derivative_2nd(test_data, np.full(len(test_data), theta), rotation)
            results[rotation] = {'d1': deriv1, 'd2': deriv2}
            print(f"Rotation {rotation}: 1st deriv = {deriv1}, 2nd deriv = {deriv2}")
        except Exception as e:
            print(f"Rotation {rotation}: ERROR - {e}")
    
    print()
    
    # Test specific rotation properties if all succeeded
    if len(results) == 4:
        print("Rotation relationship checks:")
        print("- All rotations should give different but finite values")
        
        for i in range(4):
            d1_finite = np.all(np.isfinite(results[i]['d1']))
            d2_finite = np.all(np.isfinite(results[i]['d2']))
            print(f"  Rotation {i}: 1st finite = {d1_finite}, 2nd finite = {d2_finite}")

if __name__ == "__main__":
    test_derivatives_comprehensive()
    print("\n" + "=" * 60 + "\n")
    test_rotation_consistency()