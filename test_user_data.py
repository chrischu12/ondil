#!/usr/bin/env python3
"""
Test with the exact user data and parameter from the original question.
"""

import sys
import os
import numpy as np

# Add the source path
ondil_path = r"C:\Users\alvar\Documents\Essen\Copula\rolch\src\ondil"
if ondil_path not in sys.path:
    sys.path.insert(0, ondil_path)

from src.ondil.distributions.bicop_clayton import _clayton_logpdf, _clayton_derivative_1st, _clayton_derivative_2nd

def test_original_user_data():
    """Test with the exact data from the user's question."""
    
    # Exact data from user: merged_data_clayton2[["u1", "u2"]].head(10).to_numpy()
    y_numpy = np.array([
        [0.93 , 0.533],
        [0.853, 0.699],
        [0.432, 0.813],
        [0.702, 0.454],
        [0.737, 0.654],
        [0.585, 0.572],
        [0.595, 0.856],
        [0.169, 0.153],
        [0.947, 0.863],
        [0.893, 0.978]
    ])
    
    # Test parameter from user code
    theta = 1.47
    
    print("Test with Original User Data")
    print("=" * 50)
    print("Data shape:", y_numpy.shape)
    print("First few rows:")
    print(y_numpy[:3])
    print(f"Theta: {theta}")
    print()
    
    # Test rotation 3 specifically (this was mentioned in user's code)
    rotation = 3
    print(f"Testing rotation {rotation} (270°) - from user's BivariateCopulaClayton(rotation={rotation}):")
    print("-" * 60)
    
    try:
        # Log PDF
        logpdf = _clayton_logpdf(y_numpy, np.full(len(y_numpy), theta))
        print(f"Log PDF values (first 5): {logpdf[:5]}")
        
        # First derivative
        deriv1 = _clayton_derivative_1st(y_numpy, np.full(len(y_numpy), theta), rotation)
        print(f"1st derivative (first 5): {deriv1[:5]}")
        print(f"1st derivative (all 10): {deriv1}")
        
        # Second derivative
        deriv2 = _clayton_derivative_2nd(y_numpy, np.full(len(y_numpy), theta), rotation)
        print(f"2nd derivative (first 5): {deriv2[:5]}")
        print(f"2nd derivative (all 10): {deriv2}")
        
        # Check for any issues
        print()
        print("Validation checks:")
        print(f"- Any NaN in 1st deriv: {np.any(np.isnan(deriv1))}")
        print(f"- Any Inf in 1st deriv: {np.any(np.isinf(deriv1))}")
        print(f"- Any NaN in 2nd deriv: {np.any(np.isnan(deriv2))}")
        print(f"- Any Inf in 2nd deriv: {np.any(np.isinf(deriv2))}")
        print(f"- All 1st deriv finite: {np.all(np.isfinite(deriv1))}")
        print(f"- All 2nd deriv finite: {np.all(np.isfinite(deriv2))}")
        
        if np.all(np.isfinite(deriv1)) and np.all(np.isfinite(deriv2)):
            print("✓ SUCCESS: All derivatives are finite!")
        else:
            print("✗ ISSUE: Some derivatives are not finite")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 50)
    
    # Test all rotations with this data
    print("Testing all rotations with user data:")
    for rot in [0, 1, 2, 3]:
        try:
            d1 = _clayton_derivative_1st(y_numpy, np.full(len(y_numpy), theta), rot)
            d2 = _clayton_derivative_2nd(y_numpy, np.full(len(y_numpy), theta), rot)
            
            d1_ok = np.all(np.isfinite(d1))
            d2_ok = np.all(np.isfinite(d2))
            
            print(f"Rotation {rot}: 1st deriv OK = {d1_ok}, 2nd deriv OK = {d2_ok}")
            if not d1_ok or not d2_ok:
                print(f"  -> 1st deriv: {d1}")
                print(f"  -> 2nd deriv: {d2}")
                
        except Exception as e:
            print(f"Rotation {rot}: ERROR - {e}")

if __name__ == "__main__":
    test_original_user_data()