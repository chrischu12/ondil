#!/usr/bin/env python3
"""
Test script to verify Clayton copula derivative calculations for different rotations.
This script tests the rotation handling against the C code logic.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add the source path
ondil_path = r"C:\Users\alvar\Documents\Essen\Copula\rolch\src\ondil"
if ondil_path not in sys.path:
    sys.path.insert(0, ondil_path)

from src.ondil.distributions.bicop_clayton import _clayton_logpdf, _clayton_derivative_1st, _clayton_derivative_2nd

def test_rotation_mapping():
    """
    Test the rotation mapping according to VineCopula C code:
    - rotation 0: Standard Clayton (copula 3)
    - rotation 1: 180° rotation (copula 13) 
    - rotation 2: 90° rotation (copula 23)
    - rotation 3: 270° rotation (copula 33)
    """
    
    # Test data from the user's example
    test_data = np.array([
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
    
    # Test parameter (should be positive for Clayton)
    theta = 1.47
    
    print("Testing Clayton Copula Rotations")
    print("=" * 50)
    print(f"Test data shape: {test_data.shape}")
    print(f"Theta parameter: {theta}")
    print()
    
    # Test all rotations
    for rotation in [0, 1, 2, 3]:
        print(f"Rotation {rotation}:")
        print("-" * 20)
        
        try:
            # Test log PDF
            logpdf = _clayton_logpdf(test_data, np.full(len(test_data), theta))
            print(f"Log PDF (first 3): {logpdf[:3]}")
            
            # Test first derivative
            deriv1 = _clayton_derivative_1st(test_data, np.full(len(test_data), theta), rotation)
            print(f"1st derivative (first 3): {deriv1[:3]}")
            
            # Test second derivative
            deriv2 = _clayton_derivative_2nd(test_data, np.full(len(test_data), theta), rotation)
            print(f"2nd derivative (first 3): {deriv2[:3]}")
            
        except Exception as e:
            print(f"ERROR in rotation {rotation}: {e}")
        
        print()
    
    return test_data, theta

def correct_clayton_derivative_1st(y, theta, rotation):
    """
    Corrected first derivative implementation following C code difflPDF_mod logic.
    """
    M = y.shape[0]
    deriv = np.empty((M,), dtype=np.float64)
    u = np.clip(y[:, 0], 1e-12, 1 - 1e-12)
    v = np.clip(y[:, 1], 1e-12, 1 - 1e-12)

    for m in range(M):
        th = theta[m] if hasattr(theta, "__len__") else theta
        
        # Handle rotations following C code difflPDF_mod pattern
        if rotation == 0:  # Standard Clayton (copula 3)
            uu, vv, tth = u[m], v[m], th
            sign = 1.0
        elif rotation == 1:  # 180° rotated Clayton (copula 13)
            uu, vv, tth = 1 - u[m], 1 - v[m], th
            sign = 1.0
        elif rotation == 2:  # 90° rotated Clayton (copula 23)
            uu, vv, tth = 1 - u[m], v[m], -th
            sign = -1.0
        elif rotation == 3:  # 270° rotated Clayton (copula 33)
            uu, vv, tth = u[m], 1 - v[m], -th
            sign = -1.0
        else:
            raise ValueError(f"Invalid rotation: {rotation}")

        # Standard Clayton derivative calculation
        t4 = np.log(uu * vv)
        t5 = uu ** (-tth)
        t6 = vv ** (-tth)
        t7 = t5 + t6 - 1.0
        t8 = np.log(t7)
        t9 = tth ** 2
        t14 = np.log(uu)
        t16 = np.log(vv)
        
        result = 1.0 / (1.0 + tth) - t4 + t8 / t9 + (1.0 / tth + 2.0) * (t5 * t14 + t6 * t16) / t7
        deriv[m] = sign * result

    return deriv

def correct_clayton_derivative_2nd(y, theta, rotation):
    """
    Corrected second derivative implementation following C code diff2lPDF_mod logic.
    """
    UMIN = 1e-12
    UMAX = 1 - 1e-12
    
    u = np.clip(y[:, 0], UMIN, UMAX)
    v = np.clip(y[:, 1], UMIN, UMAX)
    
    if np.isscalar(theta):
        theta = np.full(len(u), theta)
    else:
        theta = np.atleast_1d(theta)
    
    n = len(u)
    out = np.zeros(n)
    
    for i in range(n):
        u_i, v_i = u[i], v[i]
        theta_i = theta[i]
        
        # Handle rotation transformations following C code diff2lPDF_mod
        if rotation == 2:  # 90° rotated (copula 23)
            u_transformed = 1 - u_i  # negu
            v_transformed = v_i
            theta_transformed = -theta_i  # nparam[0] = -param[0]
        elif rotation == 3:  # 270° rotated (copula 33) 
            u_transformed = u_i
            v_transformed = 1 - v_i  # negv
            theta_transformed = -theta_i  # nparam[0] = -param[0]
        elif rotation == 1:  # 180° rotated (copula 13)
            u_transformed = 1 - u_i  # negu
            v_transformed = 1 - v_i  # negv  
            theta_transformed = theta_i  # param unchanged
        else:  # Standard Clayton (copula 3)
            u_transformed = u_i
            v_transformed = v_i
            theta_transformed = theta_i
        
        # Clip for numerical stability
        u_transformed = np.clip(u_transformed, UMIN, UMAX)
        v_transformed = np.clip(v_transformed, UMIN, UMAX)
        
        # Clayton second derivative calculation
        theta_val = theta_transformed
        
        t1 = u_transformed * v_transformed
        t2 = -theta_val - 1.0
        t3 = np.power(t1, t2)
        t4 = np.log(t1)
        
        t6 = np.power(u_transformed, -theta_val)
        t7 = np.power(v_transformed, -theta_val)
        t8 = t6 + t7 - 1.0
        
        t10 = -2.0 - 1.0/theta_val
        t11 = np.power(t8, t10)
        
        t15 = theta_val * theta_val
        t16 = 1.0 / t15
        t17 = np.log(t8)
        
        t19 = np.log(u_transformed)
        t21 = np.log(v_transformed)
        
        t23 = -t6 * t19 - t7 * t21
        t25 = 1.0 / t8
        t27 = t16 * t17 + t10 * t23 * t25
        
        t30 = -t2 * t3
        t31 = t4 * t4
        t34 = t4 * t11
        t38 = t27 * t27
        
        t48 = t19 * t19
        t50 = t21 * t21
        t55 = t23 * t23
        t57 = t8 * t8
        
        t64 = -1.0 / t2
        t68 = 1.0 / (t3 * t11)
        
        # Main calculation following C code structure
        result = (
            -2.0 * t3 * t4 * t11 + 
            2.0 * t3 * t11 * t27 + 
            t30 * t31 * t11 - 
            2.0 * t30 * t34 * t27 + 
            t30 * t11 * t38 + 
            t30 * t11 * (
                -2.0 / (t15 * theta_val) * t17 + 
                2.0 * t16 * t23 * t25 + 
                t10 * (t6 * t48 + t7 * t50) * t25 - 
                t10 * t55 / t57
            )
        ) * t64 * t68
        
        out[i] = result
    
    return out

def compare_implementations():
    """Compare original vs corrected implementations."""
    
    test_data = np.array([
        [0.93 , 0.533],
        [0.853, 0.699], 
        [0.432, 0.813]
    ])
    
    theta = 1.47
    
    print("Comparing Original vs Corrected Implementations")
    print("=" * 60)
    
    for rotation in [0, 1, 2, 3]:
        print(f"\nRotation {rotation}:")
        print("-" * 30)
        
        # Original implementations
        try:
            orig_d1 = _clayton_derivative_1st(test_data, np.full(len(test_data), theta), rotation)
            orig_d2 = _clayton_derivative_2nd(test_data, np.full(len(test_data), theta), rotation)
        except Exception as e:
            print(f"Original failed: {e}")
            continue
            
        # Corrected implementations  
        try:
            corr_d1 = correct_clayton_derivative_1st(test_data, np.full(len(test_data), theta), rotation)
            corr_d2 = correct_clayton_derivative_2nd(test_data, np.full(len(test_data), theta), rotation)
        except Exception as e:
            print(f"Corrected failed: {e}")
            continue
        
        print(f"1st Derivative - Original: {orig_d1}")
        print(f"1st Derivative - Corrected: {corr_d1}")
        print(f"1st Derivative - Diff: {np.abs(orig_d1 - corr_d1)}")
        
        print(f"2nd Derivative - Original: {orig_d2}")
        print(f"2nd Derivative - Corrected: {corr_d2}")
        print(f"2nd Derivative - Diff: {np.abs(orig_d2 - corr_d2)}")

if __name__ == "__main__":
    print("Clayton Copula Rotation Test")
    print("=" * 40)
    
    # Run initial test
    test_data, theta = test_rotation_mapping()
    
    print("\n" + "=" * 60)
    
    # Compare implementations
    compare_implementations()