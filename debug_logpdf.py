#!/usr/bin/env python3
"""
Debug script to understand the logpdf issue with negative theta values.
"""

import numpy as np
from src.ondil.distributions.bicop_clayton import BivariateCopulaClayton

# Test with a single data point
u, v = 0.5, 0.7
y_test = np.array([[u, v]])

print("=== Debug: Single point test ===")
print(f"Test point: u={u}, v={v}")
print()

# Test different theta values
test_thetas = [2.0, -2.0, 1.5, -1.5]

clayton_301 = BivariateCopulaClayton(family_code=301)

for theta in test_thetas:
    print(f"--- Theta = {theta} ---")
    
    # Determine rotation
    if theta > 0:
        expected_rotation = 0  # Standard
        expected_desc = "Standard Clayton"
    else:
        expected_rotation = 2  # 90° rotation  
        expected_desc = "90° rotated Clayton"
    
    print(f"Expected rotation: {expected_rotation} ({expected_desc})")
    
    # Show what data transformation should be applied
    if expected_rotation == 0:
        u_rot, v_rot = u, v
        print(f"Data transformation: u={u_rot:.3f}, v={v_rot:.3f} (no change)")
    elif expected_rotation == 2:
        u_rot, v_rot = 1-u, v
        print(f"Data transformation: u={u_rot:.3f}, v={v_rot:.3f} (u rotated)")
    
    # Compute logpdf
    logpdf = clayton_301.logpdf(y_test, {0: theta})
    print(f"LogPDF result: {logpdf[0]:.6f}")
    
    # Manual calculation to check
    theta_abs = abs(theta)
    if expected_rotation == 0:
        u_calc, v_calc = u, v
    elif expected_rotation == 2:
        u_calc, v_calc = 1-u, v
    
    t5 = u_calc ** (-theta_abs)
    t6 = v_calc ** (-theta_abs)
    t7 = t5 + t6 - 1.0
    manual_logpdf = (
        np.log(theta_abs + 1)
        - (theta_abs + 1) * (np.log(u_calc) + np.log(v_calc))
        - (2.0 + 1.0 / theta_abs) * np.log(t7)
    )
    print(f"Manual calculation: {manual_logpdf:.6f}")
    print(f"Match: {abs(logpdf[0] - manual_logpdf) < 1e-10}")
    
    print()