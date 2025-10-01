#!/usr/bin/env python3
"""
Direct debug of the logpdf function to see what's happening.
"""

import numpy as np
from src.ondil.distributions.bicop_clayton import _clayton_logpdf, _get_effective_rotation

# Test with a single data point
u, v = 0.5, 0.7
y_test = np.array([[u, v]])

print("=== Direct function debug ===")
print(f"Test point: u={u}, v={v}")
print()

theta = -2.0
family_code = 301

print(f"Testing theta = {theta}, family_code = {family_code}")

# Check rotation
rotation = _get_effective_rotation(theta, family_code)
print(f"Effective rotation: {rotation}")

# Call the function and see what happens
result = _clayton_logpdf(y_test, theta, family_code)
print(f"Function result: {result}")
print(f"Result shape: {result.shape}")
print(f"Result values: {result}")

# Step through the function manually
print("\n--- Manual step-through ---")
u_clip = np.clip(y_test[:, 0], 1e-12, 1 - 1e-12)
v_clip = np.clip(y_test[:, 1], 1e-12, 1 - 1e-12)
print(f"u_clip: {u_clip}")
print(f"v_clip: {v_clip}")

# Check if theta is scalar
print(f"theta is scalar: {np.isscalar(theta)}")

if np.isscalar(theta):
    print("Taking scalar path...")
    
    rotation = _get_effective_rotation(theta, family_code)
    print(f"Rotation: {rotation}")
    
    # Apply rotation transformations to data
    u_rot, v_rot = u_clip, v_clip
    if rotation == 1:  # 180° rotation (survival)
        u_rot = 1 - u_clip
        v_rot = 1 - v_clip
    elif rotation == 2:  # 90° rotation
        u_rot = 1 - u_clip
        # v_rot stays the same
    elif rotation == 3:  # 270° rotation
        # u_rot stays the same
        v_rot = 1 - v_clip
    
    print(f"u_rot: {u_rot}, v_rot: {v_rot}")
    
    # Use absolute value of theta for the copula calculation
    theta_abs = np.maximum(np.abs(theta), 1e-6)
    print(f"theta_abs: {theta_abs}")
    
    t5 = u_rot ** (-theta_abs)
    t6 = v_rot ** (-theta_abs)
    print(f"t5: {t5}, t6: {t6}")
    
    t7 = t5 + t6 - 1.0
    t7 = np.maximum(t7, 1e-12)
    print(f"t7: {t7}")
    
    logpdf = (
        np.log(theta_abs + 1)
        - (theta_abs + 1) * (np.log(u_rot) + np.log(v_rot))
        - (2.0 + 1.0 / theta_abs) * np.log(t7)
    )
    print(f"logpdf calculation: {logpdf}")
    
    logpdf = np.where(np.isfinite(logpdf), logpdf, np.log(1e-16))
    print(f"logpdf after finite check: {logpdf}")
    
    final_result = np.full(len(y_test), logpdf)
    print(f"final_result: {final_result}")