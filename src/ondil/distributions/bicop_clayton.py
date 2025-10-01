# Author: Your Name
# MIT Licence
from typing import Dict

import numpy as np
import scipy.stats as st

from ..base import Distribution, LinkFunction, CopulaMixin
from ..links import LogShiftValue,Log, KendallsTauToParameterClayton,Identity
from ..types import ParameterShapes

class BivariateCopulaClayton(CopulaMixin, Distribution):
    """
    Bivariate Clayton copula distribution class.
    """

    corresponding_gamlss: str = None
    parameter_names = {0: "theta"}
    parameter_support = {0: (1e-6, np.inf)}
    distribution_support = (0, 1)
    n_params = len(parameter_names)
    parameter_shape = {0: ParameterShapes.SCALAR}

    def __init__(
        self,
        link: LinkFunction = Log(),
        param_link: LinkFunction = KendallsTauToParameterClayton(),
        family_code: int = 301
    ):
        super().__init__(
            links={0: link},
            param_links={0: param_link},
            rotation=0,  # Default rotation, overridden by family_code logic
        )
        self.family_code = family_code  # gamCopula family code (301, 302, 303, 304)
        self.is_multivariate = True
        self._adr_lower_diag = {0: False}
        self._regularization_allowed = {0: False}
        self._regularization = ""
        self._scoring = "fisher"

    @staticmethod
    def fitted_elements(dim: int):
        return {0: int(dim * (dim - 1) // 2)}

    def get_effective_rotation(self, theta_value):
        """
        Get the effective rotation based on family code and parameter sign.
        This mimics the gamCopula logic from getFams() and bicoppd1d2().
        
        Args:
            theta_value: The copula parameter value
            
        Returns:
            int: The effective rotation (0, 1, 2, 3)
        """
        # Map gamCopula family codes to VineCopula rotations
        # Based on getFams() function from utilsFamilies.R
        if self.family_code == 301:
            # Double Clayton type I (standard and rotated 90 degrees)
            if theta_value > 0:
                return 0  # Standard Clayton (3)
            else:
                return 2  # 90° rotation (23)
        elif self.family_code == 302:
            # Double Clayton type II (standard and rotated 270 degrees)
            if theta_value > 0:
                return 0  # Standard Clayton (3)
            else:
                return 3  # 270° rotation (33)
        elif self.family_code == 303:
            # Double Clayton type III (survival and rotated 90 degrees)
            if theta_value > 0:
                return 1  # 180° rotation (13) - survival
            else:
                return 2  # 90° rotation (23)
        elif self.family_code == 304:
            # Double Clayton type IV (survival and rotated 270 degrees)
            if theta_value > 0:
                return 1  # 180° rotation (13) - survival
            else:
                return 3  # 270° rotation (33)
        else:
            # Default to 301 behavior if invalid family code
            return 0 if theta_value > 0 else 2

    @property
    def param_structure(self):
        return self._param_structure

    @staticmethod
    def set_theta_element(theta: dict, value: np.ndarray, param: int, k: int) -> dict:
        theta[param] = value
        return theta

    def theta_to_params(self, theta):
        if isinstance(theta, dict):
            theta = theta[0]
        theta_array = np.asarray(theta)
        # For gamCopula family codes, we need to preserve the sign for rotation selection
        # Only clip the absolute value to prevent extreme values
        sign = np.sign(theta_array)
        abs_theta = np.abs(theta_array)
        abs_theta_clipped = np.clip(abs_theta, 1e-6, 200)  # Match R bounds
        return sign * abs_theta_clipped

    def theta_to_scipy_params(self, theta: np.ndarray) -> dict:
        return {"theta": theta}

    def set_initial_guess(self, theta, param):
        return theta

    def initial_values(self, y, param=0):
        M = y.shape[0]
        # Compute the empirical Pearson correlation for each sample
        # y is expected to be (M, 2)
        tau = st.kendalltau(y[:, 0], y[:, 1]).correlation
        chol = np.full((M, 1), tau)
        return chol.reshape(-1)

    def cube_to_flat(self, x: np.ndarray, param: int):
        return x

    def flat_to_cube(self, x: np.ndarray, param: int):
        return x

    def param_conditional_likelihood(
        self, y: np.ndarray, theta: dict, eta: np.ndarray, param: int
    ) -> np.ndarray:
        fitted = self.flat_to_cube(eta, param=param)
        fitted = self.link_inverse(fitted, param=param)
        return self.log_likelihood(y, theta={**theta, param: fitted})

    def theta_to_scipy(self, theta: dict):
        return {"theta": theta}

    def logpdf(self, y, theta):
        theta = self.theta_to_params(theta)
        result = _clayton_logpdf(y, theta, self.family_code)
        return result.reshape(-1)
    
    def pdf(self, y, theta):
        return np.exp(self.logpdf(y, theta))

    def dl1_dp1(self, y, theta, param=0):
        theta = self.theta_to_params(theta)
        return _clayton_derivative_1st(y, theta, self.family_code)

    def dl2_dp2(self, y, theta, param=0, clip=False):
        """
        Second derivative with proper chain rule matching gamBiCopFit.R
        """
        theta = self.theta_to_params(theta)
        return _clayton_derivative_2nd(y, theta, self.family_code)
        

    def element_score(self, y, theta, param=0, k=0):
        return self.element_dl1_dp1(y, theta, param, k)

    def element_hessian(self, y, theta, param=0, k=0):
        return self.element_dl2_dp2(y, theta, param, k)

    def element_dl1_dp1(self, y, theta, param=0, k=0, clip=False):
    # Apply proper chain rule like dl1_dp1
        theta_params = self.theta_to_params(theta)
        raw_d1 = _clayton_derivative_1st(y, theta_params, self.family_code)
        
        # Apply link derivative
        # eta = self.links[param].link(theta_params)
        # dpar_deta = self.links[param].link_derivative(eta)
        
        return raw_d1 

    def element_dl2_dp2(self, y, theta, param=0, k=0, clip=False):
        # Apply proper chain rule like dl2_dp2
        theta_params = self.theta_to_params(theta)
        
        raw_d2 = _clayton_derivative_2nd(y, theta_params, self.family_code)
        return raw_d2

    def dl2_dpp(self, y, theta, param=0):
        raise NotImplementedError("Not implemented for Clayton copula.")

    def element_link_function(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        return self.links[param].element_link(y)

    def element_link_function_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        return self.links[param].element_derivative(y)

    def element_link_function_second_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        return self.links[param].element_link_second_derivative(y)

    def element_link_inverse(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        return self.links[param].inverse(y)

    def element_link_inverse_derivative(
        self, y: np.ndarray, param: int = 0, k: int = 0, d: int = 0
    ) -> np.ndarray:
        return self.links[param].element_inverse_derivative(y)

    def calculate_conditional_initial_values(
        self, y: np.ndarray, theta: dict
    ) -> dict:
        raise NotImplementedError("Not implemented for Clayton copula.")

    def cdf(self, y, theta):
        raise NotImplementedError("Not implemented for Clayton copula.")

    def ppf(self, q, theta):
        raise NotImplementedError("Not implemented for Clayton copula.")

    def rvs(self, size, theta):
        theta = self.theta_to_params(theta)
        u = np.random.uniform(size=size)
        w = np.random.uniform(size=size)
        v = (w ** (-theta / (1 + theta)) * (u ** (-theta) - 1) + 1) ** (-1 / theta)
        return np.column_stack((u, v))

    def logcdf(self, y, theta):
        raise NotImplementedError("Not implemented for Clayton copula.")

    def logpmf(self, y, theta):
        raise NotImplementedError("Not implemented for Clayton copula.")

    def pmf(self, y, theta):
        raise NotImplementedError("Not implemented for Clayton copula.")

    def hfunc(self, u: np.ndarray, v: np.ndarray, theta: np.ndarray, un: int) -> np.ndarray:
        M = u.shape[0]
        UMIN = 1e-12
        UMAX = 1 - 1e-12

        u = np.clip(u, UMIN, UMAX)
        v = np.clip(v, UMIN, UMAX)

        # Swap u and v if un == 2
        if un == 2:
            u, v = v, u

        h = np.empty(M)
        for m in range(M):
            th = theta[m] if hasattr(theta, "__len__") else theta
            # Conditional distribution function for Clayton copula
            # h(u|v) = ∂C(u,v)/∂v = (u^{-θ-1}) * (u^{-θ} + v^{-θ} - 1)^{-1/θ - 1}
            t1 = u[m] ** (-th - 1)
            t2 = u[m] ** (-th) + v[m] ** (-th) - 1
            t2 = np.maximum(t2, UMIN)
            t3 = -1.0 / th - 1.0
            h[m] = t1 * (t2 ** t3)
        h = np.clip(h, UMIN, UMAX)
        return h
    
    def get_regularization_size(self, dim: int) -> int:
        return dim

##########################################################
# Helper functions for the log-likelihood and derivatives #
##########################################################

def _get_effective_rotation(theta_value, family_code):
    """
    Get the effective rotation based on family code and parameter sign.
    This mimics the gamCopula logic from getFams() and bicoppd1d2().
    """
    # Map gamCopula family codes to VineCopula rotations
    # Based on getFams() function from utilsFamilies.R
    if family_code == 301:
        # Double Clayton type I (standard and rotated 90 degrees)
        return 0 if theta_value > 0 else 2  # Standard (3) or 90° (23)
    elif family_code == 302:
        # Double Clayton type II (standard and rotated 270 degrees)
        return 0 if theta_value > 0 else 3  # Standard (3) or 270° (33)
    elif family_code == 303:
        # Double Clayton type III (survival and rotated 90 degrees)
        return 1 if theta_value > 0 else 2  # 180° (13) or 90° (23)
    elif family_code == 304:
        # Double Clayton type IV (survival and rotated 270 degrees)
        return 1 if theta_value > 0 else 3  # 180° (13) or 270° (33)
    else:
        # Default to 301 behavior if invalid family code
        return 0 if theta_value > 0 else 2

def _clayton_logpdf(y, theta, family_code=301):
    """
    Clayton copula log-PDF with proper rotation handling.
    The likelihood uses the correct rotated copula for the given parameter.
    """
    u = np.clip(y[:, 0], 1e-12, 1 - 1e-12)
    v = np.clip(y[:, 1], 1e-12, 1 - 1e-12)
    
    if np.isscalar(theta):
        # Get the effective rotation for this theta and family
        rotation = _get_effective_rotation(theta, family_code)
        
        # Apply rotation transformations to data
        u_rot, v_rot = u, v
        if rotation == 1:  # 180° rotation (survival)
            u_rot = 1 - u
            v_rot = 1 - v
        elif rotation == 2:  # 90° rotation
            u_rot = 1 - u
            # v_rot stays the same
        elif rotation == 3:  # 270° rotation
            # u_rot stays the same
            v_rot = 1 - v
        
        # Use absolute value of theta for the copula calculation
        theta_abs = np.maximum(np.abs(theta), 1e-6)
        
        t5 = u_rot ** (-theta_abs)
        t6 = v_rot ** (-theta_abs)
        t7 = t5 + t6 - 1.0
        t7 = np.maximum(t7, 1e-12)
        logpdf = (
            np.log(theta_abs + 1)
            - (theta_abs + 1) * (np.log(u_rot) + np.log(v_rot))
            - (2.0 + 1.0 / theta_abs) * np.log(t7)
        )
        logpdf = np.where(np.isfinite(logpdf), logpdf, np.log(1e-16))
        return np.full(len(y), logpdf)
    else:
        # Handle array case
        M = len(theta)
        logpdf = np.zeros(M)
        for i in range(M):
            theta_i = theta[i]
            rotation = _get_effective_rotation(theta_i, family_code)
            
            # Apply rotation transformations to data
            u_rot, v_rot = u[i], v[i]
            if rotation == 1:  # 180° rotation (survival)
                u_rot = 1 - u[i]
                v_rot = 1 - v[i]
            elif rotation == 2:  # 90° rotation
                u_rot = 1 - u[i]
                # v_rot stays the same
            elif rotation == 3:  # 270° rotation
                # u_rot stays the same
                v_rot = 1 - v[i]
            
            theta_abs_i = np.maximum(np.abs(theta_i), 1e-6)
            t5 = u_rot ** (-theta_abs_i)
            t6 = v_rot ** (-theta_abs_i)
            t7 = t5 + t6 - 1.0
            t7 = np.maximum(t7, 1e-12)
            logpdf[i] = (
                np.log(theta_abs_i + 1)
                - (theta_abs_i + 1) * (np.log(u_rot) + np.log(v_rot))
                - (2.0 + 1.0 / theta_abs_i) * np.log(t7)
            )
            if not np.isfinite(logpdf[i]):
                logpdf[i] = np.log(1e-16)
        
        return logpdf

def _clayton_derivative_1st(y, theta, family_code=301):
    """
    Computes the first derivative of the bivariate Clayton copula log-likelihood
    with respect to theta, supporting automatic rotation selection via family_code.

    Args:
        y (np.ndarray): Input data of shape (M, 2)
        theta (np.ndarray or float): Copula parameter(s), shape (M,) or scalar
        family_code (int): gamCopula family code (301, 302, 303, 304) for automatic rotation

    Returns:
        np.ndarray: First derivative, shape (M,)
    """
    M = y.shape[0]
    deriv = np.empty((M,), dtype=np.float64)
    u = np.clip(y[:, 0], 1e-12, 1 - 1e-12)
    v = np.clip(y[:, 1], 1e-12, 1 - 1e-12)

    for m in range(M):
        th = theta[m] if hasattr(theta, "__len__") else theta
        
        # Get effective rotation based on family code and parameter sign
        rotation = _get_effective_rotation(th, family_code)
        
        # Handle rotations - use absolute value for calculations
        if rotation == 0:  # Standard Clayton
            uu, vv, tth = u[m], v[m], abs(th)
            sign = 1.0 if th >= 0 else -1.0
        elif rotation == 1:  # 180° rotated Clayton (survival)
            uu, vv, tth = 1 - u[m], 1 - v[m], abs(th)
            sign = 1.0 if th >= 0 else -1.0
        elif rotation == 2:  # 90° rotated Clayton
            uu, vv, tth = 1 - u[m], v[m], abs(th)
            sign = -1.0 if th >= 0 else 1.0
        elif rotation == 3:  # 270° rotated Clayton
            uu, vv, tth = u[m], 1 - v[m], abs(th)
            sign = -1.0 if th >= 0 else 1.0
        else:
            raise NotImplementedError("Copula family not implemented.")

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

def _clayton_derivative_2nd(y, theta, family_code=301):
    """
    Second derivative of Clayton copula PDF w.r.t. parameter theta.
    Based on diff2PDF_mod from VineCopula C code with automatic rotation selection.
    
    Args:
        y: array of shape (n, 2) - copula data [u, v]
        theta: array of shape (n,) or scalar - copula parameters
        family_code: gamCopula family code (301, 302, 303, 304) for automatic rotation
    
    Returns:
        np.ndarray: second derivative values for each observation
    """
    
    # Constants for numerical stability
    UMIN = 1e-12
    UMAX = 1 - 1e-12
    
    # Extract u and v from the y matrix
    u = np.clip(y[:, 0], UMIN, UMAX)
    v = np.clip(y[:, 1], UMIN, UMAX)
    
    # Ensure arrays - FIX: Handle scalar theta properly
    u = np.atleast_1d(u)
    v = np.atleast_1d(v)
    
    # Fix the theta handling
    if np.isscalar(theta):
        theta = np.full(len(u), theta)
    else:
        theta = np.atleast_1d(theta)
    
    n = len(u)
    out = np.zeros(n)
    
    for i in range(n):
        u_i, v_i = u[i], v[i]
        theta_i = theta[i]  # Now this will work since theta is always an array
        
        # Get effective rotation based on family code and parameter sign
        rotation = _get_effective_rotation(theta_i, family_code)
        
        # Handle rotation transformations (following diff2PDF_mod structure)
        if rotation == 2:  # 90° rotated copulas
            negv = 1 - v_i
            u_transformed = u_i
            v_transformed = negv
            theta_transformed = abs(theta_i)  # Use absolute value
        elif rotation == 3:  # 270° rotated copulas
            negu = 1 - u_i
            u_transformed = negu
            v_transformed = v_i
            theta_transformed = abs(theta_i)  # Use absolute value
        elif rotation == 1:  # 180° rotated copulas (survival)
            negv = 1 - v_i
            negu = 1 - u_i
            u_transformed = negu
            v_transformed = negv
            theta_transformed = abs(theta_i)  # Use absolute value
        else:  # Standard copulas (including 3 = Clayton)
            u_transformed = u_i
            v_transformed = v_i
            theta_transformed = abs(theta_i)  # Use absolute value
        
        # Clip transformed values for numerical stability
        u_transformed = np.clip(u_transformed, UMIN, UMAX)
        v_transformed = np.clip(v_transformed, UMIN, UMAX)
        
        # Following the exact C code structure from diff2PDF function
        theta_val = theta_transformed  # Rename to avoid confusion with theta array
        
        # Basic terms (matching C variable names)
        t1 = u_transformed * v_transformed
        t2 = -theta_val - 1.0
        t3 = np.power(t1, t2)
        t4 = np.log(t1)
        
        t6 = np.power(u_transformed, -theta_val)
        t7 = np.power(v_transformed, -theta_val)
        t8 = t6 + t7 - 1.0
        
        t10 = -2.0 - 1.0/theta_val
        t11 = np.power(t8, t10)
        
        # Higher order terms
        t15 = theta_val * theta_val
        t16 = 1.0 / t15
        t17 = np.log(t8)
        
        t19 = np.log(u_transformed)
        t21 = np.log(v_transformed)
        
        t24 = -t6 * t19 - t7 * t21
        
        t26 = 1.0 / t8
        t27 = t16 * t17 + t10 * t24 * t26
        
        t30 = -t2 * t3
        t32 = t4 * t4
        t14 = t27 * t27
        t13 = t19 * t19
        t12 = t21 * t21
        t9 = t24 * t24
        t5 = t8 * t8
        
        # The complete second derivative expression from C code
        term1 = -2.0 * t3 * t4 * t11
        term2 = 2.0 * t3 * t11 * t27
        term3 = t30 * t32 * t11
        term4 = -2.0 * t30 * t4 * t11 * t27
        term5 = t30 * t11 * t14
        
        # Additional correction terms
        t67 = 2.0 / (t15 * theta_val)  # Triple derivative term
        t70 = t6 * t13 + t7 * t12  # Second log derivative terms
        t74 = t9 / t5  # Ratio correction
        
        correction = t30 * t11 * (-t67 * t17 + 2.0 * t16 * t24 * t26 + 
                                    t10 * t70 * t26 - t10 * t74)
        
        result = term1 + term2 + term3 + term4 + term5 + correction
        out[i] = result

    
    return out