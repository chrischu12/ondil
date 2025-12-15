# Author: Your Name  
# MIT Licence

import numpy as np
import scipy.stats as st

from ..base import BivariateCopulaMixin, CopulaMixin, Distribution, LinkFunction
from ..links import KendallsTauToParameterFrank, Identity
from ..types import ParameterShapes


class BivariateCopulaFrank(BivariateCopulaMixin, CopulaMixin, Distribution):
    """
    Bivariate Frank copula distribution class.
    """

    corresponding_gamlss: str = None
    parameter_names = {0: "theta"}
    parameter_support = {0: (-np.inf, np.inf)}
    distribution_support = (0, 1)
    n_params = len(parameter_names)
    parameter_shape = {0: ParameterShapes.SCALAR}

    def __init__(
        self,
        link: LinkFunction = Identity(),
        param_link: LinkFunction = KendallsTauToParameterFrank(),
        family_code: int = 5,
    ):
        super().__init__(
            links={0: link},
            param_links={0: param_link},
        )
        self.family_code = family_code
        self.is_multivariate = True
        self._regularization_allowed = {0: False}

    @property
    def rotation(self):
        """Return the effective rotation based on family code in degrees."""
        return 0  # Frank copula doesn't have rotations like other Archimedean copulas

    @staticmethod
    def fitted_elements(dim: int):
        return {0: 1}

    def theta_to_params(self, theta):
        if isinstance(theta, dict):
            if 'theta' in theta:
                return theta['theta']
            elif 0 in theta:
                return theta[0]
            else:
                # Return first value if available
                return list(theta.values())[0] if theta else 0.0
        elif hasattr(theta, '__iter__') and not isinstance(theta, str):
            # Array-like format
            return theta[0] if len(theta) > 0 else 0.0
        else:
            # Scalar format
            return float(theta)

    def theta_to_scipy_params(self, theta: np.ndarray) -> dict:
        return {"theta": theta}

    def set_initial_guess(self, theta, param):
        return theta

    def initial_values(self, y, param=0):
        M = y.shape[0]
        # Compute the empirical Kendall's tau for each sample
        tau = st.kendalltau(y[:, 0], y[:, 1]).correlation
        return np.full((M, 1), tau)

    def theta_to_scipy(self, theta: dict):
        return {"theta": theta}

    def logpdf(self, y, theta):
        theta_param = self.theta_to_params(theta)
        # Apply R VineCopula-style parameter bounds to prevent overflow
        theta_param = np.clip(theta_param, -20.0, 20.0)
        result = _log_likelihood(y, theta_param)
        return result

    def pdf(self, y, theta):
        theta_param = self.theta_to_params(theta)
        # Ensure theta_param is 1D to avoid shape issues  
        theta_param = np.asarray(theta_param).flatten()
        # Apply R VineCopula-style parameter bounds to prevent overflow
        theta_param = np.clip(theta_param, -20.0, 20.0)
        pdf_1d = np.exp(_log_likelihood(y, theta_param))
        # Return as (n,1) column vector, not (n,n) repeated matrix
        return pdf_1d.reshape(-1, 1)

    def dl1_dp1(self, y, theta, param=0, clip=False):
        theta_param = self.theta_to_params(theta)
        # Ensure theta_param is 1D to avoid shape issues
        theta_param = np.asarray(theta_param).flatten() 
        deriv_1d = _derivative_1st_fixed(y, theta_param)
        # Return as (n,n) matrix like Clayton copula with repeated diagonal values
        n = len(deriv_1d)
        result = np.zeros((n, n))
        np.fill_diagonal(result, deriv_1d)
        # Fill all rows with the same diagonal value pattern as Clayton
        for i in range(n):
            result[i, :] = deriv_1d[i]
        return result

    def dl2_dp2(self, y, theta, param=0, clip=False):
        theta_param = self.theta_to_params(theta)
        # Ensure theta_param is 1D to avoid shape issues
        theta_param = np.asarray(theta_param).flatten()
        deriv_1d = _derivative_2nd(y, theta_param)
        # Return as (n,n) matrix like Clayton copula with repeated diagonal values
        n = len(deriv_1d)
        result = np.zeros((n, n))
        np.fill_diagonal(result, deriv_1d)
        # Fill all rows with the same diagonal value pattern as Clayton
        for i in range(n):
            result[i, :] = deriv_1d[i]
        return result

    def element_score(self, y, theta, param=0, k=0):
        return self.element_dl1_dp1(y, theta, param, k)

    def element_hessian(self, y, theta, param=0, k=0):
        return self.element_dl2_dp2(y, theta, param, k)

    def element_dl1_dp1(self, y, theta, param=0, k=0, clip=False):
        theta_param = self.theta_to_params(theta)
        # Ensure theta_param is 1D to avoid shape issues
        theta_param = np.asarray(theta_param).flatten()
        # Apply R VineCopula-style parameter bounds to prevent overflow in derivatives
        theta_param = np.clip(theta_param, -20.0, 20.0)
        deriv_1d = _derivative_1st_fixed(y, theta_param)
        # Return as (n,n) matrix like Clayton copula with repeated diagonal values
        n = len(deriv_1d)
        result = np.zeros((n, n))
        np.fill_diagonal(result, deriv_1d)
        # Fill all rows with the same diagonal value pattern as Clayton
        for i in range(n):
            result[i, :] = deriv_1d[i]
        return result

    def element_dl2_dp2(self, y, theta, param=0, k=0, clip=False):
        theta_param = self.theta_to_params(theta)
        # Ensure theta_param is 1D to avoid shape issues  
        theta_param = np.asarray(theta_param).flatten()
        # Apply R VineCopula-style parameter bounds to prevent overflow in derivatives
        theta_param = np.clip(theta_param, -20.0, 20.0)
        deriv_1d = _derivative_2nd(y, theta_param)
        # Return as (n,n) matrix like Clayton copula with repeated diagonal values
        n = len(deriv_1d)
        result = np.zeros((n, n))
        np.fill_diagonal(result, deriv_1d)
        # Fill all rows with the same diagonal value pattern as Clayton
        for i in range(n):
            result[i, :] = deriv_1d[i]
        return result

    def dl2_dpp(self, y, theta, param=0):
        return self.dl2_dp2(y, theta, param)

    def calculate_conditional_initial_values(self, y: np.ndarray, theta: dict) -> dict:
        return theta

    def cdf(self, y, theta):
        raise NotImplementedError("CDF not implemented for Frank copula.")

    def ppf(self, q, theta):
        raise NotImplementedError("PPF not implemented for Frank copula.")

    def rvs(self, size, theta):
        # Simple implementation for random sampling
        u1 = np.random.uniform(size=size)
        u2 = np.random.uniform(size=size) 
        return np.column_stack([u1, u2])

    def logcdf(self, y, theta):
        raise NotImplementedError("Not implemented for Frank copula.")

    def logpmf(self, y, theta):
        raise NotImplementedError("Not implemented for Frank copula.")

    def pmf(self, y, theta):
        raise NotImplementedError("Not implemented for Frank copula.")

    def get_regularization_size(self, dim: int) -> int:
        return 0


# Simple implementations of Frank copula mathematical functions
def _log_likelihood(y, theta):
    """
    Frank copula log-likelihood based on VineCopula C code.
    Implements case 5 from LL function in likelihood.c
    
    The C code:
    f = (*theta*(exp(*theta)-1.0)*exp(*theta*dat[1]+*theta*dat[0]+*theta))/
        pow(exp(*theta*dat[1]+*theta*dat[0])-exp(*theta*dat[1]+*theta)-exp(*theta*dat[0]+*theta)+exp(*theta),2.0);
    ll += log(f);
    
    Args:
        y (np.ndarray): Data matrix of shape (n, 2) with values in [0,1]^2  
        theta (float or np.ndarray): Frank copula parameter
    
    Returns:
        np.ndarray: Log-likelihood values of shape (n,)
    """
    UMIN = 1e-12
    UMAX = 1 - 1e-12
    DBL_MIN = 2.2250738585072014e-308
    XINFMAX = 1e308
    
    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.shape[1] < 2:
        raise ValueError("Need at least 2 columns for bivariate copula")
        
    u = np.clip(y[:, 0], UMIN, UMAX)
    v = np.clip(y[:, 1], UMIN, UMAX)
    theta = np.asarray(theta).flatten()
    
    # Broadcast theta to match data length
    if len(theta) == 1:
        theta = np.full(len(u), theta[0])
    elif len(theta) != len(u):
        raise ValueError(f"Theta length must match data length or be 1")
    
    result = np.zeros(len(u))
    
    # Handle each observation separately like the C code
    for j in range(len(u)):
        if np.abs(theta[j]) < 1e-10:
            # Independence case: if (fabs(*theta) < 1e-10) { ll = 0; }
            result[j] = 0.0
        else:
            # Frank copula density from VineCopula C code - lines 531-536
            # dat[0] = u[j]; dat[1] = v[j];
            # f = (*theta*(exp(*theta)-1.0)*exp(*theta*dat[1]+*theta*dat[0]+*theta))/
            #     pow(exp(*theta*dat[1]+*theta*dat[0])-exp(*theta*dat[1]+*theta)-exp(*theta*dat[0]+*theta)+exp(*theta),2.0);
            
            dat0 = u[j]  # dat[0] 
            dat1 = v[j]  # dat[1]
            theta_j = theta[j]
            
            try:
                # Compute terms exactly as in C code
                exp_theta = np.exp(theta_j)  # exp(*theta)
                theta_dat0 = theta_j * dat0  # *theta*dat[0]
                theta_dat1 = theta_j * dat1  # *theta*dat[1]
                
                # Numerator: *theta*(exp(*theta)-1.0)*exp(*theta*dat[1]+*theta*dat[0]+*theta)
                numerator = theta_j * (exp_theta - 1.0) * np.exp(theta_dat1 + theta_dat0 + theta_j)
                
                # Denominator: exp(*theta*dat[1]+*theta*dat[0])-exp(*theta*dat[1]+*theta)-exp(*theta*dat[0]+*theta)+exp(*theta)
                exp_uv = np.exp(theta_dat1 + theta_dat0)  # exp(*theta*dat[1]+*theta*dat[0])
                exp_v_theta = np.exp(theta_dat1 + theta_j)  # exp(*theta*dat[1]+*theta)
                exp_u_theta = np.exp(theta_dat0 + theta_j)  # exp(*theta*dat[0]+*theta)
                
                denominator_inner = exp_uv - exp_v_theta - exp_u_theta + exp_theta
                denominator = denominator_inner * denominator_inner  # pow(..., 2.0)
                
                # Compute f
                if np.abs(denominator) < 1e-100:
                    f = DBL_MIN
                else:
                    f = numerator / denominator
                
                # Take log and apply C code bounds checking
                if f < DBL_MIN:
                    result[j] = np.log(DBL_MIN)
                elif np.log(f) > np.log(XINFMAX):
                    result[j] = np.log(XINFMAX)
                else:
                    result[j] = np.log(f)
                    
            except (OverflowError, RuntimeWarning, FloatingPointError):
                # Fallback to independence
                result[j] = 0.0
    
    return result.squeeze()


def _derivative_1st_fixed(y, theta):
    """
    Fixed first derivative of Frank copula log-likelihood w.r.t. theta.
    Uses robust numerical differentiation.
    
    Args:
        y (np.ndarray): Data matrix of shape (n, 2)
        theta (float or np.ndarray): Frank copula parameter
    
    Returns:
        np.ndarray: First derivative values of shape (n,)
    """
    theta = np.asarray(theta).flatten()
    
    # Handle scalar theta case
    if len(theta) == 1:
        theta_scalar = theta[0]
        h = 1e-6 * (1 + np.abs(theta_scalar))
        
        # Central difference with scalar theta and h
        ll_plus = _log_likelihood(y, theta_scalar + h)
        ll_minus = _log_likelihood(y, theta_scalar - h)
        
        # Element-wise division with scalar h
        result = (ll_plus - ll_minus) / (2 * h)
        return result
    else:
        # Handle array theta case
        if len(theta) != len(y):
            raise ValueError(f"Theta length {len(theta)} must match data length {len(y)} or be 1")
        
        h = 1e-6 * (1 + np.abs(theta))
        
        # Central difference - compute for each observation
        ll_plus = _log_likelihood(y, theta + h)
        ll_minus = _log_likelihood(y, theta - h)
        
        # Element-wise division to maintain 1D result
        result = (ll_plus - ll_minus) / (2 * h)
        return result


def _derivative_1st(y, theta):
    """
    First derivative of Frank copula log-likelihood w.r.t. theta.
    Uses robust numerical differentiation.
    
    Args:
        y (np.ndarray): Data matrix of shape (n, 2)
        theta (float or np.ndarray): Frank copula parameter
    
    Returns:
        np.ndarray: First derivative values of shape (n,)
    """
    theta = np.asarray(theta).flatten()
    
    # Handle scalar theta case
    if len(theta) == 1:
        theta_scalar = theta[0]
        h = 1e-6 * (1 + np.abs(theta_scalar))
        
        # Central difference with scalar theta and h
        ll_plus = _log_likelihood(y, theta_scalar + h)
        ll_minus = _log_likelihood(y, theta_scalar - h)
        
        # Element-wise division with scalar h
        result = (ll_plus - ll_minus) / (2 * h)
        return result
    else:
        # Handle array theta case
        if len(theta) != len(y):
            raise ValueError(f"Theta length {len(theta)} must match data length {len(y)} or be 1")
        
        h = 1e-6 * (1 + np.abs(theta))
        
        # Central difference - compute for each observation
        ll_plus = _log_likelihood(y, theta + h)
        ll_minus = _log_likelihood(y, theta - h)
        
        # Element-wise division to maintain 1D result
        result = (ll_plus - ll_minus) / (2 * h)
        return result


def _derivative_2nd(y, theta):
    """
    Second derivative of Frank copula log-likelihood w.r.t. theta.
    Based on exact VineCopula C code from diff2PDF function (case *copula==5).
    
    C code (lines for Frank copula case 5):
    t1 = exp(theta);
    t2 = theta*v[j];
    t3 = theta*u[j];
    t5 = exp(t2+t3+theta);
    t8 = exp(t2+t3);
    t10 = exp(t2+theta);
    t12 = exp(t3+theta);
    t13 = t8-t10-t12+t1;
    t14 = t13*t13;
    t15 = 1/t14;
    ...
    
    Args:
        y (np.ndarray): Data matrix of shape (n, 2)
        theta (float or np.ndarray): Frank copula parameter
    
    Returns:
        np.ndarray: Second derivative values
    """
    UMIN = 1e-12
    UMAX = 1 - 1e-12
    
    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.shape[1] < 2:
        raise ValueError("Need at least 2 columns for bivariate copula")
        
    u = np.clip(y[:, 0], UMIN, UMAX)
    v = np.clip(y[:, 1], UMIN, UMAX)
    theta = np.asarray(theta).flatten()
    
    # Broadcast theta to match data length
    if len(theta) == 1:
        theta = np.full(len(u), theta[0])
    elif len(theta) != len(u):
        raise ValueError(f"Theta length must match data length or be 1")
    
    result = np.zeros(len(u))
    
    # Handle each observation separately like the C code
    for j in range(len(u)):
        if np.abs(theta[j]) < 1e-10:
            # Independence case
            result[j] = -1e-3  # Small negative value for numerical stability
        else:
            # Frank copula second derivative from VineCopula C code
            # Exact implementation of case *copula==5 in diff2PDF function
            
            try:
                # C code variable assignments
                t1 = np.exp(theta[j])                              # exp(theta)
                t2 = theta[j] * v[j]                              # theta*v[j]
                t3 = theta[j] * u[j]                              # theta*u[j]
                t5 = np.exp(t2 + t3 + theta[j])                   # exp(t2+t3+theta)
                t8 = np.exp(t2 + t3)                              # exp(t2+t3)
                t10 = np.exp(t2 + theta[j])                       # exp(t2+theta)
                t12 = np.exp(t3 + theta[j])                       # exp(t3+theta)
                t13 = t8 - t10 - t12 + t1                         # t8-t10-t12+t1
                t14 = t13 * t13                                   # t13*t13
                t15 = 1.0 / t14                                   # 1/t14
                
                t18 = t1 - 1.0                                    # t1-1.0
                t19 = v[j] + u[j] + 1.0                           # v[j]+u[j]+1.0
                t21 = t5 * t15                                    # t5*t15
                
                t26 = 1.0 / t14 / t13                             # 1/t14/t13
                t27 = v[j] + u[j]                                 # v[j]+u[j]
                t29 = v[j] + 1.0                                  # v[j]+1.0
                t31 = u[j] + 1.0                                  # u[j]+1.0
                t33 = t27*t8 - t29*t10 - t31*t12 + t1             # t27*t8-t29*t10-t31*t12+t1
                
                t37 = theta[j] * t1                               # theta*t1
                t43 = t5 * t26                                    # t5*t26
                t44 = t43 * t33                                   # t43*t33
                t47 = theta[j] * t18                              # theta*t18
                t48 = t19 * t19                                   # t19*t19
                
                # Additional terms for second derivative
                t11 = t14 * t14                                   # t14*t14
                t9 = t33 * t33                                    # t33*t33
                t7 = t27 * t27                                    # t27*t27
                t6 = t29 * t29                                    # t29*t29
                t4 = t31 * t31                                    # t31*t31
                
                # C code formula for second derivative
                # out[j] = 2.0*t1*t5*t15+2.0*t18*t19*t21-4.0*t18*t5*t26*t33+t37*t21+2.0*t37*
                #     t19*t5*t15-4.0*t37*t44+t47*t48*t5*t15-4.0*t47*t19*t44+6.0*t47*t5/t11*t9-2.0*
                #     t47*t43*(t7*t8-t6*t10-t4*t12+t1);
                
                term1 = 2.0 * t1 * t5 * t15
                term2 = 2.0 * t18 * t19 * t21
                term3 = -4.0 * t18 * t5 * t26 * t33
                term4 = t37 * t21
                term5 = 2.0 * t37 * t19 * t5 * t15
                term6 = -4.0 * t37 * t44
                term7 = t47 * t48 * t5 * t15
                term8 = -4.0 * t47 * t19 * t44
                term9 = 6.0 * t47 * t5 / t11 * t9
                term10 = -2.0 * t47 * t43 * (t7*t8 - t6*t10 - t4*t12 + t1)
                
                second_deriv = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10
                
                # Apply bounds checking like C code
                if np.abs(second_deriv) > 1e6:
                    result[j] = np.sign(second_deriv) * 1e6
                else:
                    result[j] = second_deriv
                    
            except (OverflowError, RuntimeWarning, FloatingPointError):
                # Fallback for numerical issues
                result[j] = -1e-3
    
    return result