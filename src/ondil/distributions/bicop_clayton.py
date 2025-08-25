# Author: Your Name
# MIT Licence
from typing import Dict

import numpy as np
import scipy.stats as st

from ..base import Distribution, LinkFunction, CopulaMixin
from ..links import LogShiftValue, KendallsTauToParameterClayton
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
        link: LinkFunction = LogShiftValue(1e-3),
        param_link: LinkFunction = KendallsTauToParameterClayton(),
    ):
        super().__init__(
            links={0: link},
            param_links={0: param_link},
        )
        self.is_multivariate = True
        self._adr_lower_diag = {0: False}
        self._regularization_allowed = {0: False}
        self._regularization = ""
        self._scoring = "fisher"

    @staticmethod
    def fitted_elements(dim: int):
        return {0: int(dim * (dim - 1) // 2)}

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
        return np.maximum(np.asarray(theta), 1e-6)

    def theta_to_scipy_params(self, theta: np.ndarray) -> dict:
        return {"theta": theta}

    def set_initial_guess(self, theta, param):
        return theta

    def initial_values(self, y, param=0):
        M = y.shape[0]
        tau = st.kendalltau(y[:, 0], y[:, 1]).correlation
        theta = 2 * tau / (1 - tau)
        theta = np.maximum(theta, 1e-3)
        return np.full((M,), theta)  # 1D array

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
        result = _clayton_logpdf(y, theta)
        return np.asarray(result).reshape(-1)  # Always 1D

    def pdf(self, y, theta):
        return np.exp(self.logpdf(y, theta))

    def dl1_dp1(self, y, theta, param=0):
        theta = self.theta_to_params(theta)
        return _clayton_derivative_1st(y, theta)

    def dl2_dp2(self, y, theta, param=0, clip=False):
        theta = self.theta_to_params(theta)
        return _clayton_derivative_2nd(y, theta)

    def element_score(self, y, theta, param=0, k=0):
        return self.element_dl1_dp1(y, theta, param, k)

    def element_hessian(self, y, theta, param=0, k=0):
        return self.element_dl2_dp2(y, theta, param, k)

    def element_dl1_dp1(self, y, theta, param=0, k=0, clip=False):
        theta = self.theta_to_params(theta)
        return _clayton_derivative_1st(y, theta)

    def element_dl2_dp2(self, y, theta, param=0, k=0, clip=False):
        theta = self.theta_to_params(theta)
        return _clayton_derivative_2nd(y, theta)

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

        if un == 2:
            u, v = v, u

        h = (u ** (-theta - 1)) * (u ** (-theta) + v ** (-theta) - 1) ** (-1 / theta - 1)
        h = np.clip(h, UMIN, UMAX)
        return h

##########################################################
# Helper functions for the log-likelihood and derivatives #
##########################################################

def _clayton_logpdf(y, theta):
    u = np.clip(y[:, 0], 1e-12, 1 - 1e-12)
    v = np.clip(y[:, 1], 1e-12, 1 - 1e-12)
    theta = np.maximum(theta, 1e-6)
    t5 = u ** (-theta)
    t6 = v ** (-theta)
    t7 = t5 + t6 - 1.0
    t7 = np.maximum(t7, 1e-12)
    logpdf = (
        np.log(theta + 1)
        - (theta + 1) * (np.log(u) + np.log(v))
        - (2.0 + 1.0 / theta) * np.log(t7)
    )
    logpdf = np.where(np.isfinite(logpdf), logpdf, np.log(1e-16))
    return logpdf.reshape(-1)  # Always 1D

def _clayton_derivative_1st(y, theta):
    u = np.clip(y[:, 0], 1e-12, 1 - 1e-12)
    v = np.clip(y[:, 1], 1e-12, 1 - 1e-12)
    theta = np.maximum(theta, 1e-6)
    t5 = u ** (-theta)
    t6 = v ** (-theta)
    t7 = t5 + t6 - 1.0
    t7 = np.maximum(t7, 1e-12)
    t9 = theta ** 2
    t14 = np.log(u)
    t16 = np.log(v)
    result = (
        1.0 / (1.0 + theta)
        - (np.log(u) + np.log(v))
        + (1.0 / t9) * np.log(t7)
        + (1.0 / theta + 2.0) * (t5 * t14 + t6 * t16) / t7
    )
    return np.asarray(result).reshape(-1)  # Always 1D

def _clayton_derivative_2nd(y, theta):
    M = y.shape[0]
    deriv = np.zeros((M,), dtype=np.float64)
    h = 1e-5
    theta = np.asarray(theta).reshape(-1)
    for m in range(M):
        t = theta[m] if theta.size > 1 else theta[0]
        ll_p = _clayton_logpdf(y[m:m+1], t + h)
        ll_0 = _clayton_logpdf(y[m:m+1], t)
        ll_m = _clayton_logpdf(y[m:m+1], t - h)
        deriv[m] = (ll_p - 2 * ll_0 + ll_m) / (h ** 2)
    return deriv  # Always 1D