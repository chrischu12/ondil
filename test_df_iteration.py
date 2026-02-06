import pandas as pd
import numpy as np
import sys

# Add ondil to path
ondil_path = r"C:\Users\alvar\Documents\Essen\Copula\rolch\src"
if ondil_path not in sys.path:
    sys.path.insert(0, ondil_path)

from ondil.distributions import BivariateCopulaStudentT
from ondil.links import FisherZLink, LogShiftTwo, Identity, KendallsTauToParameter
from ondil.estimators import MultivariateOnlineDistributionalRegressionPath

print("DF ITERATION TEST")
print("=" * 20)

# Load data
merged_data_t = pd.read_csv("C:/Users/alvar/Documents/Essen/merged_data_t.csv")
y_numpy = merged_data_t[["u1", "u2"]].to_numpy()
X_numpy = merged_data_t.drop(columns=["u1", "u2"]).to_numpy()

# Setup with both parameters  
H = 2
equation = {
    0: {h: np.arange(X_numpy.shape[1]) for h in range(H)},
    1: {0: 'intercept'}
}

distribution = BivariateCopulaStudentT(
    link_1=FisherZLink(), param_link_1=KendallsTauToParameter(),  
    link_2=LogShiftTwo(), param_link_2=Identity()
)

estimator = MultivariateOnlineDistributionalRegressionPath(
    distribution=distribution, equation=equation, method="ols",
    early_stopping=False, verbose=2, max_iterations_inner=2, 
    max_iterations_outer=2, scale_inputs=False
)

initial_nu = distribution.initial_values(y_numpy, param=1)[0][0]
print(f"Initial nu: {initial_nu:.6f}")

estimator.fit(X_numpy, y_numpy)

final_nu = 2 + np.exp(estimator.coef_[1][0].item())
print(f"Final nu: {final_nu:.6f} (change: {final_nu-initial_nu:+.3f})")

if abs(final_nu - initial_nu) > 0.1:
    print("SUCCESS: DF iteration working!")
else:
    print("WARNING: DF iteration may not be working")