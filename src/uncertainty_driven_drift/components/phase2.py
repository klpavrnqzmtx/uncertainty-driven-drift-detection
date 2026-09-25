"""Phase 2 component bundle.

Importing this module registers the Gaussian-mixture stream, the Bayesian
logistic regression model, and the Laplace-MC uncertainty estimator.
"""

from uncertainty_driven_drift.components import bayes_logreg  # noqa: F401
from uncertainty_driven_drift.components import laplace_mc   # noqa: F401
from uncertainty_driven_drift.components import synthetic    # noqa: F401
