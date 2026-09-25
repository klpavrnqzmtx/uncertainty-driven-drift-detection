"""Phase 2 tests: GMM oracle, drift schedule, TV bounds."""

from __future__ import annotations

import numpy as np

import uncertainty_driven_drift.components.phase2  # noqa: F401 — register components
from uncertainty_driven_drift.components.synthetic import (
    DriftEvent,
    GMMParams,
    _params_trajectory,
)


# ---------------------------------------------------------------------------
# Oracle posterior
# ---------------------------------------------------------------------------

def test_gmm_posterior_symmetry_and_normalization() -> None:
    params = GMMParams(
        priors=np.array([0.5, 0.5]),
        means=np.array([[-1.0, 0.0], [1.0, 0.0]]),
        covs=np.stack([np.eye(2), np.eye(2)]),
    )
    x = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    p = params.posterior(x)

    assert p.shape == (4, 2)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(p[0], [0.5, 0.5], atol=1e-10)
    assert p[1, 1] > 0.5  # x=+0.5 favours class 1
    assert p[2, 1] > p[1, 1]  # farther to the right ⇒ more confident
    np.testing.assert_allclose(p[3, 0], p[2, 1], atol=1e-10)  # mirror symmetry


def test_gmm_posterior_matches_bayes_rule_brute_force() -> None:
    rng = np.random.default_rng(0)
    K, D = 3, 2
    priors = np.array([0.2, 0.5, 0.3])
    means = rng.normal(size=(K, D))
    A = rng.normal(size=(K, D, D))
    covs = np.stack([A[k] @ A[k].T + np.eye(D) for k in range(K)])
    params = GMMParams(priors=priors, means=means, covs=covs)

    x = rng.normal(size=(16, D))
    got = params.posterior(x)

    # Closed-form reference using log-densities computed independently.
    from scipy.stats import multivariate_normal

    ref = np.zeros_like(got)
    for k in range(K):
        ref[:, k] = priors[k] * multivariate_normal(means[k], covs[k]).pdf(x)
    ref = ref / ref.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(got, ref, atol=1e-10)


# ---------------------------------------------------------------------------
# Drift schedule
# ---------------------------------------------------------------------------

def test_drift_trajectory_abrupt_and_gradual() -> None:
    base = GMMParams(
        priors=np.array([0.5, 0.5]),
        means=np.array([[0.0, 0.0], [1.0, 0.0]]),
        covs=np.stack([np.eye(2), np.eye(2)]),
    )
    target = GMMParams(
        priors=np.array([0.5, 0.5]),
        means=np.array([[10.0, 0.0], [11.0, 0.0]]),
        covs=np.stack([np.eye(2), np.eye(2)]),
    )

    # Abrupt event at t=5.
    traj = _params_trajectory(base, [DriftEvent(5, 5, target)], n_batches=10)
    np.testing.assert_allclose(traj[4].means, base.means)
    np.testing.assert_allclose(traj[5].means, target.means)
    np.testing.assert_allclose(traj[9].means, target.means)

    # Gradual event spanning t=5..9 (width 4).
    traj = _params_trajectory(base, [DriftEvent(5, 9, target)], n_batches=10)
    np.testing.assert_allclose(traj[4].means, base.means)
    np.testing.assert_allclose(traj[9].means, target.means)
    # Midpoint alpha = (7-5)/4 = 0.5 ⇒ mean halfway between base and target.
    np.testing.assert_allclose(
        traj[7].means, 0.5 * (base.means + target.means), atol=1e-12
    )


# ---------------------------------------------------------------------------
# TV distance properties
# ---------------------------------------------------------------------------

def test_tv_is_bounded_symmetric_and_zero_on_equal_inputs() -> None:
    rng = np.random.default_rng(0)
    B, K = 64, 4
    p1 = rng.dirichlet(np.ones(K), size=B)
    p2 = rng.dirichlet(np.ones(K), size=B)

    tv = 0.5 * np.abs(p1 - p2).sum(axis=1)
    assert np.all(tv >= 0.0)
    assert np.all(tv <= 1.0 + 1e-12)
    np.testing.assert_allclose(tv, 0.5 * np.abs(p2 - p1).sum(axis=1))
    np.testing.assert_allclose(
        0.5 * np.abs(p1 - p1).sum(axis=1), np.zeros(B), atol=0.0
    )
