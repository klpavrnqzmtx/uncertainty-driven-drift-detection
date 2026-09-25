"""Real-world tabular streaming datasets from the river library.

Both adapters load the full dataset into memory at construction time (the
sizes — ~45k and ~53k rows — are small enough to fit comfortably), then
expose them as :class:`~uncertainty_driven_drift.data.base.StreamBatch` iterators
compatible with the framework's prequential loop.

Design choices
--------------
* **Elec2** is binary (electricity price UP/DOWN), so it uses
  ``BayesianLogisticRegression + LaplaceMCUncertainty`` directly.
* **Insects** carries six species labels (strings); we binarize by integer
  value split — species {2, 3, 4} → class 0, species {5, 11, 12} → class 1 —
  so the same binary stack applies.  Drift indices are inferred from
  batch-level majority-label group changes.
* Features are z-score normalised at load time so gradient-based models
  converge without hand-tuning learning rates.
* ``drift_indices`` for Elec2 are empty (no labelled change-points in the
  original data); the Figure-3 table marks its "true alarms" using the
  supervised DDM/EDDM output as a proxy.
"""

from __future__ import annotations

from typing import Iterator, List, Sequence

import numpy as np

from uncertainty_driven_drift.data.base import DatasetStream, StreamBatch, StreamSpec
from uncertainty_driven_drift.registry import register


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _z_score(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    return (X - mean) / std




# ---------------------------------------------------------------------------
# Elec2
# ---------------------------------------------------------------------------

@register("dataset", "elec2")
class Elec2Stream(DatasetStream):
    """Electricity pricing stream (Harries 1999) via the river library.

    45 312 samples, 7 features (date is dropped), binary classification
    (price UP=1 / DOWN=0).  The dataset exhibits temporal concept drift
    due to seasonal demand patterns; no labelled drift indices are
    available so downstream table analysis uses the supervised baseline
    as a proxy.

    Parameters
    ----------
    batch_size :
        Samples per streaming batch.
    seed :
        Controls per-batch shuffling within the stream (default 0 = no
        shuffle, preserving the original temporal order which is the
        natural drift benchmark).
    normalize :
        Z-score features at load time.
    drop_features :
        Feature keys to exclude.  ``date`` is a raw day counter that
        leaks the timestamp, so it is dropped by default.
    """

    def __init__(
        self,
        batch_size: int = 256,
        seed: int = 0,
        normalize: bool = True,
        drop_features: Sequence[str] = ("date",),
    ) -> None:
        from river.datasets import Elec2 as _Elec2  # type: ignore

        drop = set(drop_features)
        rows_x: List[List[float]] = []
        rows_y: List[int] = []
        feature_names: List[str] | None = None

        for x_dict, y in _Elec2():
            if feature_names is None:
                feature_names = [k for k in x_dict if k not in drop]
            rows_x.append([float(x_dict[k]) for k in feature_names])
            rows_y.append(int(bool(y)))

        X = np.array(rows_x, dtype=np.float32)
        y_arr = np.array(rows_y, dtype=np.int64)
        if normalize:
            X = _z_score(X).astype(np.float32)

        self._X = X
        self._y = y_arr
        self.batch_size = int(batch_size)
        self.seed = int(seed)

        n_batches = len(X) // batch_size
        self.spec = StreamSpec(
            name="elec2",
            input_shape=(X.shape[1],),
            n_classes=2,
            n_batches=n_batches,
            batch_size=batch_size,
            drift_indices=(),
            has_true_posterior=False,
            extras={"n_features": int(X.shape[1]), "feature_names": feature_names or []},
        )

    def __iter__(self) -> Iterator[StreamBatch]:
        n = len(self._X)
        for t in range(self.spec.n_batches):
            s = t * self.batch_size
            e = min(s + self.batch_size, n)
            yield StreamBatch(
                index=t,
                x=self._X[s:e],
                y=self._y[s:e],
                concept_id=0,
                is_drift=False,
                true_posterior=None,
            )


# ---------------------------------------------------------------------------
# Insects
# ---------------------------------------------------------------------------

# Species whose integer label is > 4 are mapped to class 1; the rest to 0.
# This splits the six species in abrupt_balanced ({2,3,4} vs {5,11,12})
# into two balanced groups, preserving the abrupt drift structure while
# keeping the problem binary for BayesianLogisticRegression.
_INSECTS_THRESHOLD = 4


@register("dataset", "insects")
class InsectsStream(DatasetStream):
    """Insects wing-beat frequency stream via the river library.

    Loads the requested variant (default ``abrupt_balanced``, 52 848 samples,
    33 features, 6 species labels).  Labels are binarized by integer value:
    species {2, 3, 4} → class 0; species {5, 11, 12} → class 1.  Drift
    indices are inferred from batch-level majority-label group switches so
    the table can count true-positive alarms.

    Parameters
    ----------
    variant :
        River Insects variant (e.g. ``abrupt_balanced``,
        ``gradual_balanced``).
    batch_size :
        Samples per streaming batch.
    seed :
        RNG seed (reserved for future shuffling; stream is kept in
        original temporal order by default).
    normalize :
        Z-score features at load time.
    """

    def __init__(
        self,
        variant: str = "abrupt_balanced",
        batch_size: int = 256,
        seed: int = 0,
        normalize: bool = True,
    ) -> None:
        from river.datasets import Insects as _Insects  # type: ignore

        rows_x: List[List[float]] = []
        raw_labels: List[int] = []

        for x_dict, y in _Insects(variant=variant):
            rows_x.append(list(x_dict.values()))
            raw_labels.append(int(str(y)))

        X = np.array(rows_x, dtype=np.float32)
        y_arr = np.array(
            [0 if lbl <= _INSECTS_THRESHOLD else 1 for lbl in raw_labels],
            dtype=np.int64,
        )
        if normalize:
            X = _z_score(X).astype(np.float32)

        self._X = X
        self._y = y_arr
        self.batch_size = int(batch_size)
        self.seed = int(seed)

        n_batches = len(X) // batch_size

        self.spec = StreamSpec(
            name="insects",
            input_shape=(X.shape[1],),
            n_classes=2,
            n_batches=n_batches,
            batch_size=batch_size,
            drift_indices=(),  # labels are interleaved; drift is in feature space
            has_true_posterior=False,
            extras={
                "variant": variant,
                "n_features": int(X.shape[1]),
                "binarize_threshold": _INSECTS_THRESHOLD,
            },
        )

    def __iter__(self) -> Iterator[StreamBatch]:
        n = len(self._X)
        drift_set = set(self.spec.drift_indices)
        for t in range(self.spec.n_batches):
            s = t * self.batch_size
            e = min(s + self.batch_size, n)
            yield StreamBatch(
                index=t,
                x=self._X[s:e],
                y=self._y[s:e],
                concept_id=0,
                is_drift=(t in drift_set),
                true_posterior=None,
            )
