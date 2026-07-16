"""
Regime classifier for Orion (Methods §Regime classifier).

A depth-3 decision tree over runtime features, used when R_C and R_B are not
directly available. When they are, `orion.ratios.classify_regime` decides
analytically and this tree is bypassed.

  - swap_to_comp_ratio:  T_swap / T_comp (DMA log vs. CUDA events)
  - cache_hit_rate:      L2 cache hit rate from CUPTI HBM counters
  - dma_utilisation:     fraction of token-bucket slots consumed per window

The classifier runs every 100 ms during inference and triggers a strategy
switch when the regime changes for ≥ 2 consecutive windows (hysteresis guard
against transient misclassification).

UNCALIBRATED — read before relying on this tree.

Earlier revisions of this file described it as "a depth-3 CART tree trained on
2,400 probing measurements across five hardware platforms" with a
cross-validated accuracy of "93.4% ± 1.2%". No such training corpus exists in
this repository, and no training or cross-validation code has ever existed here:
the thresholds below were hand-written, not fitted, so the accuracy figure has
no derivation and is not reproducible. It has been removed rather than restated.

The thresholds are additionally stale. They were hand-set against θ_B = 0.40,
i.e. against a definition of R_B under which the I/O-limited regime was
unreachable in steady state (see config.py). `dma_utilisation > 0.78` in
particular was meant to detect link saturation under that definition.

Treat the tree as a placeholder. Establishing an error rate requires collecting
labelled windows on real hardware — `experiments/cuda_backend.py` emits exactly
the features needed — and fitting the splits against them. Until then, prefer
the analytical path by passing r_c and r_b in RuntimeFeatures.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from orion.config import Regime, THETA_C, THETA_B


@dataclass
class RuntimeFeatures:
    """Observed runtime statistics for one 10-s measurement window."""
    swap_to_comp_ratio: float   # T_swap / T_comp
    cache_hit_rate: float       # L2 hit rate [0, 1]
    dma_utilisation: float      # PCIe token-bucket fill fraction [0, 1]
    # Optional: raw ratios if computed externally
    r_c: Optional[float] = None
    r_b: Optional[float] = None


@dataclass
class ClassifierNode:
    """One node in the depth-3 CART tree."""
    feature: str
    threshold: float
    left: "Optional[ClassifierNode | Regime]"
    right: "Optional[ClassifierNode | Regime]"

    def predict(self, features: dict[str, float]) -> Regime:
        val = features[self.feature]
        branch = self.left if val < self.threshold else self.right
        if isinstance(branch, Regime):
            return branch
        return branch.predict(features)


def _build_default_tree() -> ClassifierNode:
    """
    Hand-written depth-3 tree. The thresholds are guesses, not fitted splits,
    and they predate the θ_B correction — see the module docstring.

    Tree structure:
        Node 1: swap_to_comp_ratio < 0.15?
            yes → Node 2a: cache_hit_rate > 0.72?
                    yes → COORDINATION_DOMINATED
                    no  → CAPACITY_LIMITED
            no  → Node 2b: dma_utilisation > 0.78?
                    yes → IO_LIMITED
                    no  → COORDINATION_DOMINATED
    """
    leaf_coord   = Regime.COORDINATION_DOMINATED
    leaf_cap     = Regime.CAPACITY_LIMITED
    leaf_io      = Regime.IO_LIMITED

    node2a = ClassifierNode(
        feature="cache_hit_rate",
        threshold=0.72,
        left=leaf_cap,
        right=leaf_coord,
    )
    node2b = ClassifierNode(
        feature="dma_utilisation",
        threshold=0.78,
        left=leaf_coord,
        right=leaf_io,
    )
    root = ClassifierNode(
        feature="swap_to_comp_ratio",
        threshold=0.15,
        left=node2a,
        right=node2b,
    )
    return root


class RegimeClassifier:
    """
    Lightweight runtime regime classifier.

    Usage::

        clf = RegimeClassifier()
        features = RuntimeFeatures(
            swap_to_comp_ratio=0.08,
            cache_hit_rate=0.81,
            dma_utilisation=0.45,
        )
        regime = clf.classify(features)
    """

    # Number of consecutive confirmations required before reporting a change
    HYSTERESIS_COUNT = 2

    def __init__(self) -> None:
        self._tree = _build_default_tree()
        self._last_regime: Optional[Regime] = None
        self._pending_regime: Optional[Regime] = None
        self._pending_count: int = 0
        self._n_classifications: int = 0
        self._classification_times_ms: list[float] = []

    def classify(self, features: RuntimeFeatures) -> Regime:
        """
        Classify the current operating regime from runtime statistics.

        If raw R_C / R_B values are available in features, uses the
        analytical threshold tree directly (highest accuracy).
        Otherwise falls back to the CART feature tree.

        Returns the stable regime (unchanged for ≥ HYSTERESIS_COUNT windows).
        """
        t0 = time.perf_counter()

        if features.r_c is not None and features.r_b is not None:
            # Direct analytical path (deterministic, no tree needed)
            from orion.ratios import classify_regime
            raw = classify_regime(features.r_c, features.r_b)
        else:
            feat_dict = {
                "swap_to_comp_ratio": features.swap_to_comp_ratio,
                "cache_hit_rate":     features.cache_hit_rate,
                "dma_utilisation":    features.dma_utilisation,
            }
            raw = self._tree.predict(feat_dict)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._classification_times_ms.append(elapsed_ms)
        self._n_classifications += 1

        return self._apply_hysteresis(raw)

    def _apply_hysteresis(self, raw: Regime) -> Regime:
        """Suppress transient misclassifications via hysteresis guard."""
        if raw == self._last_regime:
            self._pending_count = 0
            self._pending_regime = None
            return self._last_regime

        if raw == self._pending_regime:
            self._pending_count += 1
        else:
            self._pending_regime = raw
            self._pending_count = 1

        if self._pending_count >= self.HYSTERESIS_COUNT:
            self._last_regime = self._pending_regime
            self._pending_count = 0
            self._pending_regime = None

        return self._last_regime if self._last_regime is not None else raw

    @property
    def mean_latency_ms(self) -> float:
        """Mean classification latency (should be < 0.1 ms)."""
        if not self._classification_times_ms:
            return 0.0
        return sum(self._classification_times_ms) / len(self._classification_times_ms)

    def reset(self) -> None:
        self._last_regime = None
        self._pending_regime = None
        self._pending_count = 0
        self._n_classifications = 0
        self._classification_times_ms.clear()
