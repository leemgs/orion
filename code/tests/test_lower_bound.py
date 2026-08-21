import math
import sys
import unittest
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from orion.lower_bound import compute_lower_bound  # noqa: E402
from orion.ratios import OperatingPoint  # noqa: E402


class LowerBoundTest(unittest.TestCase):
    def test_bound_uses_maximum_to_allow_ideal_overlap(self):
        op = OperatingPoint(r_c=0.5, r_b=0.4, w_gb=2.0, d_gb=1.0,
                            b_slow_gbs=2.0, t_comp_s=0.25)
        result = compute_lower_bound(op, rho=1e-10, t_measured_s=1.0)

        self.assertAlmostEqual(result.t_comp_min_s, 0.25)
        self.assertAlmostEqual(result.t_mem_min_s, 0.1)
        self.assertAlmostEqual(result.t_swap_min_s, 0.5)
        self.assertAlmostEqual(result.t_lower_bound_s, 0.5)
        self.assertAlmostEqual(result.residual_fraction, 0.5)

    def test_missing_measurement_has_nan_achievability(self):
        result = compute_lower_bound(OperatingPoint(r_c=1.0, r_b=1.0), rho=0.0)
        self.assertTrue(math.isnan(result.achievability))


if __name__ == "__main__":
    unittest.main()
