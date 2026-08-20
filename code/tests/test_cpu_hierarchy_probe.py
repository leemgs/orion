import sys
import unittest
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from experiments.cpu_hierarchy_probe import (  # noqa: E402
    chase_latency_ns,
    detect_llc_bytes,
    load_chase_kernel,
    make_chase,
)


class CpuHierarchyProbeTest(unittest.TestCase):
    def test_compiled_kernel_traverses_seeded_cycle(self):
        rng = np.random.default_rng(7)
        cycle = make_chase(4096, rng)
        latency = chase_latency_ns(load_chase_kernel(), cycle, 100_000)
        self.assertGreater(latency, 0.0)

    @unittest.skipUnless(Path("/sys/devices/system/cpu/cpu0/cache").exists(),
                         "Linux cache sysfs is unavailable")
    def test_llc_detection_returns_positive_size(self):
        self.assertGreater(detect_llc_bytes(), 0)


if __name__ == "__main__":
    unittest.main()
