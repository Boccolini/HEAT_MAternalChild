from __future__ import annotations

import unittest
from pathlib import Path

from heat_mnch.preflight import build_inventory, run_preflight


class PreflightTest(unittest.TestCase):
    def test_preflight_writes_status_report(self) -> None:
        root = Path(__file__).resolve().parents[1]
        payload = run_preflight(root)
        self.assertIn("critical_probes", payload)
        self.assertTrue((root / "artifacts" / "status" / "preflight.json").exists())

    def test_inventory_exposes_expected_layers(self) -> None:
        root = Path(__file__).resolve().parents[1]
        payload = build_inventory(root)
        self.assertEqual(set(payload["layers"].keys()), {"raw", "bronze", "silver", "gold"})


if __name__ == "__main__":
    unittest.main()
