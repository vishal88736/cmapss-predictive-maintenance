"""Input cadence and target-specific website reports, using untrained fixtures."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from grace_rul.config import COLUMNS, DEFAULT_CONFIG, FORMAT_VERSION, quick_config
from grace_rul.data import OperatingRegimeScaler
from grace_rul.model import GraceRUL
from website.backend.inference import InputError, parse_csv, regression_metrics
from website.backend.service import InferenceService


def history(cycles=(100, 101), unit=1, truth=170.):
    frame = pd.DataFrame(np.zeros((len(cycles), len(COLUMNS))), columns=COLUMNS)
    frame["unit"], frame["cycle"] = unit, cycles
    frame["true_rul"] = truth
    return frame


def conformal_contract(radius):
    protocols = {}
    for protocol, target in [("capped_125", "target"), ("official_uncapped", "raw_target")]:
        unbounded = radius is None
        count, rank = (4, 5) if unbounded else (20, 19)
        protocols[protocol] = {"method": "split_conformal_absolute", "policy": "engine_prefix_max",
            "target_convention": protocol, "target_column": target, "alpha": .1, "engines": count,
            "n": count, "rank": rank, "radius": radius, "unbounded": unbounded,
            "lower_offset": -radius if radius is not None else None, "upper_offset": radius,
            "lower_bound": 0., "scope": f"fixture {protocol} calibration"}
    return {**protocols["capped_125"], "primary_protocol": "capped_125", "protocols": protocols}


def fixture_service(calibration):
    config = quick_config({**DEFAULT_CONFIG, "device": "cpu", "train_repair": False})
    torch.manual_seed(12)
    model = GraceRUL(config).eval()
    scaler = OperatingRegimeScaler({"setting_mean": [0.] * 3, "setting_std": [1.] * 3,
        "centers": [[0.] * 3], "sensor_mean": [[0.] * 21], "sensor_std": [[1.] * 21],
        "sensor_transform": config["sensor_transform"], "sensor_transform_scale": config["sensor_transform_scale"]})
    service = InferenceService()
    service.config = config
    service.bundle = {"format_version": FORMAT_VERSION}
    service.path = Path("/tmp/website-contract-fixture/grace_rul_ALL.pt")
    service.loaded = {"FD001": {"model": model, "scaler": scaler,
        "entry": {"calibration": calibration, "thresholds": [1.] * 21}}}
    service.refresh = lambda: None
    return service


class WebsiteTargetContract(unittest.TestCase):
    def test_cycle_gaps_rejected_but_consecutive_shifted_histories_accepted(self):
        with self.assertRaisesRegex(InputError, "consecutive cycles"):
            parse_csv(history((1, 3)).to_csv(index=False).encode(), "FD001")
        parsed = parse_csv(history().to_csv(index=False).encode(), "FD001")
        self.assertEqual(parsed.cycle.tolist(), [100, 101])
        frame = pd.concat([history((3, 4)), history((10, 11), unit=2)])
        frame.loc[frame.index == 0, "sensor_1"] = np.nan
        parsed = parse_csv(frame.to_csv(index=False).encode(), "FD001")
        self.assertEqual(len(parsed), 4)
        self.assertTrue(parsed.sensor_1.isna().any())

    def test_legacy_reports_keep_raw_truth_and_capped_metrics_distinct(self):
        service = fixture_service({"lower_offset": -10., "upper_offset": 10., "alpha": .1,
            "scope": "legacy capped calibration"})
        frame = parse_csv(history().to_csv(index=False).encode(), "FD001")
        report = service.predict(frame, "fixture.csv")
        engine = report["engines"][0]
        self.assertEqual(engine["true_rul"], 170.)
        self.assertEqual(engine["true_rul_capped"], 125.)
        self.assertEqual(engine["target_convention"], "capped_125")
        self.assertEqual(engine["train_rul_cap"], 125.)
        self.assertEqual(engine["failure_cycle_proxy"], 101 + engine["predicted_rul"])
        self.assertIn("horizon-limited proxy", engine["failure_cycle_proxy_scope"])
        self.assertEqual(report["metrics_by_protocol"]["capped"]["protocol"], "capped_125")
        self.assertAlmostEqual(report["metrics_by_protocol"]["capped"]["RMSE"], abs(engine["predicted_rul"] - 125.))
        self.assertAlmostEqual(report["metrics_by_protocol"]["official_uncapped"]["RMSE"], abs(engine["predicted_rul"] - 170.))
        self.assertEqual(report["metrics"]["protocol"], "official_uncapped")
        self.assertIsNone(report["metrics"]["Interval_Coverage"])
        self.assertEqual(report["metrics"]["Interval_N"], 0)
        self.assertFalse(engine["intervals"]["official_uncapped"]["available"])
        self.assertEqual(service.status()["interval_method"], "empirical_residual_quantiles")
        json.dumps(report, allow_nan=False)

    def test_unbounded_intervals_are_explicit_json_nulls_with_valid_coverage(self):
        service = fixture_service(conformal_contract(None))
        frame = parse_csv(history().to_csv(index=False).encode(), "FD001")
        report = service.predict(frame, "fixture.csv")
        engine = report["engines"][0]
        self.assertEqual(engine["interval_lower"], 0.)
        self.assertIsNone(engine["interval_upper"])
        self.assertIsNone(engine["interval_width"])
        self.assertTrue(engine["interval_unbounded"])
        self.assertIsNone(report["summary"]["mean_interval_width"])
        self.assertEqual(report["summary"]["unbounded_intervals"], 1)
        self.assertTrue(all(row["interval_unbounded"] and row["interval_upper"] is None for row in engine["trajectory"]))
        for protocol in ["capped", "official_uncapped"]:
            self.assertEqual(report["metrics_by_protocol"][protocol]["Interval_Coverage"], 1.)
            self.assertEqual(regression_metrics(report["engines"], protocol)["Interval_Coverage"], 1.)
        self.assertEqual(service.status()["interval_method"], "split_conformal_absolute")
        json.dumps(report, allow_nan=False)

    def test_capped_and_raw_coverage_use_their_own_finite_calibrations(self):
        calibration = conformal_contract(1.)
        calibration["protocols"]["official_uncapped"].update(radius=1000., lower_offset=-1000., upper_offset=1000.)
        service = fixture_service(calibration)
        frame = parse_csv(history().to_csv(index=False).encode(), "FD001")
        report = service.predict(frame, "fixture.csv")
        engine = report["engines"][0]
        capped, raw = engine["intervals"]["capped"], engine["intervals"]["official_uncapped"]
        self.assertAlmostEqual(raw["interval_upper"] - capped["interval_upper"], 999.)
        self.assertEqual(report["metrics_by_protocol"]["official_uncapped"]["Interval_Coverage"], 1.)
        expected = float(capped["interval_lower"] <= 125. <= capped["interval_upper"])
        self.assertEqual(report["metrics_by_protocol"]["capped"]["Interval_Coverage"], expected)
        self.assertFalse(engine["interval_unbounded"])
        self.assertEqual(report["summary"]["unbounded_intervals"], 0)
        json.dumps(report, allow_nan=False)

    def test_mixed_finite_and_unbounded_engine_report_has_no_finite_mean_width(self):
        service = fixture_service(conformal_contract(None))
        service.loaded["FD002"] = fixture_service(conformal_contract(10.)).loaded["FD001"]
        first, second = history(), history()
        first["subset"], second["subset"] = "FD001", "FD002"
        frame = parse_csv(pd.concat([first, second]).to_csv(index=False).encode(), "FD001")
        report = service.predict(frame, "mixed-fixture.csv")
        self.assertEqual(len(report["engines"]), 2)
        self.assertEqual(report["summary"]["unbounded_intervals"], 1)
        self.assertIsNone(report["summary"]["mean_interval_width"])
        bounded = next(engine for engine in report["engines"] if engine["subset"] == "FD002")
        self.assertEqual(bounded["interval_width"], 20.)
        unbounded = next(engine for engine in report["engines"] if engine["subset"] == "FD001")
        self.assertIsNone(unbounded["interval_width"])
        self.assertIsNone(unbounded["interval_upper"])
        self.assertEqual(report["metrics_by_subset_protocol"]["FD001"]["official_uncapped"]["Interval_Coverage"], 1.)
        json.dumps(report, allow_nan=False)

    def test_discovery_accepts_supported_legacy_and_current_full_manifests(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"GRACE_MODEL_PATH": "", "GRACE_ARTIFACTS_DIR": directory}):
            service = InferenceService()
            for version in [5, 6, 7]:
                folder = Path(directory) / f"run_{version}"
                folder.mkdir()
                model = folder / "grace_rul_ALL.pt"
                model.touch()
                os.utime(model, (version, version))
                (folder / "run_manifest.json").write_text(json.dumps({"format_version": version, "quick_run": False}))
                self.assertEqual(service.candidate(), model)
            quick = Path(directory) / "quick"
            quick.mkdir()
            (quick / "grace_rul_ALL.pt").touch()
            (quick / "run_manifest.json").write_text(json.dumps({"format_version": 7, "quick_run": True}))
            self.assertEqual(service.candidate().parent.name, "run_7")


if __name__ == "__main__":
    unittest.main()
