"""Behavior checks using real source data. No trained checkpoint is manufactured."""
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from .app import app, uploads, jobs
from .project import PROJECT_ROOT, RAW_COLUMNS, TRAINING_ROOT
from .validation import InputError, validate_test, validate_truth


class UploadWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.test_bytes = (PROJECT_ROOT / "data" / "test_FD001.txt").read_bytes()
        cls.truth_bytes = (PROJECT_ROOT / "data" / "RUL_FD001.txt").read_bytes()
        cls.expected = np.loadtxt(PROJECT_ROOT / "data" / "test_FD001.txt")

    def tearDown(self):
        uploads.clear()
        jobs.clear()

    def post(self, content=None, name="test_FD001.txt", truth=None):
        files = {"test_file": (name, content if content is not None else self.test_bytes, "text/plain")}
        if truth is not None:
            files["rul_file"] = ("RUL_FD001.txt", truth, "text/plain")
        return self.client.post("/api/uploads", data={"subset": "FD001"}, files=files)

    def test_real_test_upload_without_truth_and_lossless_telemetry(self):
        response = self.post()
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["groundTruthAvailable"])
        self.assertEqual(data["columns"], RAW_COLUMNS)
        self.assertEqual(data["rowCount"], len(self.expected))
        self.assertEqual(data["engineIds"], np.unique(self.expected[:, 0]).astype(int).tolist())
        self.assertEqual(data["stages"]["validation"], "completed")
        self.assertEqual(data["stages"]["preprocessing"], "pending")
        unit = data["engineIds"][0]
        telemetry = self.client.get(f"/api/uploads/{data['uploadId']}/engines/{unit}").json()
        np.testing.assert_array_equal(telemetry["readings"], self.expected[self.expected[:, 0] == unit, 1:])

    def test_truth_is_optional_but_checked_when_attached(self):
        response = self.post(truth=self.truth_bytes)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["groundTruthAvailable"])
        self.assertEqual(self.post(truth=b"1\n").status_code, 422)

    def test_missing_sensor_and_csv_are_rejected_without_silent_repair(self):
        first = self.test_bytes.decode().splitlines()[0].split()
        response = self.post(content=" ".join(first[:-1]).encode())
        self.assertEqual(response.status_code, 422)
        self.assertIn("sensor_21", response.json()["detail"]["message"])
        self.assertEqual(self.post(name="input.csv").status_code, 422)
        self.assertEqual(self.post(content=b"").status_code, 422)

    def test_duplicate_and_nonfinite_readings_are_rejected(self):
        first = self.test_bytes.decode().splitlines()[0]
        with self.assertRaisesRegex(InputError, "duplicate"):
            validate_test("test.txt", f"{first}\n{first}".encode())
        fields = first.split()
        fields[-1] = "nan"
        with self.assertRaisesRegex(InputError, "non-finite"):
            validate_test("test.txt", " ".join(fields).encode())

    def test_missing_model_never_generates_outputs_or_completed_inference(self):
        data = self.post().json()
        with patch("backend.app.artifact_status", return_value={"ready": False, "reason": "Trained model could not be loaded."}):
            response = self.client.post("/api/predictions", json={"uploadId": data["uploadId"]})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"]["code"], "model_unavailable")
        self.assertFalse(jobs)
        self.assertNotIn("predictions", data)

    def test_discard_removes_uploaded_sensor_data(self):
        data = self.post().json()
        self.client.delete(f"/api/uploads/{data['uploadId']}")
        self.assertEqual(self.client.get(f"/api/uploads/{data['uploadId']}/engines/1").status_code, 410)

    def test_metrics_are_exact_saved_output_with_source_and_no_invented_mae(self):
        notebook = json.loads((TRAINING_ROOT / "cmapss_ag.ipynb").read_text())
        output = "\n".join("".join(item.get("text", [])) for cell in notebook["cells"] for item in cell.get("outputs", []) if item.get("name") == "stdout")
        recorded = {m[0]: (float(m[2]), float(m[3])) for m in re.findall(r"^(FD00[1-4])\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*$", output, re.MULTILINE)}
        response = self.client.get("/api/metrics").json()
        self.assertEqual(len(response["records"]), len(recorded))
        for record in response["records"]:
            self.assertEqual((record["metrics"]["RMSE"], record["metrics"]["CMAPSS_Score"]), recorded[record["subset"]])
            self.assertNotIn("MAE", record["metrics"])
            self.assertNotIn("R2", record["metrics"])
            self.assertEqual(record["predictions"], [])
            self.assertTrue(record["sourceSHA256"])
            self.assertIn("saved notebook", record["note"])


if __name__ == "__main__":
    unittest.main()
