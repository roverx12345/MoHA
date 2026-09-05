import json
import tempfile
import unittest
from pathlib import Path
from moha.demo import run_demo
from moha.models import digest
from moha.profiles import calibrated_profile, export_profile


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / "run"
        run_demo(self.run)
        self.identity = {"config": {
            "models": {"planner": {"spec": {"model": "planner/A", "base_url": "https://unit.invalid/v1"},
                                   "key": {"file": "CREDENTIAL_PATH_MUST_NOT_EXPORT", "field": "SECRET"}}},
            "observer": {"model": "observer/B", "backend": "qwen3omni", "key": {"env": "SECRET_ENV"}},
            "budget": {"f_episode": 256},
            "asr": {"spec": {"model": "whisper", "base_url": "http://127.0.0.1:8093/v1"},
                    "key": {"local": True}, "language": None}},
            "source": {"commit": "unit", "source_hash": "unit", "dirty": False},
            "runtime": {"commit": "dependency", "source_hash": "unit", "dirty": False},
            "input_hashes": {"calibration": "a", "validation": "b"}}
        self.write("manifest.json", {"schema": "moha_run_v1", "identity": self.identity,
                                      "identity_hash": digest(self.identity)})
        frozen = self.read("frozen_harness.json")
        frozen["run_identity"] = digest(self.identity)
        self.write("frozen_harness.json", frozen)

    def read(self, name):
        return json.loads((self.run / name).read_text())

    def write(self, name, value):
        (self.run / name).write_text(json.dumps(value))

    def test_export_contains_frozen_settings_and_provenance_without_credentials(self):
        profile = calibrated_profile(self.run)
        self.assertTrue(profile["harness"]["overview"])
        self.assertEqual(profile["pair"], {"planner": "planner/A", "observer": "observer/B"})
        self.assertEqual(profile["provenance"]["run_identity"], digest(self.identity))
        self.assertEqual(profile["stack"]["asr"]["language"], None)
        self.assertNotIn("CREDENTIAL_PATH", json.dumps(profile))
        self.assertNotIn("SECRET", json.dumps(profile))

    def test_unfinished_calibration_and_modified_identity_are_rejected(self):
        checkpoint = self.read("checkpoint.json")
        checkpoint["status"] = "running"
        self.write("checkpoint.json", checkpoint)
        with self.assertRaisesRegex(ValueError, "completed"):
            calibrated_profile(self.run)
        self.write("checkpoint.json", self.read("result.json"))
        frozen = self.read("frozen_harness.json")
        frozen["run_identity"] = "different"
        self.write("frozen_harness.json", frozen)
        with self.assertRaisesRegex(ValueError, "identity"):
            calibrated_profile(self.run)
        (self.run / "frozen_harness.json").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            calibrated_profile(self.run)

    def test_frozen_harness_must_follow_accepted_interventions(self):
        frozen = self.read("frozen_harness.json")
        frozen["harness"]["specialists"] = ["asr"]
        self.write("frozen_harness.json", frozen)
        with self.assertRaisesRegex(ValueError, "frozen harness"):
            calibrated_profile(self.run)

    def test_one_current_file_per_model_pair_and_source_run_is_immutable(self):
        registry = self.root / "calibrated"
        result = export_profile(self.run, registry)
        self.assertEqual(export_profile(self.run, registry), result)
        self.assertEqual(len(list(registry.glob("*.json"))), 1)
        self.assertEqual(Path(result["profile"]).parent, registry.resolve())
        with self.assertRaisesRegex(ValueError, "outside"):
            export_profile(self.run, self.run / "calibrated")


if __name__ == "__main__":
    unittest.main()
