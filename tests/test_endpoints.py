"""Provider routing with the pinned clients, without network requests."""
import copy
import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch
from moha.bridge import build, lane_clients, prepare
from moha.demo import sample
from moha.store import RunStore

try:
    from video_os.core.budget import BudgetContract
except ImportError:
    BudgetContract = None


@unittest.skipIf(BudgetContract is None, "pinned Video OS runtime is required")
class EndpointTests(unittest.TestCase):
    def budget(self):
        return BudgetContract(**json.loads((Path(__file__).parents[1] / "config.example.json").read_text())["budget"])

    def models(self, urls):
        return {name: {"spec": {"model": name, "base_url": urls if name == "planner" else "http://judge/v1"},
                       "key": {"local": True}} for name in ("planner", "judge", "extractor")}

    def test_two_planners_share_observer_and_keep_separate_clients(self):
        models = self.models("http://p0/v1/, http://p1/v1")
        original = copy.deepcopy(models)
        planners, observers, lanes = lane_clients(models, "http://omni/v1", self.budget())
        self.assertEqual(planners, ("http://p0/v1", "http://p1/v1"))
        self.assertEqual(observers, ("http://omni/v1",) * 2)
        self.assertEqual([lane["planner"].spec.base_url for lane in lanes], list(planners))
        for name in ("planner", "extractor"):
            self.assertIsNot(lanes[0][name], lanes[1][name])
        self.assertTrue(all("judge" not in lane for lane in lanes))
        self.assertEqual(models, original)

    def test_shared_planner_retains_old_observer_lane_mapping(self):
        planners, observers, lanes = lane_clients(self.models("http://planner/v1"),
            "http://o0/v1,http://o1/v1", self.budget())
        self.assertEqual(planners, ("http://planner/v1",) * 2)
        self.assertEqual(observers, ("http://o0/v1", "http://o1/v1"))
        self.assertIsNot(lanes[0]["planner"], lanes[1]["planner"])

    def test_matching_pools_pair_in_order_and_singletons_stay_single(self):
        for count in (1, 2):
            planners = tuple(f"http://p{i}/v1" for i in range(count))
            observers = tuple(f"http://o{i}/v1" for i in range(count))
            actual_p, actual_o, lanes = lane_clients(self.models(",".join(planners)),
                ",".join(observers), self.budget())
            self.assertEqual((actual_p, actual_o), (planners, observers))
            self.assertEqual(len(lanes), count)

    def test_ambiguous_pools_fail_before_creating_clients(self):
        with patch("moha.bridge.text_client") as create, self.assertRaisesRegex(ValueError, "counts must match"):
            lane_clients(self.models("http://p0/v1,http://p1/v1"),
                "http://o0/v1,http://o1/v1,http://o2/v1", self.budget())
        create.assert_not_called()

    def test_prepare_and_build_isolate_four_judges_and_original_probe_clients(self):
        cfg = json.loads((Path(__file__).parents[1] / "config.example.json").read_text())
        cfg["models"] = self.models("http://p0/v1,http://p1/v1")
        cfg["specialists"] = []
        cfg.pop("image")
        cfg.pop("asr")
        cfg.pop("diagnosis_workers")  # Future configs default to four workers.
        cfg["observer"]["key"] = {"local": True}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(cfg))
            splits = [([sample(name)], {}, {"samples": [{"media_sha256": name}]}) for name in ("cal", "val")]
            with patch("moha.bridge.activate_runtime", return_value=Path("/runtime")), \
                 patch("moha.bridge.load_split", side_effect=splits), \
                 patch("moha.bridge.source_identity", return_value={}), patch("moha.bridge._snapshot", return_value={}):
                prepared = prepare(path, Path(tmp))
            adapters = prepared["clients"]["judges"] + prepared["clients"]["probe_judges"]
            self.assertEqual(len(prepared["clients"]["judges"]), 4)
            self.assertEqual(len(prepared["clients"]["probe_judges"]), 2)
            self.assertEqual(len({id(c) for c in adapters}), 6)
            with RunStore(Path(tmp) / "run", {}) as store, patch("moha.observer.ObserverService") as service:
                service.side_effect = [unittest.mock.Mock(), unittest.mock.Mock()]
                cal = build(prepared, store)
                self.assertEqual([j.client.adapter for j in cal.judges], prepared["clients"]["judges"])
                self.assertEqual([r.role.client.adapter for r in cal.resolvers], prepared["clients"]["probe_judges"])
                self.assertEqual([r.runner.service for r in cal.resolvers], [r.service for r in cal.runners])


class DiagnosisConfigTests(unittest.TestCase):
    def test_invalid_worker_count_fails_before_runtime_or_client_setup(self):
        cfg = json.loads((Path(__file__).parents[1] / "config.example.json").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            for value in (0, -1, True, 2.5, "4", None):
                cfg["diagnosis_workers"] = value
                path.write_text(json.dumps(cfg))
                with patch("moha.bridge.activate_runtime") as activate, self.assertRaisesRegex(ValueError, "positive integer"):
                    prepare(path, Path(tmp))
                activate.assert_not_called()
