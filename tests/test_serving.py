"""Server launch contracts; no GPU, endpoint, or model invocation."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moha.serving import launch_plan, main


class ServingTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).parents[1] / "config.example.json").read_text())
        self.config["observer"]["base_url"] = "http://omni:8092/v1,http://omni:8097/v1"

    def plan(self, **kwargs):
        return launch_plan(self.config, vllm_bin="/test/vllm-omni", model_path="/test/model",
                           gpu="0,1", **kwargs)

    def test_host_limit_drives_actual_loader_flag_for_new_and_old_configs(self):
        for frames in (32, 64, 128):
            self.config["budget"]["f_view"] = frames
            before = copy.deepcopy(self.config)
            plan = self.plan()
            command = plan["command"]
            loader = json.loads(command[command.index("--media-io-kwargs") + 1])
            self.assertEqual(loader, {"video": {"num_frames": frames}})
            self.assertEqual(plan["host_max_frames"], frames)
            self.assertEqual(command[command.index("--max-model-len") + 1], "32768")
            self.assertEqual(command[command.index("--max-num-seqs") + 1], "1")
            self.assertEqual(plan["max_num_seqs"], 1)
            self.assertNotIn("--mm-processor-kwargs", command)
            self.assertEqual(self.config, before)

    def test_endpoint_pool_selects_corresponding_listen_port(self):
        for index, port in enumerate(("8092", "8097")):
            command = self.plan(endpoint_index=index)["command"]
            self.assertEqual(command[command.index("--port") + 1], port)
        with self.assertRaisesRegex(ValueError, "endpoint_index"):
            self.plan(endpoint_index=2)

    def test_unbounded_or_invalid_frame_limits_cannot_start_bounded_server(self):
        for value in (None, True, 0, -1, 32.0, "128"):
            self.config["budget"]["f_view"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "budget.f_view"):
                self.plan()

    def test_omni_scheduler_is_serial_and_rejects_invalid_limits(self):
        self.assertEqual(self.plan(max_num_seqs=1)["max_num_seqs"], 1)
        for value in (None, True, 0, -1, 1.0, "1"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "max_num_seqs"):
                self.plan(max_num_seqs=value)

    def test_dry_run_never_starts_server_or_writes_record_or_discloses_credentials(self):
        self.config["observer"]["key"] = {"file": "/private/secret-sentinel", "field": "secret-field"}
        with tempfile.TemporaryDirectory() as directory:
            cfg = Path(directory) / "config.json"
            cfg.write_text(json.dumps(self.config))
            record = Path(directory) / "launch.json"
            out = io.StringIO()
            with patch("moha.serving.os.execvpe") as execute, patch("sys.stdout", out):
                self.assertEqual(main(["--config", str(cfg), "--gpu", "0,1", "--vllm-bin", "/test/vllm",
                    "--model-path", "/test/model", "--launch-record", str(record), "--dry-run"]), 0)
            execute.assert_not_called()
            self.assertFalse(record.exists())
            self.assertNotIn("secret-sentinel", out.getvalue())
            self.assertNotIn("secret-field", out.getvalue())
            self.assertFalse(json.loads(out.getvalue())["server_verified"])

    def test_real_launch_records_the_same_argv_and_preserves_existing_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, executable, model, record = [root / p for p in ("config.json", "vllm", "model", "launch.json")]
            cfg.write_text(json.dumps(self.config))
            executable.write_text("#!/bin/sh\nexit 99\n")
            executable.chmod(0o700)
            model.mkdir()
            argv = ["--config", str(cfg), "--gpu", "0,1", "--vllm-bin", str(executable),
                    "--model-path", str(model), "--launch-record", str(record)]
            with patch("moha.serving.os.execvpe") as execute:
                main(argv)
                audit = json.loads(record.read_text())
                self.assertEqual(execute.call_args.args[1], audit["command"])
                self.assertEqual(execute.call_args.args[2]["CUDA_VISIBLE_DEVICES"], "0,1")
            before = record.read_bytes()
            with patch("moha.serving.os.execvpe") as execute, self.assertRaises(FileExistsError):
                main(argv)
            execute.assert_not_called()
            self.assertEqual(record.read_bytes(), before)

    def test_credentials_in_endpoint_and_invalid_gpu_lists_are_rejected(self):
        self.config["observer"]["base_url"] = "http://user:password@host:8092/v1"
        with self.assertRaisesRegex(ValueError, "no credentials"):
            self.plan()
        for gpu in ("", "0,0", "0; touch /tmp/no", "-1"):
            with self.assertRaisesRegex(ValueError, "GPU indices"):
                launch_plan(self.config, vllm_bin="unused", model_path="unused", gpu=gpu)
