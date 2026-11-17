"""The checkpoint and statistics stages cannot be reached without the preregistration serial."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_pipeline  # noqa: E402
from fsms import pipeline  # noqa: E402
from fsms.config import Config, ConfigError  # noqa: E402


def make_config(root: Path) -> Config:
    raw = {"schema": "fsms-config/1", "name": "gate-test", "paths": {"data_root": str(root / "data"), "run_root": str(root / "run")}}
    path = root / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return Config.load(path)


class RunGateTests(unittest.TestCase):
    def test_stage_order_ends_with_checkpoint_then_stats(self):
        self.assertEqual(pipeline.STAGE_ORDER[-2:], ("checkpoint", "stats"))

    def test_checkpoint_and_stats_need_the_serial(self):
        with tempfile.TemporaryDirectory() as t:
            config = make_config(Path(t))
            for stage in ("checkpoint", "stats"):
                with self.assertRaises(pipeline.PipelineError):
                    pipeline.run(config, through=stage, prereg_id=None)
            self.assertFalse((config.run_root / "state").exists())       # refused before any stage ran

    def test_cli_reports_missing_serial(self):
        with tempfile.TemporaryDirectory() as t:
            make_config(Path(t))
            code = run_pipeline.main(["run", "--config", str(Path(t) / "config.json"), "--through", "stats"])
            self.assertEqual(code, 2)

    def test_config_schema_is_checked(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "bad.json"
            path.write_text(json.dumps({"schema": "other", "paths": {}}), encoding="utf-8")
            with self.assertRaises(ConfigError):
                Config.load(path)

    def test_analysis_constants_can_be_overridden_but_have_defaults(self):
        with tempfile.TemporaryDirectory() as t:
            config = make_config(Path(t))
            self.assertEqual(config.analysis()["min_baseline_windows"], 5)
            config.raw["analysis"] = {"min_baseline_windows": 7}
            self.assertEqual(config.analysis()["min_baseline_windows"], 7)
            self.assertEqual(config.analysis()["min_window_transitions"], 30)


if __name__ == "__main__":
    unittest.main()
