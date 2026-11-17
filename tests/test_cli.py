"""Every command-line driver imports cleanly and documents its arguments (no data needed)."""
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("fsms.heppy.run_heppy_stage", "fsms.clips.snip_clips", "fsms.manifest.build_vg_manifest",
           "fsms.fixture.make_tueg_fixture", "fsms.autonomic")


class CommandLineTests(unittest.TestCase):
    def test_help_exits_cleanly_for_every_driver(self):
        for module in MODULES:
            with self.subTest(module=module):
                done = subprocess.run([sys.executable, "-B", "-m", module, "--help"], cwd=ROOT, capture_output=True,
                                      text=True, env=dict(os.environ, PYTHONPATH=str(ROOT)))
                self.assertEqual(done.returncode, 0, done.stderr[-1500:])
                self.assertIn("usage:", done.stdout)
                self.assertNotIn("--mode", done.stdout)


if __name__ == "__main__":
    unittest.main()
