"""Multiple random starts of the global microstate fit: configuration, fingerprint and release bundle."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fsms import pipeline  # noqa: E402
from fsms import release as R  # noqa: E402
from fsms.config import DEFAULT_N_STARTS, Config  # noqa: E402


def make_config(root: Path, microstates: dict) -> Config:
    raw = {"schema": "fsms-config/1", "paths": {"data_root": str(root / "data"), "run_root": str(root / "run")},
           "microstates": microstates}
    config = Config(raw, root / "study.json")
    config.vg_dir.mkdir(parents=True)
    config.manifest_mat.write_text("file_path\nA.mat\n", encoding="utf-8")
    return config


class StartsConfigTests(unittest.TestCase):
    def test_default_is_multi_start(self):
        self.assertGreater(DEFAULT_N_STARTS, 1)

    def test_number_of_starts_changes_the_hierarchy_folder(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            few = pipeline.hierarchy_dir(make_config(root / "a", {"n_starts": 5}))
            many = pipeline.hierarchy_dir(make_config(root / "b", {"n_starts": 20}))
            self.assertNotEqual(few.name, many.name)


class StartsReleaseTests(unittest.TestCase):
    def test_start_table_is_published_with_the_k_selection(self):
        with tempfile.TemporaryDirectory() as t:
            source, target = Path(t) / "global", Path(t) / "release"
            source.mkdir()
            for name in ("k_selection.csv", "free_energy_starts.csv", "state_labels.csv", "global_centres.csv"):
                (source / name).write_text("K\n4\n", encoding="utf-8")
            R.copy_microstate_solution(source, target)
            self.assertTrue((target / "free_energy_starts.csv").exists())


if __name__ == "__main__":
    unittest.main()
