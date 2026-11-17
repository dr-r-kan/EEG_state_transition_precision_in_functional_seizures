"""Pure helpers of the FIF-to-EEGLAB/MAT converter (the full conversion is covered by test_manifest)."""
import unittest

import numpy as np

from fsms.manifest import convert_fif_manifest as convert


class ConverterHelperTests(unittest.TestCase):
    def test_front_back_landmarks_define_the_anterior_axis(self):
        labels = ["C18", "A25"]
        ras = np.array([[0.0, 0.093, 0.056], [0.0, -0.086, 0.0]])
        rotated = np.array([[0.093, 0.0, 0.056], [-0.086, 0.0, 0.0]])
        self.assertTrue(convert.heppy_front_back_landmarks_ok(labels, ras))
        self.assertFalse(convert.heppy_front_back_landmarks_ok(labels, rotated))
        self.assertIsNone(convert.heppy_front_back_landmarks_ok(["Cz", "Pz"], ras))      # landmarks absent: no verdict

    def test_mne_coordinates_are_kept_in_ras(self):
        c18 = np.array([0.0, 0.093, 0.056])
        np.testing.assert_allclose(convert.mne_xyz_to_pipeline_xyz(c18), c18)

    def test_channel_labels_are_normalised_for_matching(self):
        self.assertEqual(convert.normalise_channel_label("EEG Fp1-REF"), "fp1ref")
        self.assertEqual(convert.normalise_channel_label("Channel 007"), "7")
        self.assertEqual(convert.normalise_channel_label("  Cz "), "cz")

    def test_usable_position(self):
        self.assertTrue(convert.usable_position([0.0, 0.1, 0.0]))
        self.assertFalse(convert.usable_position(None))
        self.assertFalse(convert.usable_position([0.0, 0.0, 0.0]))
        self.assertFalse(convert.usable_position([np.nan, 0.0, 0.1]))

    def test_file_column_detection_ignores_case_and_punctuation(self):
        self.assertEqual(convert.first_file_column(["id", "File Path", "x"]), "File Path")
        self.assertEqual(convert.first_file_column(["id", "x"]), "")


if __name__ == "__main__":
    unittest.main()
