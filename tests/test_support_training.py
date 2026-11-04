import unittest
from pathlib import Path

from models.train_support_detector import original_group, partition_groups


class TrainingSplitTests(unittest.TestCase):
    def test_rotated_photos_share_the_same_source_group(self):
        base = "mobilenet_experiment_0340dc35-5215-48ab-8db7-06af99fcb358___leaf"
        self.assertEqual(
            original_group(Path(base + ".JPG")),
            original_group(Path(base + "_270deg.JPG")),
        )

    def test_rice_augmentations_share_the_original_leaf_group(self):
        self.assertEqual(
            original_group(Path("rice_4class_01600_BLAST4_114.jpg")),
            original_group(Path("rice_4class_01605_BLAST4_100.jpg")),
        )
        self.assertNotEqual(
            original_group(Path("rice_4class_01600_BLAST4_114.jpg")),
            original_group(Path("rice_4class_01604_BLAST1_051.JPG")),
        )

    def test_small_group_splits_are_disjoint_and_reproducible(self):
        groups = {"a", "b", "c", "d", "e", "f"}
        split = partition_groups(groups, 42)
        self.assertEqual(split, partition_groups(groups, 42))
        self.assertTrue(all(split.values()))
        flattened = [group for values in split.values() for group in values]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), groups)
        with self.assertRaises(ValueError):
            partition_groups({"one", "two"}, 42)


if __name__ == "__main__":
    unittest.main()
