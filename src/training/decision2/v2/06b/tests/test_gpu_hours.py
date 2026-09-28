import importlib
import unittest

hours = importlib.import_module("v2.06b.gpu_hours")


class OccupancyTest(unittest.TestCase):
    def test_overlapping_runs_on_one_gpu_count_once(self):
        self.assertEqual(hours.merged_seconds([(0, 10), (5, 20), (30, 40)]), 30)
        self.assertEqual(hours.merged_seconds([]), 0)
        runs = [
            {
                "gpu": 0,
                "start_utc": "2026-09-28T00:00:00Z",
                "end_utc": "2026-09-28T01:00:00Z",
            },
            {
                "gpu": 0,
                "start_utc": "2026-09-28T00:30:00Z",
                "end_utc": "2026-09-28T01:30:00Z",
            },
            {
                "gpu": 1,
                "start_utc": "2026-09-28T00:00:00Z",
                "end_utc": "2026-09-28T00:30:00Z",
            },
        ]
        report = hours.occupancy(runs)
        self.assertEqual(report["per_gpu_hours"], {"0": 1.5, "1": 0.5})
        self.assertEqual(report["total_gpu_hours"], 2.0)
        self.assertEqual(report["summed_run_hours"], 2.5)


if __name__ == "__main__":
    unittest.main()
