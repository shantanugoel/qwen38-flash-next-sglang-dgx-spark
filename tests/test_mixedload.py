"""Timing regressions that can conceal a prefill stall."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from mixedload import gaps_in_window


class GapTests(unittest.TestCase):
    def test_boundary_crossing_stall_is_included_in_full(self):
        chunks = [{'t_abs': t} for t in [0, 1, 10, 11]]
        self.assertEqual(gaps_in_window(chunks, 2, 9), [9])

    def test_outside_gaps_excluded(self):
        chunks = [{'t_abs': t} for t in [0, 1, 3, 5, 9, 10]]
        self.assertEqual(gaps_in_window(chunks, 2, 6), [2, 2, 4])

    def test_no_chunks(self):
        self.assertEqual(gaps_in_window([], 2, 6), [])
