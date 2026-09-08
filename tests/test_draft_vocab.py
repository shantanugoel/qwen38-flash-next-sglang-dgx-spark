import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    path = ROOT / 'scripts' / 'build_draft_vocab.py'
    spec = importlib.util.spec_from_file_location('build_draft_vocab', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DraftVocabTests(unittest.TestCase):
    def test_rank_keeps_specials_first_and_unique(self):
        from collections import Counter
        mod = load_builder()
        counts = Counter({7: 9, 3: 5, 1: 4, 99: 1})
        ids, report = mod.rank_ids([3, 8, 3], counts, vocab_size=20, size=8, fill=True)
        self.assertEqual(ids[:2], [3, 8])
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 8)
        self.assertGreater(report['n_fill'], 0)
        self.assertTrue(all(0 <= i < 20 for i in ids))

    def test_rank_keeps_high_specials_inside_full_vocab(self):
        from collections import Counter
        mod = load_builder()
        counts = Counter({7: 9})
        ids, report = mod.rank_ids([248058, 3], counts, vocab_size=248320, size=8, fill=True)
        self.assertEqual(ids[0], 248058)
        self.assertIn(3, ids)
        self.assertTrue(all(i < 248320 for i in ids))

    def test_rank_without_fill_stops_at_corpus(self):
        from collections import Counter
        mod = load_builder()
        counts = Counter({1: 2, 2: 1})
        ids, report = mod.rank_ids([0], counts, vocab_size=50, size=64, fill=False)
        self.assertEqual(ids, [0, 1, 2])
        self.assertEqual(report['n_fill'], 0)
        self.assertEqual(report['n_final'], 3)

    def test_serve_mounts_token_map_after_mounts_exist(self):
        text = (ROOT / 'scripts' / 'serve.sh').read_text()
        self.assertIn('SPECULATIVE_TOKEN_MAP', text)
        self.assertIn('/speculative-token-map.pt', text)
        self.assertLess(text.index('MOUNTS=('), text.index('DEFAULT_TOKEN_MAP='))
        self.assertIn('bench/draft_vocab/hot_tokens_64k.pt', text)
        self.assertTrue((ROOT / 'bench' / 'draft_vocab' / 'hot_tokens_64k.pt').is_file())

    def test_safe_flags_include_token_map(self):
        import sys
        sys.path.insert(0, str(ROOT / 'scripts'))
        from experiment import SAFE_FLAGS
        self.assertIn('speculative_token_map', SAFE_FLAGS)


if __name__ == '__main__':
    unittest.main()
