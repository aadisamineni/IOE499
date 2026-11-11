"""Offline checks for the dashboard's data boundaries and model integration."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('dashboard', Path(__file__).with_name('server.py'))
app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(app)


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        fields = ['article_id', 'published_at_utc', 'headline', 'summary', 'source', 'url']
        for prefix in ['processed', 'alpaca/processed']:
            folder = self.root / prefix
            folder.mkdir(parents=True)
            for symbol in app.SYMBOLS:
                with (folder / f'{symbol}_news.csv').open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    # Same-provider identity is shared across symbol exports.
                    writer.writerow(dict(article_id='1', published_at_utc='2026-11-05T10:00:00Z',
                        headline='<b>Chip partnership</b>', summary='Revenue grew,\nwith new demand.',
                        source='Example publisher', url='https://example.com/1'))
        with (self.root / 'processed/NVDA_news.csv').open('a', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writerow(dict(article_id='2',published_at_utc='2026-11-06T10:00:00Z',
                headline='Future chip news', summary='Future detail',source='Example',url=''))
        self.source = app.Source(folder=self.root)
        self.db, self.audit = app.build_index(self.source, self.root / 'cache')

    def test_dedup_preserves_symbols_and_separates_providers(self):
        result = app.search_articles(self.db, '2026-11-05', symbol='META')
        self.assertEqual(result['total'], 2)
        self.assertEqual({r['provider'] for r in result['articles']}, {'alpaca', 'finnhub'})
        self.assertEqual(result['articles'][0]['symbols'], 'NVDA,SPY,META,PFE')

    def test_search_is_day_bounded_and_handles_query_syntax_as_text(self):
        result = app.search_articles(self.db, '2026-11-05', query='chip " OR *')
        self.assertEqual(result['total'], 2)
        self.assertTrue(all(r['article_id'] == '1' for r in result['articles']))
        self.assertEqual(app.search_articles(self.db, '2026-11-05', provider='alpaca')['total'], 1)
        self.assertEqual(app.search_articles(self.db, '2026-11-04')['total'], 0)
        with self.assertRaises(ValueError):
            app.search_articles(self.db, 'invalid')

    def test_text_is_plain_and_multiline_csv_is_preserved(self):
        row = app.search_articles(self.db, '2026-11-05')['articles'][0]
        self.assertEqual(row['headline'], 'Chip partnership')
        self.assertEqual(row['summary'], 'Revenue grew,\nwith new demand.')

    def test_lfs_pointer_is_reported_and_not_indexed_as_news(self):
        (self.root / 'alpaca/processed/NVDA_news.csv').write_text(
            'version https://git-lfs.github.com/spec/v1\noid sha256:123\nsize 1234\n')
        _, audit = app.build_index(self.source, self.root / 'cache', refresh=True)
        self.assertEqual(len(audit['missing']), 1)
        self.assertEqual(audit['missing'][0]['file'], 'alpaca/processed/NVDA_news.csv')

    def test_brief_uses_only_retrieved_text_and_flags_unknown_citations(self):
        with patch.object(app, 'ollama_request', return_value={
                'message': {'content': 'Evidence [alpaca:1]. Unsupported [alpaca:999].'}}) as model:
            result = app.brief(self.db, {'date': '2026-11-05', 'question': 'What changed?'})
        self.assertEqual(result['unknown_citations'], ['alpaca:999'])
        payload = model.call_args.args[1]
        evidence = json.loads(payload['messages'][1]['content'].split('Evidence:\n')[1])
        self.assertTrue(all(r['article_id'] == '1' for r in result['evidence']))
        self.assertNotIn('Future detail', json.dumps(evidence))
        self.assertTrue(result['has_citations'])


if __name__ == '__main__':
    unittest.main()
