"""Checks for the local CSV-to-Qwen dashboard integration."""
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
        self.csv = self.root / 'news.csv'
        fields = ['date_utc', *[f'weight_{s}' for s in app.SYMBOLS], 'provider', 'article_id',
                  'query_symbols', 'related_symbols', 'published_at_utc', 'headline', 'summary',
                  'publisher', 'url', 'retrieved_at_utc', 'content_html']
        rows = [
            ['2026-01-01', '.2', '.4', '.2', '.2', 'alpaca', 'a1', 'NVDA', 'NVDA', '2026-01-01T10:00:00Z', 'Chip news', 'First story', 'Example', 'https://example.com/a', '2026-11-01T00:00:00Z', '<p>Long content</p>'],
            ['2026-01-01', '.2', '.4', '.2', '.2', 'finnhub', 'b2', 'META', 'META', '2026-01-01T11:00:00Z', 'Meta update', 'Second story', 'Example', '', '2026-11-01T00:00:00Z', 'content'],
            ['2026-01-02', '.2', '.4', '.2', '.2', 'alpaca', 'c3', 'NVDA', 'NVDA', '2026-01-02T10:00:00Z', 'Future news', 'Future story', 'Example', '', '2026-11-01T00:00:00Z', 'content'],
        ]
        with self.csv.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(rows)
        self.db = app.build_index(self.csv, self.root / 'cache')
        self.output = {'date_utc':'2026-01-01', 'target_weights':{'NVDA':.2,'SPY':.4,'META':.2,'PFE':.2},
                       'reasoning':{s:'Explanation' for s in app.SYMBOLS}, 'supporting_article_ids':['a1'],
                       'portfolio_reasoning':'An illustrative allocation.'}

    def test_news_filtering_and_calendar_use_merged_csv(self):
        self.assertEqual(app.overview(self.db)['total'], 3)
        result = app.search_articles(self.db, '2026-01-01', symbol='META', provider='finnhub', query='Meta')
        self.assertEqual(result['total'], 1)
        self.assertEqual(result['articles'][0]['article_id'], 'b2')
        self.assertNotIn('content_html', result['articles'][0])
        self.assertEqual(app.search_articles(self.db, '2026-01-01', query='Future')['total'], 0)
        self.assertEqual(app.search_articles(self.db, '2026-01-01', query='%')['total'], 0)

    def test_analysis_calls_existing_qwen_with_one_matching_record(self):
        with patch.object(app.local_qwen, 'ask_ollama', return_value=json.dumps(self.output)) as model:
            result = app.analyze(self.db, {'date':'2026-01-01', 'symbol':'NVDA'})
        payload = model.call_args.args[0]
        self.assertEqual(payload['model'], app.local_qwen.MODEL)
        self.assertIn('a1', payload['messages'][1]['content'])
        self.assertNotIn('Future news', payload['messages'][1]['content'])
        self.assertNotIn('Long content', payload['messages'][1]['content'])
        self.assertEqual(result['output'], self.output)
        self.assertEqual(result['warnings'], [])
        self.assertEqual(result['evidence'][0]['retrieved_at_utc'], '2026-11-01T00:00:00Z')

    def test_flags_invalid_weights_and_citations(self):
        self.output['target_weights']['SPY'] = .1
        self.output['supporting_article_ids'] = ['not-in-input']
        with patch.object(app.local_qwen, 'ask_ollama', return_value=json.dumps(self.output)):
            result = app.analyze(self.db, {'date':'2026-01-01'})
        self.assertTrue(any('100%' in warning for warning in result['warnings']))
        self.assertTrue(any('constraints' in warning for warning in result['warnings']))
        self.assertTrue(any('article' in warning for warning in result['warnings']))

    def test_cluster_qwen_requests_a_json_answer_without_thinking(self):
        with patch.object(app.local_qwen, 'ask_ollama', return_value=json.dumps(self.output)) as model:
            app.analyze(self.db, {'date': '2026-01-01'}, model='qwen3:32b')
        payload = model.call_args.args[0]
        self.assertFalse(payload['think'])
        self.assertEqual(payload['format'], 'json')

    def test_rejects_invalid_model_output_and_empty_evidence(self):
        with patch.object(app.local_qwen, 'ask_ollama', return_value='not JSON'):
            with self.assertRaisesRegex(ValueError, 'invalid JSON'):
                app.analyze(self.db, {'date':'2026-01-01'})
        with self.assertRaisesRegex(ValueError, 'No articles'):
            app.analyze(self.db, {'date':'2026-01-01', 'query':'missing'})


if __name__ == '__main__':
    unittest.main()
