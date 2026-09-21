import unittest

from News_word_frequency.group_word_frequency import frequencies, group_membership


class GroupWordTests(unittest.TestCase):
    def test_inclusive_groups_use_saved_values(self):
        for value, signed, category in [('0.03', '1', '3pct_up'), ('-0.03', '-1', '3pct_down')]:
            flags = group_membership({'adjusted_close_return_1d': value, 'price_move_1pct': signed})
            self.assertTrue(flags[category])
        flags = group_membership({'adjusted_close_return_1d': '0.029999', 'price_move_1pct': '1'})
        self.assertFalse(flags['3pct_up'])
        self.assertTrue(flags['1pct_up'])

    def test_occurrences_differ_from_article_counts(self):
        articles = [{'headline': 'The chip chip', 'summary': 'AI 2026'},
                    {'headline': 'chip demand', 'summary': 'the'}]
        rows, total = frequencies(articles, True)
        self.assertEqual(rows[0]['word'], 'chip')
        self.assertEqual(rows[0]['occurrences'], 3)
        self.assertEqual(rows[0]['articles_containing_word'], 2)
        self.assertEqual(total, 5)
        self.assertNotIn('the', {r['word'] for r in rows})
        raw, _ = frequencies(articles, False)
        self.assertIn('the', {r['word'] for r in raw})


if __name__ == '__main__':
    unittest.main()
