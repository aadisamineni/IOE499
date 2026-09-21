import unittest
from News_word_frequency.distinctive_terms import candidates, article_terms, prevalence


class DistinctiveTermTests(unittest.TestCase):
    def test_extraction_boundaries_negation_and_alphanumeric(self):
        found = candidates('H200 is not approved. Strong demand! No restrictions. chip 2026 demand')
        for term in ['h200', 'not', 'no', 'not approved', 'strong demand', 'no restrictions']:
            self.assertIn(term, found)
        for term in ['2026', 'approved strong', 'demand no', 'chip demand', 'is']:
            self.assertNotIn(term, found)
        self.assertIn('not approved', candidates("isn't approved"))
        self.assertNotIn('chip demand', candidates('chip https://example.com demand'))
        self.assertNotIn('strong demand', article_terms({'headline':'strong', 'summary':'demand'}))
        self.assertNotIn('the and', candidates('the and'))

    def test_day_prevalence_minimum_and_unequal_denominators(self):
        terms = {'a':{'growth'}, 'b':{'growth'}, 'c':{'growth'}, 'd':set()}
        directions = {'a':'up','b':'up','c':'down','d':'down'}
        row = prevalence(terms,directions,set(terms),3)[0]
        self.assertEqual(row['total_support_days'],3)
        self.assertEqual(row['up_prevalence_pct'],100)
        self.assertEqual(row['down_prevalence_pct'],50)
        self.assertEqual(row['up_minus_down_pp'],50)
        self.assertEqual(prevalence(terms,directions,set(terms),4),[])
        self.assertEqual(prevalence(terms,directions,{'a','c','d'},1)[0]['up_total_days'],1)


if __name__ == '__main__':
    unittest.main()
