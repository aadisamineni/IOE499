import unittest
import numpy as np
from News_word_frequency.run_logistic_regression import choose_terms, matrix, metrics


class LogisticNewsTests(unittest.TestCase):
    def test_feature_selection_ignores_future_terms_and_labels(self):
        train = [f'd{i}' for i in range(12)]
        directions = {d: 'up' if i < 6 else 'down' for i,d in enumerate(train)}
        terms = {d: {'alpha'} if i < 6 else {'beta'} for i,d in enumerate(train)}
        expected = choose_terms(train, terms, directions)
        terms['future'] = {'future_only', 'alpha'}
        directions['future'] = 'down'
        self.assertEqual(choose_terms(train, terms, directions), expected)
        self.assertNotIn('future_only', expected)
        self.assertEqual(matrix(['future'], expected, terms).shape, (1,2))

    def test_metrics_and_baseline(self):
        result = metrics(np.array([0,1,1,0]), np.array([.1,.8,.9,.2]))
        self.assertEqual(result['accuracy'],1)
        self.assertEqual(result['roc_auc'],1)
        self.assertEqual(result['confusion_matrix_rows_actual_down_up_columns_predicted_down_up'],[[2,0],[0,2]])
        baseline = metrics(np.array([0,1,1,0]), np.full(4,.6))
        self.assertEqual(baseline['accuracy'],.5)
        self.assertEqual(baseline['roc_auc'],.5)


if __name__=='__main__': unittest.main()
