import unittest
import numpy as np
import pandas as pd
from hourly_models.run_models import window_is_covered, split_days, classification, regression


class HourlyModelTests(unittest.TestCase):
    def test_coverage_excludes_exact_end_date(self):
        start=pd.Timestamp('2026-01-01T00:00:00Z')
        end=pd.Timestamp('2026-01-02T00:00:00Z')
        self.assertTrue(window_is_covered(start,end,{'2026-01-01'}))
        self.assertFalse(window_is_covered(start,end+pd.Timedelta(seconds=1),{'2026-01-01'}))

    def test_split_keeps_days_together_and_embargoes_overlapping_windows(self):
        times=pd.date_range('2026-01-01T14:30:00Z',periods=10,freq='D')
        rows=[]
        for t in times:
            for offset in [0,1]:
                s=t+pd.Timedelta(hours=offset)
                rows.append({'session':str(t.date()),'start_utc':s.isoformat(),
                             'end_utc':(s+pd.Timedelta(hours=1)).isoformat()})
        frame=pd.DataFrame(rows)
        train,test,gap=split_days(frame)
        self.assertEqual(test.sum(),6)
        self.assertEqual(gap.sum(),2)
        self.assertFalse(set(frame.loc[train,'session'])&set(frame.loc[test,'session']))
        boundary=pd.Timestamp(frame.loc[test,'start_utc'].min())-pd.Timedelta(hours=24)
        self.assertTrue((pd.to_datetime(frame.loc[train,'end_utc'],utc=True)<=boundary).all())

    def test_metrics_dont_hide_rare_class_failure(self):
        metrics=classification(np.array([0,0,0,1]),np.full(4,.25))
        self.assertEqual(metrics['accuracy'],.75)
        self.assertEqual(metrics['recall'],0)
        self.assertEqual(metrics['roc_auc'],.5)
        self.assertEqual(regression(np.array([.01,.02]),np.array([.01,.02]))['mae'],0)


if __name__=='__main__':unittest.main()
