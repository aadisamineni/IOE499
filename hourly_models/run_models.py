"""Hourly forward-return logistic and ridge models, with audited temporal matching."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import os
import sys
import warnings
from importlib.metadata import version
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import (roc_auc_score,average_precision_score,log_loss,brier_score_loss,
    accuracy_score,balanced_accuracy_score,precision_score,recall_score,confusion_matrix,
    mean_absolute_error,mean_squared_error,r2_score,roc_curve,precision_recall_curve)
from sklearn.exceptions import ConvergenceWarning
from News_word_frequency.distinctive_terms import article_terms
from matching.match_news import parse_timestamp

OUT=Path(__file__).resolve().parent/'output'
HOURLY=ROOT/'data/raw/NVDA_hourly_2025-10-01_2026-05-31.csv'
NEWS=ROOT/'news data pull/processed/all_classified_articles.csv'
COVERAGE=ROOT/'data/matched/2025-09-15_2026-05-15/collection_coverage_by_day.csv'
WORDS=['ai','chip','earnings','infrastructure','demand','china','hit']
MARKET=['prior_return','prior_abs_return','prior_volatility_20bars','log_prior_volume',
        'log_average_volume_20bars','minutes_since_open','hours_since_prior_bar']
COUNTS=['log_article_count_1h','log_article_count_24h']
TEXT=[f'log_{w}_articles_{h}h' for h in [1,24] for w in WORDS]
SETS={'market_only':MARKET,'market_plus_news_count':MARKET+COUNTS,'market_plus_news_words':MARKET+COUNTS+TEXT}
THRESHOLD=.01
warnings.filterwarnings('error',category=ConvergenceWarning)


def hash_file(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_bars():
    bars=pd.read_csv(HOURLY)
    bars['start']=pd.to_datetime(bars.trading_timestamp,utc=True)
    assert bars.start.is_monotonic_increasing and bars.start.is_unique
    bars['session']=bars.start.dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
    schedule=mcal.get_calendar('NASDAQ').schedule(start_date=bars.session.min(),end_date=bars.session.max())
    byday={str(d.date()):r for d,r in schedule.iterrows()}
    bars['session_open']=[byday[d].market_open for d in bars.session]
    bars['session_close']=[byday[d].market_close for d in bars.session]
    bars['end']=[min(t+pd.Timedelta(hours=1),c) for t,c in zip(bars.start,bars.session_close)]
    assert ((bars.start>=bars.session_open)&(bars.start<bars.session_close)).all()
    assert (((bars.start-bars.session_open).dt.total_seconds()%3600)==0).all()
    # Confirm provider hasn't omitted an expected row. Null rows are retained as gaps.
    for day,group in bars.groupby('session'):
        expected=pd.date_range(byday[day].market_open,byday[day].market_close,freq='h',inclusive='left')
        assert set(group.start)==set(expected),f'Missing hourly timestamps: {day}'
    cols=['open','high','low','close','adjusted_close','volume']
    valid=np.isfinite(bars[cols]).all(axis=1)&(bars[['open','high','low','close','adjusted_close']]>0).all(axis=1)&(bars.volume>=0)
    valid &= (bars.high>=bars.open)&(bars.high>=bars.close)&(bars.low<=bars.open)&(bars.low<=bars.close)&(bars.high>=bars.low)
    bars['valid_price']=valid
    bars['duration_minutes']=(bars.end-bars.start).dt.total_seconds()/60
    bars['return_1h']=(bars.close/bars.open-1).where(valid & bars.duration_minutes.eq(60))
    # All price controls refer only to completed bars, never current high/low/close/volume.
    bars['prior_return']=bars.return_1h.shift(1)
    bars['prior_abs_return']=bars.prior_return.abs()
    # Last 20 complete-duration bars; missing prices inside those bars propagate.
    full=bars.loc[bars.duration_minutes.eq(60)]
    past_vol=full.return_1h.rolling(20,min_periods=20).std(ddof=1).shift(1)
    past_volume=full.volume.where(full.valid_price).rolling(20,min_periods=20).mean().shift(1)
    bars['prior_volatility_20bars']=past_vol.reindex(bars.index)
    bars['log_average_volume_20bars']=np.log1p(past_volume.reindex(bars.index))
    # At the opening, previous row can be a completed half-hour bar. Its own
    # open-close return is a lag control; never label it as an hourly outcome.
    bars['prior_return']=(bars.close/bars.open-1).where(valid).shift(1)
    bars['prior_abs_return']=bars.prior_return.abs()
    bars['log_prior_volume']=np.log1p(bars.volume.where(valid).shift(1))
    bars['hours_since_prior_bar']=(bars.start-bars.end.shift(1)).dt.total_seconds()/3600
    bars['minutes_since_open']=(bars.start-bars.session_open).dt.total_seconds()/60
    bars['control_latest_price_time']=bars.end.shift(1)
    return bars


def prepare_news():
    news=pd.read_csv(NEWS,dtype=str,keep_default_na=False)
    news=news.loc[news.relevance_type.isin(['direct','indirect'])].drop_duplicates().copy()
    parsed=news.published_at_utc.map(parse_timestamp)
    news['published']=[r[0] for r in parsed]
    rejected=news.loc[news.published.isna()].copy()
    news=news.loc[news.published.notna()].copy()
    news['terms']=[article_terms(r) for r in news.to_dict('records')]
    news=news.sort_values('published')
    coverage=pd.read_csv(COVERAGE,dtype=str,keep_default_na=False)
    good=set(coverage.groupby('coverage_date_utc').filter(
        lambda g:set(g.ticker)==set(coverage.ticker) and g.coverage_state.eq('success').all()).coverage_date_utc)
    return news,good,rejected


def window_is_covered(start,end,good):
    return all(str(t.date()) in good for t in pd.date_range(start.floor('D'),(end-pd.Timedelta(nanoseconds=1)).floor('D')))


def build_dataset():
    bars=prepare_bars();news,good,rejected=prepare_news()
    rows=[];audit=[];links=[]
    for _,r in bars.iterrows():
        t=r.start; lower=t-pd.Timedelta(hours=24)
        reasons=[]
        if r.duration_minutes!=60:reasons.append('not_full_60_minute_bar')
        if not r.valid_price:reasons.append('missing_or_invalid_current_prices')
        if not np.isfinite(r[MARKET].to_numpy(dtype=float)).all():reasons.append('missing_prior_market_controls')
        if not window_is_covered(lower,t,good):reasons.append('incomplete_or_uncertain_news_coverage')
        audit.append({'start_utc':t.isoformat(),'session':r.session,'duration_minutes':r.duration_minutes,
                      'included':int(not reasons),'exclusion_reasons':'|'.join(reasons)})
        if reasons:continue
        window=news.loc[(news.published>=lower)&(news.published<t)]
        # Existing source has no revision fields. Honor them if added later.
        for field in ['updated_at_utc','modified_at_utc','revised_at_utc','first_available_at_utc']:
            if field in window:
                acceptable=window[field].map(lambda s:not s or (parse_timestamp(s)[0] is not None and parse_timestamp(s)[0]<t))
                window=window.loc[acceptable]
        record={'start_utc':t.isoformat(),'end_utc':r.end.isoformat(),'session':r.session,
                'start_new_york':t.tz_convert('America/New_York').isoformat(),
                'return_1h':r.return_1h,'absolute_return_1h':abs(r.return_1h),
                'large_move_1pct':int(abs(r.return_1h)>=THRESHOLD),
                'control_latest_price_time':r.control_latest_price_time.isoformat(),
                **{k:float(r[k]) for k in MARKET}}
        assert r.control_latest_price_time<=t
        for h in [1,24]:
            subset=window.loc[window.published>=t-pd.Timedelta(hours=h)]
            record[f'log_article_count_{h}h']=np.log1p(len(subset))
            for word in WORDS:
                record[f'log_{word}_articles_{h}h']=np.log1p(sum(word in ts for ts in subset.terms))
        record['latest_news_timestamp']=window.published.max().isoformat() if len(window) else ''
        for a in window.itertuples():
            links.append({'start_utc':t.isoformat(),'article_id':a.article_id,'published_utc':a.published.isoformat(),
                          'in_1h':int(a.published>=t-pd.Timedelta(hours=1)),'in_24h':1})
        rows.append(record)
    frame=pd.DataFrame(rows).sort_values('start_utc').reset_index(drop=True)
    return frame,pd.DataFrame(audit),pd.DataFrame(links),rejected


def split_days(frame):
    sessions=sorted(frame.session.unique());cut=int(len(sessions)*.7)
    test=frame.session.isin(sessions[cut:])
    first=pd.Timestamp(frame.loc[test,'start_utc'].min())
    train=(~test)&(pd.to_datetime(frame.end_utc,utc=True)<=first-pd.Timedelta(hours=24))
    return train,test,~(train|test)


def classification(y,p):
    pred=np.asarray(p)>=.5
    return {'n':len(y),'positive_events':int(np.sum(y)),'prevalence':float(np.mean(y)),
        'roc_auc':float(roc_auc_score(y,p)) if len(set(y))>1 else None,
        'average_precision':float(average_precision_score(y,p)),
        'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
        'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),
        'brier_score':float(brier_score_loss(y,p)),'log_loss':float(log_loss(y,p,labels=[0,1])),
        'confusion_matrix_actual_rows_0_1':confusion_matrix(y,pred,labels=[0,1]).tolist()}


def regression(y,p):
    return {'mae':float(mean_absolute_error(y,p)),'rmse':float(np.sqrt(mean_squared_error(y,p))),
            'r2':float(r2_score(y,p)),'mean_prediction':float(np.mean(p))}


def paired_uncertainty(test):
    rng=np.random.default_rng(42);groups=[g for _,g in test.groupby('session')]
    differences=[]
    for _ in range(2000):
        b=pd.concat([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        y=b.large_move_1pct.to_numpy();a=b.absolute_return_1h.to_numpy()
        differences.append([np.mean((b.prob_market_plus_news_words-y)**2)-np.mean((b.prob_market_only-y)**2),
            np.mean(abs(b.pred_market_plus_news_words-a))-np.mean(abs(b.pred_market_only-a))])
    q=np.quantile(differences,[.025,.975],axis=0)
    return {'method':'2000 paired trading-day bootstrap samples; fixed fitted models; seed=42',
            'news_minus_market_brier_95pct':[float(x) for x in q[:,0]],
            'news_minus_market_MAE_95pct':[float(x) for x in q[:,1]],
            'interpretation':'Negative favors news. Conditional sampling uncertainty, not a causal/significance guarantee.'}


def plot_results(test,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(14,10))
    for name in ['market_only','market_plus_news_count','market_plus_news_words']:
        p=test['prob_'+name];fpr,tpr,_=roc_curve(test.large_move_1pct,p)
        axes[0,0].plot(fpr,tpr,label=f'{name}: {report["logistic"][name]["roc_auc"]:.3f}')
        precision,recall,_=precision_recall_curve(test.large_move_1pct,p)
        axes[0,1].plot(recall,precision,label=name)
    axes[0,0].plot([0,1],[0,1],'k--');axes[0,0].set(title='Test ROC: absolute hourly return ≥1%',xlabel='False positive rate',ylabel='True positive rate')
    axes[0,1].axhline(test.large_move_1pct.mean(),ls='--',color='gray',label='Test prevalence')
    axes[0,1].set(title='Test precision–recall',xlabel='Recall',ylabel='Precision')
    for ax in axes[0]:ax.legend(fontsize=8)
    names=list(report['ridge']);mae=[100*report['ridge'][n]['mae'] for n in names]
    axes[1,0].barh(names,mae,color='#4279a3');axes[1,0].set(title='Magnitude error: lower is better',xlabel='MAE (return percentage points)')
    actual=test.absolute_return_1h*100;pred=test.pred_market_plus_news_words*100
    axes[1,1].scatter(actual,pred,alpha=.5,s=18)
    lim=max(actual.max(),pred.max());axes[1,1].plot([0,lim],[0,lim],'k--')
    axes[1,1].set(title='Ridge with news: test observations',xlabel='Actual absolute return (%)',ylabel='Predicted absolute return (%)')
    fig.suptitle('Hourly news experiments — chronological test set',fontsize=16)
    fig.tight_layout();fig.savefig(OUT/'model_comparison.png',dpi=170);plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    paths=[HOURLY,NEWS,COVERAGE]
    hashes={str(p.relative_to(ROOT)):hash_file(p) for p in paths}
    frame,audit,links,rejected=build_dataset()
    train,test,gap=split_days(frame)
    frame['split']=np.select([train,test],['train','test'],default='embargo')
    assert set(frame.loc[train,'session']).isdisjoint(set(frame.loc[test,'session']))
    assert set(links.loc[links.start_utc.isin(frame.loc[train,'start_utc']),'article_id']).isdisjoint(
        set(links.loc[links.start_utc.isin(frame.loc[test,'start_utc']),'article_id']))
    assert (pd.to_datetime(links.published_utc,utc=True)<pd.to_datetime(links.start_utc,utc=True)).all()
    y=frame.large_move_1pct.to_numpy();a=frame.absolute_return_1h.to_numpy()
    assert len(set(y[train]))==len(set(y[test]))==2
    logistic={};ridge={};coeff=[]
    frame['prob_constant']=float(y[train].mean())
    frame['pred_training_mean']=float(a[train].mean());frame['pred_training_median']=float(np.median(a[train]))
    logistic['constant']=classification(y[test],frame.loc[test,'prob_constant'])
    for name in ['training_mean','training_median']:ridge[name]=regression(a[test],frame.loc[test,'pred_'+name])
    for name,features in SETS.items():
        x=frame[features]
        models={'logistic':make_pipeline(StandardScaler(),LogisticRegression(C=.1,l1_ratio=0,solver='lbfgs',max_iter=5000)),
                'ridge':make_pipeline(StandardScaler(),Ridge(alpha=10))}
        for kind,model in models.items():
            model.fit(x.loc[train],(y if kind=='logistic' else a)[train])
            if kind=='logistic':
                frame['prob_'+name]=model.predict_proba(x)[:,1]
                logistic[name]=classification(y[test],frame.loc[test,'prob_'+name])
            else:
                raw=model.predict(x);frame['raw_pred_'+name]=raw
                frame['pred_'+name]=np.maximum(raw,0)
                ridge[name]=regression(a[test],frame.loc[test,'pred_'+name])
                ridge[name]['negative_raw_test_predictions_clipped']=int((raw[test]<0).sum())
            fit=model.steps[-1][1];coef=np.ravel(fit.coef_)
            for feature,value in zip(features,coef):
                coeff.append({'model':name,'kind':kind,'feature':feature,'coefficient_per_training_sd':float(value),
                              'odds_ratio_per_training_sd':float(np.exp(value)) if kind=='logistic' else None})
    counts=lambda mask:{'hours':int(mask.sum()),'days':int(frame.loc[mask,'session'].nunique()),
                       'large_moves':int(frame.loc[mask,'large_move_1pct'].sum()),
                       'start':frame.loc[mask,'start_utc'].min(),'end':frame.loc[mask,'end_utc'].max()}
    report={'generated_utc':pd.Timestamp.now(tz='UTC').isoformat(),'hourly_source_commit':'c024dae052f7b5fca064e67e8242b054b8d005a6',
        'source_sha256':hashes,'versions':{p:version(p) for p in ['pandas','numpy','scikit-learn','pandas_market_calendars','matplotlib']},
        'threshold':THRESHOLD,'return_definition':'same full 60-minute bar close / open - 1; regular session; no overnight target',
        'news_windows':'[start-1h,start) and [start-24h,start); UTC comparisons, NASDAQ calendar with DST/early closes',
        'settings':{'logistic_C':.1,'ridge_alpha':10,'logistic_probability_threshold':.5,'train_day_fraction':.7,'overlap_embargo_hours':24},
        'input_bars':len(audit),'usable_hours':len(frame),'train':counts(train),'test':counts(test),'embargo_hours':int(gap.sum()),
        'exclusion_reason_counts_nonexclusive':dict(Counter(r for s in audit.exclusion_reasons for r in s.split('|') if r)),
        'news_timestamp_rejections':len(rejected),'feature_sets':SETS,'logistic':logistic,'ridge':ridge,
        'paired_uncertainty':paired_uncertainty(frame.loc[test]),
        'limitations':['One stock and already explored historical period; not a pristine prospective test.',
            'News collected retrospectively; versions and publisher coverage are not guaranteed.',
            'Known hourly price gaps remain missing; no interpolation. Final half-hour bars excluded as outcomes.',
            'Seven words fixed for this run; hit was motivated by prior full-sample exploration.',
            'Hourly rows within a day are dependent. News repeats across windows; test is separated by days and 24-hour embargo.',
            'Preceding windows are elapsed hours, not full weekend aggregation. First-hour targets omit overnight price jumps.',
            'Absolute return is a magnitude proxy, not a full measure of realized volatility. No causal interpretation.'],
        'validation':{'source_files_unchanged':True,'news_strictly_before_open':True,'controls_available_by_open':True,
                      'only_full_hour_targets':True,'train_test_dates_disjoint':True,'train_test_article_ids_disjoint':True,
                      'scaling_fitted_on_training_only':True,'models_converged':True}}
    assert all(hash_file(p)==hashes[str(p.relative_to(ROOT))] for p in paths)
    frame.to_csv(OUT/'hourly_dataset_and_predictions.csv',index=False)
    frame.loc[test].to_csv(OUT/'test_predictions.csv',index=False)
    audit.to_csv(OUT/'bar_audit.csv',index=False);links.to_csv(OUT/'hourly_article_links.csv',index=False)
    rejected.drop(columns=['terms'],errors='ignore').to_csv(OUT/'timestamp_review.csv',index=False)
    pd.DataFrame(coeff).to_csv(OUT/'coefficients.csv',index=False)
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    plot_results(frame.loc[test],report)
    page=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Hourly NVDA models</title><style>body{font:16px system-ui;max-width:1300px;margin:30px auto;padding:20px}img{width:100%}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}</style><h1>Hourly NVDA news models</h1>',
        f'<p>Logistic threshold: absolute hourly move ≥1%. Train: {train.sum()} hours. Test: {test.sum()} hours. News ends strictly before each full-hour price interval.</p>',
        '<img src="model_comparison.png" alt="Model performance graphs"><h2>Logistic regression: test</h2>',
        pd.DataFrame(logistic).T.drop(columns=['confusion_matrix_actual_rows_0_1']).to_html(float_format=lambda v:f'{v:.4f}'),
        '<h2>Ridge regression: test (returns are fractions)</h2>',pd.DataFrame(ridge).T.to_html(float_format=lambda v:f'{v:.6f}'),
        '<p>Negative R² means worse squared error than predicting the test mean. That is a diagnostic, not an available training baseline. Smaller MAE, RMSE, Brier and log loss are better.</p>',
        '<p>Exploratory historical results; no causal claims. See metrics.json and README.md for coverage exclusions and day-bootstrap uncertainty.</p></html>']
    (OUT/'report.html').write_text('\n'.join(page))
    print(json.dumps({k:report[k] for k in ['usable_hours','train','test','embargo_hours','logistic','ridge','paired_uncertainty']},indent=2))


if __name__=='__main__':main()
