"""Exploratory logistic regression with temporal evaluation and train-only selection."""
from pathlib import Path
import sys
import json
import hashlib
import html
from importlib.metadata import version
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from News_word_frequency.distinctive_terms import article_terms, prevalence, leaders, read, INPUT
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score,
    log_loss, brier_score_loss, confusion_matrix, precision_score, recall_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.exceptions import ConvergenceWarning

OUT = Path(__file__).resolve().parent / 'logistic_regression'
C = 0.1
warnings.filterwarnings('error', category=ConvergenceWarning)


def matrix(days, terms, day_terms):
    return np.array([[int(t in day_terms[d]) for t in terms] for d in days], dtype=float)


def choose_terms(train_days, day_terms, directions):
    ranked = prevalence(day_terms, directions, set(train_days), 5)
    # Remove constants: an always-present feature cannot distinguish directions.
    ranked = [r for r in ranked if r['total_support_days'] < len(train_days)]
    return sorted({r['term'] for side in ['up','down'] for r in leaders(ranked, side)})


def fit(x, y):
    model = LogisticRegression(C=C, solver='lbfgs', l1_ratio=0,
                               fit_intercept=True, max_iter=5000, tol=1e-8)
    return model.fit(x, y)


def metrics(y, p):
    predicted = (np.array(p) >= .5).astype(int)
    return {'n':len(y), 'actual_up':int(sum(y)), 'actual_down':int(len(y)-sum(y)),
        'accuracy':accuracy_score(y,predicted), 'balanced_accuracy':balanced_accuracy_score(y,predicted),
        'roc_auc':roc_auc_score(y,p) if len(set(y))>1 else None,
        'log_loss':log_loss(y,p,labels=[0,1]), 'brier_score':brier_score_loss(y,p),
        'up_precision':precision_score(y,predicted,zero_division=0),
        'up_recall':recall_score(y,predicted,zero_division=0),
        'confusion_matrix_rows_actual_down_up_columns_predicted_down_up':confusion_matrix(y,predicted,labels=[0,1]).tolist()}


def coefficients(model, terms, x, y, label):
    rows=[]
    for j,t in enumerate(terms):
        present=x[:,j]
        phi=float(np.corrcoef(present,y)[0,1]) if np.std(present)>0 and np.std(y)>0 else None
        rows.append({'term':t,'fit_scope':label,'coefficient_log_odds':float(model.coef_[0,j]),
            'odds_ratio_for_presence':float(np.exp(model.coef_[0,j])),
            'phi_correlation_with_up':phi,'support_days':int(present.sum()),
            'up_support_days':int(present[y==1].sum()),'down_support_days':int(present[y==0].sum())})
    return pd.DataFrame(rows).sort_values('coefficient_log_odds',ascending=False)


def main():
    OUT.mkdir(exist_ok=True)
    source_paths=[INPUT/'event_summary.csv',INPUT/'event_article_matches.csv',
                  Path(__file__).resolve().parent/'distinctive_terms/top20_up.csv',
                  Path(__file__).resolve().parent/'distinctive_terms/top20_down.csv']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    all_events=read(source_paths[0])
    events=sorted([e for e in all_events if e['collection_window_verified_24h']=='1'],key=lambda e:e['event_date'])
    days=[e['event_date'] for e in events]
    event_map={e['event_date']:e for e in events}
    directions={e['event_date']:'up' if e['price_move_1pct']=='1' else 'down' for e in events}
    day_terms={d:set() for d in days}; ids={d:set() for d in days}; cache={}
    for row in read(source_paths[1]):
        day=row['event_date']
        if day not in day_terms or row['in_24h']!='1': continue
        assert 0<float(row['hours_before_cutoff'])<=24
        key=row['article_match_key']
        cache.setdefault(key,article_terms(row))
        day_terms[day].update(cache[key]); ids[day].add(key)
    labels=lambda ds:np.array([int(directions[d]=='up') for d in ds])
    split=int(len(days)*.75)
    test=days[split:]
    # Conservative embargo: labels ending exactly at first evaluation cutoff are excluded.
    cutoff=pd.Timestamp(event_map[test[0]]['cutoff_utc'])
    train=[d for d in days[:split] if pd.Timestamp(event_map[d]['interval_end_utc'])<cutoff]
    embargo=sorted(set(days[:split])-set(train))
    assert max(train)<min(test)
    assert not set().union(*(ids[d] for d in train)) & set().union(*(ids[d] for d in test))
    terms=choose_terms(train,day_terms,directions)
    xtrain,xtest=matrix(train,terms,day_terms),matrix(test,terms,day_terms)
    ytrain,ytest=labels(train),labels(test)
    model=fit(xtrain,ytrain)
    p=model.predict_proba(xtest)[:,1]
    baseline=np.full(len(test),float(ytrain.mean()))
    coef=coefficients(model,terms,xtrain,ytrain,'training_period_only')
    coef.to_csv(OUT/'training_coefficients.csv',index=False)
    predictions=pd.DataFrame({'event_date':test,'actual_up':ytest,'probability_up':p,
        'predicted_up':(p>=.5).astype(int),'baseline_probability_up':baseline})
    predictions.to_csv(OUT/'test_predictions.csv',index=False)
    audit=pd.DataFrame({'event_date':days,'actual_up':labels(days),
        'split':['train' if d in train else 'test' if d in test else 'embargo' for d in days],
        'matched_articles_24h':[len(ids[d]) for d in days]})
    audit.to_csv(OUT/'split_audit.csv',index=False)
    # Training-period expanding-window diagnostics; reselect vocabulary inside each fold.
    cv=[]
    for fold,(tr,va) in enumerate(TimeSeriesSplit(n_splits=3,gap=1).split(train),1):
        td=[train[i] for i in tr]; vd=[train[i] for i in va]
        boundary=pd.Timestamp(event_map[vd[0]]['cutoff_utc'])
        td=[d for d in td if pd.Timestamp(event_map[d]['interval_end_utc'])<boundary]
        assert not set().union(*(ids[d] for d in td)) & set().union(*(ids[d] for d in vd))
        vocabulary=choose_terms(td,day_terms,directions)
        fitted=fit(matrix(td,vocabulary,day_terms),labels(td))
        fp=fitted.predict_proba(matrix(vd,vocabulary,day_terms))[:,1]
        cv.append({'fold':fold,'train_start':td[0],'train_end':td[-1],
                   'validation_start':vd[0],'validation_end':vd[-1],'features':vocabulary,
                   'model':metrics(labels(vd),fp),
                   'baseline':metrics(labels(vd),np.full(len(vd),float(labels(td).mean())))})
    # The previously displayed vocabulary used full-sample labels: descriptive only.
    inspected=sorted({r['term'] for path in source_paths[2:] for r in read(path)})
    xa,ya=matrix(days,inspected,day_terms),labels(days)
    descriptive=fit(xa,ya)
    description=coefficients(descriptive,inspected,xa,ya,'full_sample_descriptive_NOT_out_of_sample')
    description.to_csv(OUT/'inspected_words_descriptive_coefficients.csv',index=False)
    pd.DataFrame(matrix(days,terms,day_terms).astype(int),index=pd.Index(days,name='event_date'),
                 columns=['term::'+t for t in terms]).to_csv(OUT/'model_feature_matrix.csv')
    result={'source_sha256':hashes,'versions':{p:version(p) for p in ['numpy','pandas','scikit-learn','scipy']},
        'model':'L2 logistic regression; C=0.1 fixed before evaluation; binary presence features; threshold=0.5',
        'target':'1 = existing 1%-up flag, 0 = existing 1%-down flag; conditional on flagged events, not all trading days',
        'total_days':len(days),'train_days':len(train),'train_start':train[0],'train_end':train[-1],
        'train_up':int(ytrain.sum()),'train_down':int(len(ytrain)-ytrain.sum()),
        'embargo_days':embargo,'test_start':test[0],'test_end':test[-1],
        'selected_features':terms,'test_model':metrics(ytest,p),'test_baseline':metrics(ytest,baseline),
        'training_model':metrics(ytrain,model.predict_proba(xtrain)[:,1]),
        'expanding_window_training_diagnostics':cv,'intercept':float(model.intercept_[0]),
        'descriptive_intercept':float(descriptive.intercept_[0]),
        'limitations':['Small event-only sample, no neutral days and no volatility prediction.',
            'The dataset and full-sample vocabulary were already explored, so the temporal test is exploratory, not a pristine prospective holdout.',
            'Test labels are excluded from automated vocabulary selection and model fitting; full-sample descriptive fit is explicitly separate.',
            'Regularized coefficients and odds ratios are conditional associations, not causal effects or significance tests; no p-values.',
            'Correlated words, repeated news episodes and retrospective article versions limit inference.',
            'Missing coverage days excluded. Larger separate future data is needed to establish predictive usefulness.'],
        'validation':{'train_precedes_test':True,'no_identical_articles_across_split':True,
                      'fold_vocabulary_training_only':True,'converged':True}}
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(ROOT))] for p in source_paths)
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    page=['<!doctype html><html lang="en"><meta charset="utf-8"><title>NVDA logistic regression</title>',
          '<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 20px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}.note{background:#fff3d0;padding:14px}</style>',
          '<h1>Logistic regression: news before 1% up/down events</h1>',
          f'<p>Training: {train[0]}–{train[-1]} ({len(train)} days). Later test: {test[0]}–{test[-1]} ({len(test)} days). {len(terms)} train-selected binary word/phrase features.</p>',
          '<p class="note">Exploratory conditional direction model, not a volatility model. The full dataset was previously inspected. Coefficients are not evidence of causation or statistical significance.</p>',
          '<h2>Later-date evaluation</h2><table><tr><th>Metric</th><th>Logistic model</th><th>Training-class-frequency baseline</th></tr>']
    for key in ['accuracy','balanced_accuracy','roc_auc','log_loss','brier_score']:
        page.append(f'<tr><td>{key}</td><td>{result["test_model"][key]:.3f}</td><td>{result["test_baseline"][key]:.3f}</td></tr>')
    page.append('</table><p>Higher accuracy/AUC is better; lower log loss/Brier score is better.</p><h2>Training-only coefficients</h2>')
    page.append(coef.to_html(index=False,float_format=lambda v:f'{v:.3f}',escape=True))
    page.append('<h2>Previously inspected words: descriptive full-sample fit only</h2><p>This table includes hit and the other displayed words. It is not independently validated.</p>')
    page.append(description.to_html(index=False,float_format=lambda v:f'{v:.3f}',escape=True))
    page.append('<p>Odds ratio above 1 favors up; below 1 favors down, holding other included terms fixed. Phi is the unadjusted binary term–direction correlation on the stated fitting sample.</p></html>')
    (OUT/'report.html').write_text('\n'.join(page))
    print(json.dumps({k:result[k] for k in ['train_days','train_start','train_end','embargo_days','test_start','test_end','test_model','test_baseline','training_model']},indent=2))
    print('Descriptive hit:',description.loc[description.term.eq('hit')].to_dict('records'))


if __name__=='__main__':main()
