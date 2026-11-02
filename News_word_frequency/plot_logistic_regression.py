"""AVS-style descriptive scatterplots plus logistic evaluation graphics."""
from pathlib import Path
import os
import sys
import json
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from News_word_frequency.distinctive_terms import read,article_terms,INPUT

OUT=Path(__file__).resolve().parent/'logistic_regression'
PLOTS=OUT/'plots'
WORDS=['bullish','scale ai','known','deals','hit','securities']


def main():
    PLOTS.mkdir(exist_ok=True)
    audit=pd.read_csv(OUT/'split_audit.csv')
    eligible=set(audit.event_date)
    counts=defaultdict(lambda:defaultdict(set))
    for r in read(INPUT/'event_article_matches.csv'):
        if r['event_date'] not in eligible or r['in_24h']!='1':continue
        found=article_terms(r)
        for word in WORDS:
            if word in found:counts[r['event_date']][word].add(r['article_match_key'])
    stock=pd.read_csv(ROOT/'data/processed/NVDA_daily_features_2025-10-01_2026-05-31.csv').set_index('trading_date')
    frame=audit.copy()
    frame['return_pct']=[100*stock.loc[d,'adjusted_close_return_1d'] for d in frame.event_date]
    for word in WORDS:frame[word]=[len(counts[d][word]) for d in frame.event_date]
    frame.to_csv(OUT/'scatter_source_data.csv',index=False)
    correlations=[]
    fig,axes=plt.subplots(2,3,figsize=(15,9),sharey=True)
    for ax,word in zip(axes.flat,WORDS):
        x,y=frame[word],frame.return_pct
        ax.scatter(x,y,s=24,alpha=.6,color='#76b900')
        slope,intercept=np.polyfit(x,y,1)
        grid=np.linspace(x.min(),x.max(),100)
        ax.plot(grid,slope*grid+intercept,color='#c0392b',lw=1.5)
        corr=x.corr(y)
        correlations.append({'term':word,'pearson_r':corr,'slope_return_percentage_points_per_article':slope,'n':len(frame)})
        ax.text(.04,.94,f'r = {corr:+.3f} | n = {len(frame)}',transform=ax.transAxes,va='top')
        ax.set_title(word);ax.set_xlabel('Preceding articles containing term (24h)')
        ax.axhline(0,color='gray',lw=.6);ax.grid(alpha=.2)
    for ax in axes[:,0]:ax.set_ylabel('Flagged-session adjusted close return (%)')
    fig.suptitle('Term frequency versus NVDA return — descriptive linear fits',fontsize=16)
    fig.text(.5,.01,'AVS-style scatterplots. Full 92-event sample; selected terms; no neutral days. Not logistic curves or out-of-sample results.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.035,1,.96]);fig.savefig(PLOTS/'term_return_scatter.png',dpi=180);plt.close(fig)
    pd.DataFrame(correlations).to_csv(OUT/'scatter_correlations.csv',index=False)

    fig,ax=plt.subplots(figsize=(16,7));right=ax.twinx()
    dates=pd.to_datetime(frame.event_date)
    for word in WORDS:ax.plot(dates,frame[word],label=word,lw=1.2,alpha=.8)
    right.plot(dates,frame.return_pct,color='black',label='NVDA return (%)',lw=1,alpha=.7)
    right.axhline(0,color='black',lw=.5);ax.set_ylabel('Preceding articles containing term (24h)')
    right.set_ylabel('Flagged-session adjusted close return (%)');ax.set_xlabel('Flagged event date')
    ax.set_title('Preceding term frequency and flagged-session return — descriptive')
    h,l=ax.get_legend_handles_labels();h2,l2=right.get_legend_handles_labels();ax.legend(h+h2,l+l2,ncol=4,loc='upper left')
    fig.tight_layout();fig.savefig(PLOTS/'layered_terms_and_returns.png',dpi=180);plt.close(fig)

    coef=pd.read_csv(OUT/'training_coefficients.csv').sort_values('coefficient_log_odds')
    shown=pd.concat([coef.head(10),coef.tail(10)]).drop_duplicates('term')
    fig,ax=plt.subplots(figsize=(11,9))
    ax.barh(shown.term,shown.coefficient_log_odds,color=['#c0392b' if c<0 else '#76b900' for c in shown.coefficient_log_odds])
    ax.axvline(0,color='black',lw=.8);ax.set_xlabel('Regularized coefficient: negative favors down, positive favors up')
    ax.set_title('Logistic regression coefficients — training data only')
    fig.text(.5,.01,'Binary term presence; coefficients are conditional associations, not significance tests.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.035,1,1]);fig.savefig(PLOTS/'logistic_coefficients.png',dpi=180);plt.close(fig)

    result=json.loads((OUT/'metrics.json').read_text())
    pred=pd.read_csv(OUT/'test_predictions.csv')
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    fpr,tpr,_=roc_curve(pred.actual_up,pred.probability_up)
    axes[0].plot(fpr,tpr,color='#76b900',lw=2,label=f'Logistic AUC = {result["test_model"]["roc_auc"]:.3f}')
    axes[0].plot([0,1],[0,1],'--',color='gray',label='Chance / constant baseline')
    axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='Later-date ROC');axes[0].legend(fontsize=9)
    cm=np.array(result['test_model']['confusion_matrix_rows_actual_down_up_columns_predicted_down_up'])
    axes[1].imshow(cm,cmap='Greens');axes[1].set(xticks=[0,1],yticks=[0,1],xticklabels=['Down','Up'],yticklabels=['Down','Up'],xlabel='Predicted',ylabel='Actual',title='Test confusion matrix')
    for i in range(2):
        for j in range(2):axes[1].text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=20,color='black')
    names=['Training','Later test','Test baseline']
    vals=[result['training_model']['accuracy'],result['test_model']['accuracy'],result['test_baseline']['accuracy']]
    bars=axes[2].bar(names,vals,color=['#76b900','#2b6cb0','#999999'])
    axes[2].set(ylim=(0,1.12),ylabel='Accuracy',title='Generalization check')
    for bar,v in zip(bars,vals):axes[2].text(bar.get_x()+bar.get_width()/2,v+.025,f'{v:.1%}',ha='center')
    fig.suptitle('Logistic regression — 23 later test events (13 up, 10 down)',fontsize=15)
    fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(PLOTS/'logistic_test_performance.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(13,5))
    x=pd.to_datetime(pred.event_date)
    ax.plot(x,pred.probability_up,'o-',color='#2b6cb0',label='Predicted probability of up')
    ax.scatter(x,pred.actual_up,color=['#76b900' if v else '#c0392b' for v in pred.actual_up],marker='x',s=65,label='Actual direction (up=1, down=0)')
    ax.axhline(.5,ls='--',color='gray',label='Decision threshold')
    ax.set(ylim=(-.08,1.08),ylabel='Probability / actual binary direction',xlabel='Test event date',title='Logistic predictions on later dates')
    ax.legend(fontsize=9);fig.tight_layout();fig.savefig(PLOTS/'test_probabilities.png',dpi=180);plt.close(fig)

    files=[('term_return_scatter.png','AVS-style scatterplots: linear descriptive fits'),
           ('layered_terms_and_returns.png','AVS-style frequency and return timeline'),
           ('logistic_coefficients.png','Logistic model: training coefficients'),
           ('logistic_test_performance.png','Logistic model: later-date evaluation'),
           ('test_probabilities.png','Logistic model: test probabilities')]
    page=['<!doctype html><html lang="en"><meta charset="utf-8"><title>NVDA regression graphs</title><style>body{font:16px system-ui;max-width:1300px;margin:30px auto;padding:0 20px}img{width:100%}p{line-height:1.5}</style><h1>NVDA regression graphs</h1>',
          '<p>The first two charts mirror AVS’s descriptive plot format. Frequency here means distinct preceding articles containing a term, with the strict 24-hour cutoff. The remaining charts evaluate logistic regression. Neither establishes causation.</p>',
          '<p><b>Later-date accuracy: 52.2%, versus 56.5% for the baseline. AUC: 0.531.</b> This test does not establish useful predictive performance.</p>']
    for name,title in files:page.append(f'<h2>{title}</h2><a href="plots/{name}"><img src="plots/{name}" alt="{title}"></a>')
    page.append('<p><a href="report.html">Full metrics and coefficient tables</a></p></html>')
    (OUT/'graphs.html').write_text('\n'.join(page))
    print(pd.DataFrame(correlations).to_string(index=False));print('Graphs:',OUT/'graphs.html')


if __name__=='__main__':main()
