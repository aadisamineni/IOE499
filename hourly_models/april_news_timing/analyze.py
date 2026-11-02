"""Offline April 2026 descriptive hourly news-timing audit."""
import sys,json,html,hashlib
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from hourly_models.run_models import prepare_bars,prepare_news,window_is_covered,HOURLY,NEWS,COVERAGE
OUT=Path(__file__).resolve().parent

def main():
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HOURLY,NEWS,COVERAGE]}
    bars=prepare_bars(); april=bars[bars.session.str.startswith('2026-04')].copy()
    news,good,rejected=prepare_news()
    news=news[news.excluded.str.lower().ne('true')].copy()
    # Deduplicate only existing exact-copy groups; retain distinct coverage.
    news['exact_key']=news.exact_duplicate_group_id.where(news.exact_duplicate_group_id.ne(''),'id:'+news.article_id)
    news=news.drop_duplicates('exact_key').drop_duplicates('article_id')
    eligible=april[april.valid_price & april.duration_minutes.eq(60)].copy()
    eligible['absolute_return']=eligible.return_1h.abs()
    eligible=eligible.sort_values(['absolute_return','start'],ascending=[False,True])
    eligible['magnitude_rank']=range(1,len(eligible)+1)
    events=eligible[eligible.absolute_return.ge(.01) | eligible.magnitude_rank.le(10)]
    summaries=[]; links=[]
    for _,e in events.iterrows():
        s,t=e.start,e.end
        row={'event_id':s.isoformat(),'rank':int(e.magnitude_rank),'start_ny':s.tz_convert('America/New_York').isoformat(),'end_ny':t.tz_convert('America/New_York').isoformat(),'start_utc':s.isoformat(),'end_utc':t.isoformat(),'open':e.open,'close':e.close,'return_pct':100*e.return_1h,'at_least_1pct':bool(e.absolute_return>=.01)}
        windows={'before_1h':(s-pd.Timedelta(hours=1),s),'during':(s,t),'after_1h':(t,t+pd.Timedelta(hours=1)),'before_24h':(s-pd.Timedelta(hours=24),s),'after_24h':(t,t+pd.Timedelta(hours=24))}
        for name,(lo,hi) in windows.items():
            selected=news[(news.published>=lo)&(news.published<hi)]
            row[name+'_count']=len(selected)
            row[name+'_direct_count']=int(selected.relevance_type.eq('direct').sum())
            row[name+'_coverage_verified']=window_is_covered(lo,hi,good)
        selected=news[(news.published>=s-pd.Timedelta(hours=24))&(news.published<t+pd.Timedelta(hours=24))]
        for _,a in selected.iterrows():
            p=a.published
            phase='before' if p<s else ('during' if p<t else 'after')
            item={k:a[k] for k in ['article_id','headline','summary','source','url','published_at_utc','relevance_type','event_category']}
            item.update(event_id=row['event_id'],rank=row['rank'],return_pct=row['return_pct'],phase=phase,published_ny=p.tz_convert('America/New_York').isoformat(),hours_from_start=(p-s).total_seconds()/3600,hours_from_end=(p-t).total_seconds()/3600)
            for name,(lo,hi) in windows.items():item[name]=bool(lo<=p<hi)
            links.append(item)
        summaries.append(row)
    summary=pd.DataFrame(summaries); pairs=pd.DataFrame(links)
    assert not pairs.duplicated(['event_id','article_id']).any()
    for _,r in summary.iterrows():
        group=pairs[pairs.event_id.eq(r.event_id)]
        for name in windows:assert int(group[name].sum())==r[name+'_count']
        assert r.before_1h_count<=r.before_24h_count and r.after_1h_count<=r.after_24h_count
    assert (pairs.loc[pairs.phase.eq('before'),'hours_from_start']<0).all()
    assert (pairs.loc[pairs.phase.eq('after'),'hours_from_end']>=0).all()
    summary.to_csv(OUT/'event_summary.csv',index=False);pairs.to_csv(OUT/'event_article_matches.csv',index=False)
    april.to_csv(OUT/'april_bar_audit.csv',index=False);rejected.to_csv(OUT/'timestamp_review.csv',index=False)
    top=summary.head(10)
    fig,ax=plt.subplots(figsize=(12,6));x=list(range(len(top)))
    for shift,col,label in [(-.25,'before_1h_count','Hour before start'),(0,'during_count','During move'),(.25,'after_1h_count','Hour after end')]:ax.bar([v+shift for v in x],top[col],width=.25,label=label)
    ax.set_xticks(x,[f"{r.start_ny[5:10]} {r.start_ny[11:16]}\n{r.return_pct:+.2f}%" for r in top.itertuples()],rotation=35,ha='right');ax.set_ylabel('Distinct collected articles');ax.set_title('Largest April hourly moves: news timing (New York time)');ax.legend();fig.tight_layout();fig.savefig(OUT/'news_timing.png',dpi=160);plt.close(fig)
    report=['<html><meta charset="utf-8"><title>April news timing</title><body><h1>April 2026 NVDA hourly news timing</h1><p>Ranked by absolute open-to-close return, full 60-minute regular-session bars only. Timing is descriptive: after publication does not establish that an article discusses this move. News includes retained direct and indirect coverage; exact copies deduplicated. Historical versions are uncertain. All times below are New York time.</p><img src="news_timing.png" width="1000">',summary.to_html(index=False)]
    for r in top.itertuples():
        report.append(f'<h2>#{r.rank}: {r.start_ny} to {r.end_ny}, {r.return_pct:+.2f}%</h2>')
        for phase in ['before','during','after']:
            group=pairs[pairs.event_id.eq(r.event_id)&pairs.phase.eq(phase)]
            report.append(f'<h3>{phase}: {len(group)} articles (24-hour outer windows)</h3><ul>')
            for a in group.itertuples():report.append(f'<li>{html.escape(a.published_ny)} [{a.relevance_type}] <a href="{html.escape(a.url,quote=True)}">{html.escape(a.headline)}</a><p>{html.escape(a.summary)}</p></li>')
            report.append('</ul>')
    (OUT/'report.html').write_text('\n'.join(report)+ '</body></html>')
    result={'month':'2026-04','source_hashes':hashes,'source_bars':len(april),'full_valid_hours':len(eligible),'invalid_price_bars':int((~april.valid_price).sum()),'short_bars':int(april.duration_minutes.ne(60).sum()),'hours_at_least_1pct':int(eligible.absolute_return.ge(.01).sum()),'events_exported':len(summary),'event_article_pairs':len(pairs),'unique_linked_articles':int(pairs.article_id.nunique()),'ambiguous_timestamps_all_dates':len(rejected),'unverified_event_windows':int((~summary.filter(like='_coverage_verified')).sum().sum()),'validation':'Pair uniqueness, phase boundaries, counts, nested windows, and unchanged input hashes checked. Source bar schedule/price validation reused from hourly_models.'}
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in hashes.items())
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));print(top[['rank','start_ny','return_pct','before_1h_count','during_count','after_1h_count','before_24h_count','after_24h_count']].to_string(index=False))
if __name__=='__main__':main()
