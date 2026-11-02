"""April hourly move browser; provisional content organization, not causal labels."""
from pathlib import Path
import sys,html,json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from hourly_models.april_news_timing.article_review.build_review import ANNOTATIONS
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
EXTRA='''139609404|D|Reports Nvidia/Marvell partnership; original announcement time unknown.
139778663|D|Reports Cadence/Nvidia robotics collaboration.
139777575|D|Reports Jane Street cloud commitment and equity investment.
139777574|D|Reports Wednesday statements on Cadence/Nvidia robotics collaboration.
139777569|DP|Reports quantum-stock rally and Nvidia software tools, not NVDA shares.
139789440|D|Reports Meta/Broadcom partnership extension.
139777566|D|Dated Cadence announcement describes expanded Nvidia collaboration.
139777608|D|Reports Meta/Broadcom agreement through 2029.
139781894|O|Analysis of AMD competitive positioning rather than a timed announcement.
139778900|D|Reports Cadence partnerships announced in Wednesday press releases.
139778906|DO|Reports expanded chip partnership and board change with interpretation.
139781874|DO|Interprets expanded Meta/Broadcom partnership and AI spending.
139762280|DPO|Reports analyst coverage initiation and historical quantum-stock underperformance.
139763736|O|General outlook on Nvidia data-center strategy.
139764695|D|Reports named Ising model release.
139762269|PO|Promotional comparison of another company's stock performance.
139765667|DPN|Reports NVDA up 1.59% and Ising launch; asserted connection is the publisher's.
139763659|D|Dated IQM announcement describes use of Nvidia Ising.
139763658|D|Nvidia release explicitly announces Ising today.
139763656|O|Speculative investment case for Arm.
139768014|PO|Reports Micron rally and analysis of memory demand, not NVDA share movement.
139777000|DPN|Reports NVDA streak and denial of acquisition rumors; historical timing uncertain.
139763654|DU|Headline suggests analyst revisions but truncated text does not identify the change.
139775326|DO|Interprets Nvidia multi-chip strategy; first-release timing unknown.
139763648|O|Industry interview about existing models and future quantum computing.
139763647|PNO|Reports NVDA winning streak and buy/sell discussion ahead of earnings.
139907220|PNO|Interprets risks from Nvidia rally and ETF concentration.
139918112|D|Reports accounting officer succession; summary dates underlying event April 24.
139907414|D|Reports accounting officer succession in SEC filing; release time unknown.
139905670|PNO|Reports NVDA approaching highs and discusses buy zone.
139905666|O|Investment thesis using historical GPU shipments.
139905664|O|Compares past quarterly results to support investment opinion.
139915945|PNO|Historical stock-performance comparison since index changes.
139906766|O|Investment thesis for AMD based on CPU shortages and historical results.
139906763|PNO|Discusses last week's chip-stock gains and valuation.'''
CATEGORIES=['Potentially relevant development','Development + price reporting','Price reporting','Background / opinion','Unclear']
# Incidental mentions, historical investment examples, or insufficient factual detail
# should not be presented as substantive information about this NVDA move.
OVERRIDES={
'139876211':('Background / opinion','Award story mentions Nvidia incidentally; no clear connection to its price move.'),
'139915271':('Background / opinion','Third-party promotional launch lists Nvidia as a competitor; relevance to this move is not established.'),
'139882104':('Unclear','Headline-only trading-service alert provides too little evidence.'),
'139708463':('Unclear','Incomplete allegation and analogy do not establish a specific timely development.'),
'139775563':('Unclear','Vague model reference and quantum-stock rally; insufficient detail to connect to NVDA.'),
'139882100':('Unclear','Closing milestone at 12:23 New York raises historical-version/timestamp concerns.'),
'139763654':('Unclear','Truncated text does not identify the analyst action.'),
'139609402':('Background / opinion','Explicit March 16 announcement repeated in April; not fresh news at publication.'),
'139751490':('Background / opinion','Discussion of a recently completed investment, not a verified fresh announcement.'),
'139729864':('Background / opinion','Six-month performance supports an investment comparison, not this hourly move.'),
'139774858':('Background / opinion','Yesterday’s rally supports valuation commentary, not this selected hourly move.'),
'139876214':('Background / opinion','Historical relative performance and previous close dominate investment discussion.'),
'139752938':('Background / opinion','Last-week performance and analyst commentary do not identify a new catalyst.'),
'139915945':('Background / opinion','Historical comparison since Dow membership changes, not this hourly move.'),
'139906763':('Background / opinion','Last-week gains and valuation commentary, not this hourly move.')}

def esc(s):return html.escape(str(s),quote=True)
def main():
    inputs=[BASE/'event_summary.csv',BASE/'event_article_matches.csv',BASE/'april_bar_audit.csv',ROOT/'news data pull/processed/all_classified_articles.csv']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    events=pd.read_csv(inputs[0]);pairs=pd.read_csv(inputs[1],keep_default_na=False)
    pairs=pairs[pairs.before_1h|pairs.during|pairs.after_1h].copy()
    notes={}
    for line in (ANNOTATIONS+'\n'+EXTRA).splitlines():
        key,codes,note=line.split('|',2);assert key not in notes;notes[key]=(codes,note)
    assert set(pairs.article_id.astype(str))==set(notes)
    source=pd.read_csv(inputs[3],dtype=str,keep_default_na=False).drop_duplicates('article_id').set_index('article_id')
    records=[]
    for _,a in pairs.iterrows():
        key=str(a.article_id);codes,note=notes[key]
        if key in OVERRIDES:category,reason=OVERRIDES[key]
        elif 'U' in codes:category,reason='Unclear',note
        elif 'D' in codes and 'N' in codes:category,reason='Development + price reporting',note
        elif 'D' in codes:category,reason='Potentially relevant development',note
        elif 'N' in codes:category,reason='Price reporting',note
        else:category,reason='Background / opinion',note
        r=a.to_dict();r.update(category=category,review_reason=reason,content_note=note,classification_status='assistant_provisional',first_announcement_time_verified=False,connection_to_exact_hour='not_established',story_group=source.loc[key,'duplicate_group_id'] or 'article_'+key,historical_version_warning=('Live Coverage' in a.headline or key in ['139882100','139775561','139777000']),supporting_text=html.unescape(a.headline+'\n'+a.summary))
        records.append(r)
    reviewed=pd.DataFrame(records);reviewed.to_csv(HERE/'classified_matches.csv',index=False)
    counts=[]
    for e in events.itertuples():
        g=reviewed[reviewed.event_id.eq(e.event_id)]
        for phase in ['before','during','after']:
            for cat in CATEGORIES:
                s=g[g.phase.eq(phase)&g.category.eq(cat)]
                counts.append(dict(event_id=e.event_id,rank=e.rank,phase=phase,category=cat,articles=len(s),existing_story_groups=s.story_group.nunique()))
    counts=pd.DataFrame(counts);counts.to_csv(HERE/'event_category_counts.csv',index=False)
    overview=counts.groupby(['phase','category'],sort=False).articles.sum().unstack(fill_value=0).reindex(['before','during','after'])[CATEGORIES]
    overview.to_csv(HERE/'month_category_counts.csv')
    css='''body{font:16px/1.55 system-ui;color:#172b40;background:#f1f5f9;margin:0}main{max-width:1150px;margin:auto;padding:28px}h1,h2{line-height:1.2}header,section{background:white;padding:24px;border-radius:14px;margin-bottom:24px}a{color:#145db1}.muted{color:#526579}.metrics{display:flex;gap:20px;flex-wrap:wrap}.metric{padding:16px;background:#edf4ff;border-radius:10px}.metric b{font-size:26px;display:block}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #dbe3eb;text-align:left}.scroll{overflow:auto}.card{border:1px solid #dbe3eb;padding:16px;border-radius:10px;margin:12px 0}.badge{font-size:13px;padding:4px 9px;border-radius:20px;background:#edf0f4;margin-right:7px}.before{background:#dbefe4}.during{background:#fff0cb}.after{background:#e5e4fb}.warning{background:#fff3cd;padding:10px}button{padding:9px 16px;margin:5px;border:1px solid #b7c5d5;border-radius:8px;cursor:pointer;background:white}button.active{background:#183c63;color:white}summary{cursor:pointer;font-weight:650;padding:10px 0}svg{width:100%;height:auto}nav{display:flex;gap:10px;flex-wrap:wrap}h3{margin-bottom:4px}'''
    page=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>April NVDA: moves and news</title><style>'+css+'</style><main><header><h1>April NVDA: price moves & nearby news</h1><p>Start with a move. See which articles report developments, which discuss prices, and which provide background.</p><div class="metrics">']
    for label,value in [('Hourly moves ≥1%',len(events)),('Distinct articles',reviewed.article_id.nunique()),('Article–move links',len(reviewed))]:page.append(f'<div class="metric"><b>{value}</b>{label}</div>')
    page.append('</div><p class="muted">April 2026 · New York time · full 60-minute regular-session bars · ±1 hour of news around each bar. Final 30-minute bars and overnight gaps are excluded.</p><p class="warning">Provisional assistant review of headlines and summaries. “Potentially relevant” means a plausible subject connection, not a verified catalyst or a new announcement at that time. No article is confirmed to explain the exact hourly move. Prices cannot locate the jump within the hour.</p></header><section><h2>How to read this</h2><p><b>Relevance section</b> describes content. The <b>timing badge</b> independently says when it was published. For a 9:30–10:30 move: before = 8:30–9:30, during = 9:30–10:30, after = 10:30–11:30. Boundaries belong to the following window.</p><p>Each article has one primary section per move. Older performance commentary and incidental mentions go under background. Related coverage is bundled using the existing story groups; these groups are approximate, not independently verified stories.</p><h2>Month overview</h2><p>Counts below are article–move links. Articles around adjacent moves can count again; these are not independent observations.</p><div class="scroll">'+overview.to_html()+'</div><p>All selected news windows have successful collection records; this does not guarantee complete publisher coverage. “No collected articles” is not evidence that no news existed.</p><nav>')
    for e in events.sort_values('start_ny').itertuples():page.append(f'<a href="#move{e.rank}">{e.start_ny[5:10]} {e.start_ny[11:16]} ({e.return_pct:+.2f}%)</a>')
    page.append('</nav></section>')
    bars=pd.read_csv(inputs[2]);
    for e in events.sort_values('start_ny').itertuples():
        g=reviewed[reviewed.event_id.eq(e.event_id)]
        day=e.start_ny[:10];daybars=bars[bars.session.eq(day)].copy()
        near=events[(events.start_ny.str[:10].eq(day))&events['rank'].ne(e.rank)]
        page.append(f'<section class="event" id="move{e.rank}"><h2>{day} · {e.start_ny[11:16]}–{e.end_ny[11:16]} · {e.return_pct:+.2f}%</h2><p>Magnitude rank #{e.rank} · Open ${e.open:.2f} → close ${e.close:.2f} · {len(g)} collected articles · {g.story_group.nunique()} existing story groups</p>')
        if len(near):page.append('<p class="muted">Other ≥1% hours this day: '+', '.join(f'<a href="#move{r.rank}">{r.start_ny[11:16]} ({r.return_pct:+.2f}%)</a>' for r in near.itertuples())+'. These may be parts of a shared episode; they are not merged or treated as independent evidence.</p>')
        low,high=daybars.low.min(),daybars.high.max();span=high-low or 1
        def y(v):return 165-(v-low)/span*125
        svg=['<svg viewBox="0 0 900 225" role="img" aria-label="Hourly OHLC prices for this session; selected hour highlighted"><rect width="900" height="225" fill="#f8fafc"/>']
        for i,b in enumerate(daybars.itertuples()):
            x=95+i*110;selected=str(b.trading_timestamp)[:16]==e.start_ny[:16];color='#168255' if b.close>=b.open else '#c54242'
            if selected:svg.append(f'<rect x="{x-35}" y="15" width="70" height="165" fill="#dbeafe"/>')
            svg.append(f'<line x1="{x}" x2="{x}" y1="{y(b.high)}" y2="{y(b.low)}" stroke="{color}"/><rect x="{x-15}" width="30" y="{min(y(b.open),y(b.close))}" height="{max(1,abs(y(b.open)-y(b.close)))}" fill="{color}"/><text x="{x}" y="200" text-anchor="middle" font-size="13">{str(b.trading_timestamp)[11:16]}</text>')
        svg.append(f'<text x="8" y="35" font-size="12">${high:.2f}</text><text x="8" y="168" font-size="12">${low:.2f}</text></svg>')
        page.extend(svg);page.append('<p class="muted">Hourly OHLC candles, not a tick path. Blue background selects this move; the last candle is 30 minutes.</p><div class="filters">')
        for phase in ['all','before','during','after']:page.append(f'<button type="button" class="{"active" if phase=="all" else ""}" onclick="filterEvent(this,\'{phase}\')">{phase.title()}'+('' if phase=='all' else f' ({int(g.phase.eq(phase).sum())})')+'</button>')
        page.append('</div>')
        for cat in CATEGORIES:
            cg=g[g.category.eq(cat)].sort_values('published_at_utc')
            page.append(f'<details class="category" open><summary>{cat} · <span class="visible-count">{len(cg)}</span> articles</summary>')
            if cg.empty:page.append('<p class="muted">No articles in this section.</p>')
            for _,story in cg.groupby('story_group',sort=False):
                page.append(f'<div class="story"><p class="muted">Coverage group · {len(story)} article(s)</p>')
                for a in story.itertuples():
                    event=events[events.event_id.eq(a.event_id)].iloc[0]
                    p=pd.Timestamp(a.published_at_utc);s=pd.Timestamp(event.start_utc);end=pd.Timestamp(event.end_utc)
                    distance=(s-p).total_seconds()/60 if a.phase=='before' else (p-(end if a.phase=='after' else s)).total_seconds()/60
                    timing=f'{distance:.1f} min '+('before start' if a.phase=='before' else 'after end' if a.phase=='after' else 'after start')
                    page.append(f'<article class="card" data-phase="{a.phase}"><span class="badge {a.phase}">{a.phase.title()}</span><span class="badge">{esc(a.published_ny[11:19])} ET · {timing}</span><h3><a href="{esc(a.url)}">{esc(html.unescape(a.headline))}</a></h3><p class="muted">{esc(a.source)} · Existing relevance: {esc(a.relevance_type)} · ID {a.article_id}</p><p><b>Why here:</b> {esc(a.review_reason)}</p><details><summary>Read saved summary and review limits</summary><p>{esc(html.unescape(a.summary)) or "No summary available."}</p><p>First announcement time: unverified. Connection to this exact hourly movement: not established.</p></details>')
                    if a.historical_version_warning:page.append('<p class="warning">Review timing: live coverage, closing-time wording, or other historical-version concern.</p>')
                    page.append('</article>')
                page.append('</div>')
            page.append('</details>')
        page.append('</section>')
    page.append('''<script>function filterEvent(button,phase){let event=button.closest('.event');event.querySelectorAll('button').forEach(b=>b.classList.remove('active'));button.classList.add('active');event.querySelectorAll('.card').forEach(c=>c.hidden=phase!=='all'&&c.dataset.phase!==phase);event.querySelectorAll('.story').forEach(s=>s.hidden=![...s.querySelectorAll('.card')].some(c=>!c.hidden));event.querySelectorAll('.category').forEach(c=>c.querySelector('.visible-count').textContent=[...c.querySelectorAll('.card')].filter(a=>!a.hidden).length);}</script></main></html>''')
    (HERE/'report.html').write_text('\n'.join(page))
    assert not reviewed.duplicated(['event_id','article_id']).any()
    assert counts.articles.sum()==len(reviewed)
    assert set(reviewed.event_id)==set(events.event_id)
    assert reviewed.category.isin(CATEGORIES).all()
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    meta={'events':len(events),'articles':int(reviewed.article_id.nunique()),'pairs':len(reviewed),'source_hashes':hashes,'validation':'Every event retained; one category per pair; category totals reconcile; unchanged inputs; no duplicate pairs','labels':'Provisional assistant review; not independently human-validated','all_selected_news_windows_coverage_verified':bool(events.filter(like='_coverage_verified').all().all())}
    (HERE/'metadata.json').write_text(json.dumps(meta,indent=2));print(overview.to_string());print(json.dumps({k:v for k,v in meta.items() if k!='source_hashes'},indent=2))
if __name__=='__main__':main()
