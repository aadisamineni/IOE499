"""Export provisional assistant content review, not independent human labels."""
from pathlib import Path
import html,json,hashlib
import pandas as pd
OUT=Path(__file__).resolve().parent
SOURCE=OUT.parent/'event_article_matches.csv'
# D concrete development; P existing price movement (any company); N explicitly
# NVDA movement; O opinion/analysis; U insufficient/ambiguous content.
# These are manually authored assistant assessments of the supplied text.
ANNOTATIONS='''140117551|D|Reports server pricing and export restrictions, not NVDA share movement.
140117545|D|Reports controller, robotaxi and truck announcements; exact announcement times absent.
140117538|PO|Reports AMD's historical gains while comparing its business with Nvidia.
140119024|D|Reports NSA testing of an AI model; timing of testing not specified.
140119023|U|Vague headline and summary do not identify what Zuckerberg said.
140122180|D|Reports Alphabet quarterly performance; not an explicit share-price move.
140118988|DPN|Reports Nvidia decline and earnings-related competition; no first-release time for earnings.
139876211|D|Reports an innovation award; Nvidia is incidental context, relevance merits review.
139882104|DU|Reports a trading-service purchase alert; headline only, underlying significance uncertain.
139876210|DP|Reports Oklo's gain and a partnership announced the previous day.
139877452|D|Reports NIO in-house chip development and CEO statements.
139877484|D|Reports Amazon/Meta chip partnership and healthcare program; release times unknown.
139879331|D|Reports Google investment plans; similar story already appears earlier in this sample.
139879426|DPN|Reports Nvidia rising that Friday and Intel earnings; not proof of which hour it describes.
139882100|PNU|Claims a closing NVDA milestone at a pre-close timestamp; original version/timing uncertain.
139879297|DPNO|Reports Friday Nvidia rally and interprets possible links to Intel earnings.
139737023|O|Trading recommendations; headline-only evidence.
139730556|O|Investment Committee's stock recommendations, not a concrete business announcement.
139717081|PO|Compares Benchmark Electronics gains with prominent technology stocks.
139728034|DPNO|Reports early-trading NVDA gain and a released inflation report; interprets trading range.
139727321|DP|Reports Dow/TSMC movements and TSMC sales; live page may have changed since publication.
139729864|PNO|Reports Nvidia flat over six months within a speculative valuation comparison.
139915262|DPN|Reports Nvidia decline and an OpenAI report; live page historical version uncertain.
139918643|PNO|Discusses Nvidia's market-cap milestone and related ETF performance.
139922423|PO|Compares semiconductor and software fund performance, not explicit NVDA share change.
139916533|D|Reports a Bengaluru office lease; original announcement time unknown.
139919676|D|Reports a named Nvidia model launch.
139916528|DPNO|Reports early NVDA decline and partnership restructuring while arguing for more upside.
139916527|DPN|Reports AI-chip share declines following a negative OpenAI report.
139916524|PNO|Reports record-high Nvidia price and asks whether to buy; milestone timing unspecified.
139918597|O|General growth and earnings outlook; no clearly timed new announcement in available text.
139918596|D|Dated announcement explicitly says Vultr deploys the model today.
139918595|O|Personal investment thesis and buying preference.
139612081|O|Analyzes an existing integration; no explicit new release or announcement time.
139609402|DO|Describes a concrete production announcement dated March 16, well before this April article.
139621190|PN|Explicitly reports Nvidia already rising Thursday.
139612021|DO|Interprets Nvidia's Marvell investment; first-release timing not established.
139620055|PNO|Compares Nvidia/Apple performance with defense stocks over a broader period.
139613499|O|Opinion on Amazon's custom-chip strategy and Cramer's comments.
139706393|D|Dated announcement explicitly says Firefly announced the collaboration today.
139716911|DP|Reports Firefly price gain and Nvidia partnership; does not report NVDA share movement.
139708487|PO|Reports Marvell year-to-date gains and investment outlook.
139708463|DUO|Alleges chip theft with an incomplete summary and analogy; details and timing unclear.
139708439|O|Investment recommendation focused on Aehr; Nvidia is comparison context.
139708437|U|Vague description of a quiet switch does not establish the underlying event.
139915278|D|Reports an India lease; announcement timing not established.
139915277|DPN|Reports Nvidia and peers already declining after an OpenAI growth report.
139915274|D|Dated press release explicitly announces controller launch today.
139915271|D|Dated press release explicitly announces image-generation product today; Nvidia comparison only.
139915270|PNO|Explicitly reports premarket NVDA decline and previous close; discusses pullback.
139918183|O|Buy recommendation based on diversification and networking sales.
139774868|D|Reports a quantum-lab milestone; truncated text does not establish original breakthrough date.
139774867|O|Historical hypothetical investment illustration, not an explicit observed price-move report.
139774866|D|Explicitly announces webhook infrastructure launch today.
139775634|D|Reports extended Meta/Broadcom deal and deployment commitment.
139776882|D|Reports executives' statements on collaboration; summary is a URL only.
139774862|D|Reports cloud deal and investment said to be announced Wednesday.
139779709|DPO|Reports Broadcom rally, partnership and analyst optimism, not NVDA movement.
139774861|O|Technical setup predicting a potential Broadcom breakout, not an explicit completed move.
139779879|DP|Reports Infleqtion surge and Nvidia model launch, not NVDA share movement.
139776805|O|Buy/sell ratings and portfolio analysis.
139775661|D|Reports European Commission antitrust action described as occurring Wednesday.
139774859|DP|Reports quantum-stock gains and Nvidia AI development, not explicit NVDA share movement.
139790584|D|Reports expanded Cadence/Nvidia engineering collaboration.
139774858|PNO|Discusses yesterday's Nvidia gain and valuation; not necessarily this hour's move.
139775563|DPU|Reports quantum-stock rally but available text lacks details of the AI-model development.
139775561|PN|Reports NVDA winning streak; 10-day versus later 11-day language warrants timing review.
139775558|PNO|Reports NVDA up in morning trading and discusses breakout prospects.
139775553|PN|Reports NVDA winning streak in a live market page; revisions possible.
139876225|D|Reports Nvidia teams testing a new model; first availability time unspecified.
139878504|D|Reports Google/Anthropic investment plans earlier than another same-day story.
139876219|O|Technical setup and upcoming earnings catalysts; no released results reported.
139876214|PNO|Reports previous close and morning price level while discussing relative performance.
139876213|D|Reports NIO chip strategy and CEO interview said to be on Friday.
139751497|O|Analysis of Nvidia's competitive position against Google TPUs.
139751490|DO|Interprets a recently completed Marvell investment; not a verified new announcement.
139751488|O|Describes an existing Planet Labs partnership and business strategy.
139765604|DP|Reports Dell gain after an acquisition rumor; rumor is not confirmed transaction.
139752938|DPNO|Reports last week's NVDA gain and Anthropic model restriction within analyst commentary.
139752906|PNO|Reports NVDA fluctuations and earlier gains within a buy/sell discussion.'''

def main():
    source_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    pairs=pd.read_csv(SOURCE,keep_default_na=False)
    pairs=pairs[pairs['rank'].le(10)&(pairs.before_1h|pairs.during|pairs.after_1h)].copy()
    notes={}
    for line in ANNOTATIONS.splitlines():
        key,codes,note=line.split('|',2)
        assert key not in notes
        notes[key]=(codes,note)
    ids=set(pairs.article_id.astype(str));assert ids==set(notes),(ids-set(notes),set(notes)-ids)
    rows=[]
    for _,a in pairs.drop_duplicates('article_id').iterrows():
        codes,note=notes[str(a.article_id)]
        row={k:a[k] for k in ['article_id','headline','summary','source','url','published_at_utc','published_ny','relevance_type']}
        row.update(concrete_development_reported='D' in codes,existing_price_movement_any_company='P' in codes,existing_nvda_price_movement='N' in codes,opinion_or_analysis='O' in codes,insufficient_or_ambiguous_content='U' in codes,development_first_release_time='not_verified',supporting_headline=html.unescape(a.headline),supporting_summary=html.unescape(a.summary),assessment_note=note,reviewer='assistant_provisional',historical_version_verified=False)
        if str(a.article_id) in ['139876210','139609402','139751490']:row['development_first_release_time']='text_explicitly_indicates_earlier_development'
        row['timestamp_or_revision_warning']=('Live Coverage' in a.headline or str(a.article_id) in ['139882100','139775561'])
        rows.append(row)
    review=pd.DataFrame(rows);review.to_csv(OUT/'provisional_article_review.csv',index=False)
    flags=['concrete_development_reported','existing_price_movement_any_company','existing_nvda_price_movement','opinion_or_analysis','insufficient_or_ambiguous_content']
    joined=pairs.merge(review[['article_id']+flags+['timestamp_or_revision_warning']],on='article_id',validate='many_to_one')
    joined.to_csv(OUT/'reviewed_event_article_pairs.csv',index=False)
    totals=[]
    for group,col in [('before','before_1h'),('during','during'),('after','after_1h')]:
        subset=joined[joined[col]].drop_duplicates('article_id')
        r={'group':group,'unique_articles':len(subset)}
        for flag in flags:r[flag]=int(subset[flag].sum())
        r['both_development_and_nvda_movement']=int((subset.concrete_development_reported&subset.existing_nvda_price_movement).sum())
        r['timestamp_or_revision_warnings']=int(subset.timestamp_or_revision_warning.sum())
        totals.append(r)
    summary=pd.DataFrame(totals);summary.to_csv(OUT/'group_summary.csv',index=False)
    # Blind sheets omit assistant labels, event returns, and phase to reduce anchoring.
    blind=review[['article_id','headline','summary','source','url','published_at_utc']].sample(frac=1,random_state=499)
    for reviewer in ['ZAS','AVS']:
        sheet=blind.copy()
        for col in flags+['supporting_excerpt','timing_concern','reviewer_notes']:sheet[col]=''
        destination=OUT/f'independent_review_{reviewer}.csv'
        if not destination.exists():sheet.to_csv(destination,index=False)
    report=['<html><meta charset="utf-8"><title>April article review</title><body><h1>Provisional April article review</h1><p>80 unique articles, 92 event–article pairs. Assistant assessments of saved headlines/summaries only; not independently validated human labels. Concrete development does not establish novelty at publication. Categories overlap; false means not evident in available text. Articles may appear in multiple timing groups. No causal claims.</p>',summary.to_html(index=False)]
    for r in review.to_dict('records'):
        report.append('<hr><h2>'+html.escape(r['headline'])+'</h2><p>'+html.escape(str(r['article_id'])+' | '+r['published_ny'])+'</p><p>'+html.escape(r['supporting_summary'])+'</p><p><b>'+', '.join(k for k in flags if r[k])+'</b></p><p>'+html.escape(r['assessment_note'])+'</p><a href="'+html.escape(r['url'],quote=True)+'">Source link</a>')
    (OUT/'report.html').write_text('\n'.join(report)+'</body></html>')
    assert len(review)==80 and len(joined)==92
    assert not review.article_id.duplicated().any()
    assert (review.existing_nvda_price_movement<=review.existing_price_movement_any_company).all()
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==source_hash
    (OUT/'metadata.json').write_text(json.dumps({'input_sha256':source_hash,'unique_articles':len(review),'pairs':len(joined),'label_origin':'assistant provisional content review, not human consensus','validation':'Every selected article reviewed exactly once; source unchanged; many-to-one link integrity; NVDA movement implies any-company movement','group_counts':totals},indent=2))
    print(summary.to_string(index=False))
if __name__=='__main__':main()
