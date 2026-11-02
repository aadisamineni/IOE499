"""Render the assistant's explicit, provisional assessments; no model/API calls.

Run from repository root: .venv/bin/python research/news_novelty_review/build_report.py
Sample selection is documented in sample_metadata.json. Returns are never loaded.
"""
import csv
import html
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

# External evidence is paraphrased; dates without offsets are deliberately not
# converted to precise UTC availability times.
SOURCES = {
    'huang': ('https://www.benzinga.com/markets/tech/25/10/48116340/nvidia-ceo-jensen-huang-says-hes-surprised-amd-would-give-away-10-of-the-company-in-its-deal-with-openai-its-clever-i-guess', '2025-10-08; displayed 10:14 PM, timezone not verified', 'Earlier coverage of Huang reacting to the AMD deal.'),
    'pegatron': ('https://svr.pegatroncorp.com/news/301', '2025-10-13; date only', 'Company release matches the product showcase.'),
    'qualcomm': ('https://www.qualcomm.com/news/releases/2025/10/qualcomm-unveils-ai200-and-ai250-redefining-rack-scale-data-cent', 'Displayed 2025-10-28; conflicts with October 27 reporting', 'Company page confirms chips; displayed date cannot establish first release.'),
    'blackwell': ('https://www.marketscreener.com/news/trump-says-nvidia-s-high-end-blackwell-chip-did-not-figure-in-talks-with-china-s-xi-ce7d5dd3d981f526', '2025-10-30 01:04 EDT; displayed publication', 'Reuters coverage predates sampled afternoon UTC article.'),
    'amazon': ('https://press.aboutamazon.com/2025/12/trainium3-ultraservers-now-available-enabling-customers-to-train-and-deploy-ai-models-faster-at-lower-cost', '2025-12-02; date only', 'Company release confirms general availability.'),
    'tesla': ('https://ceomorningbrief.theedgemalaysia.com/pdf/2025/20251216_CMB_1074/251216l7zz09.pdf', 'Reuters item dated 2025-12-15 in December 16 compilation', 'Earlier reporting of board compensation; not an NVDA development.'),
    'supermicro': ('https://ir.supermicro.com/news/news-details/2026/Supermicro-Announces-Support-for-Upcoming-NVIDIA-Vera-Rubin-NVL72-HGX-Rubin-NVL8-and-Expanded-Rack-Scale-Manufacturing-Capacity-for-Liquid-Cooled-AI-Solutions/default.aspx', '2026-01-05; date only', 'Company announced capacity expansion before sampled January 6 article.'),
    'coreweave': ('https://coreweave.com/news/coreweave-extends-its-cloud-platform-with-nvidia-rubin-platform', '2026-01-05; date only', 'Company announced Rubin support before sampled January 6 article.'),
    'rubin': ('https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer', '2026-01-05; date only', 'NVIDIA launch preceded January 6 CES roundup.'),
    'h200': ('https://www.ndtvprofit.com/technology/china-tells-alibaba-tech-firms-to-ready-nvidia-h200-orders-10866337', '2026-01-23 21:45 IST = 16:15 UTC; displayed publication', 'Bloomberg-syndicated report precedes sample by about 68 minutes.'),
    'maia': ('https://news.microsoft.com/source/emea/2026/01/microsoft-introduces-maia-200-new-inference-accelerator-enhances-ai-performance-in-azure/', '2026-01-26; date only', 'Company confirms chip and SDK announcement.'),
    'pc': ('https://www.digitaltrends.com/computing/nvidia-could-launch-its-first-laptops-with-its-own-processors-later-this-year/', '2026-02-22; date only', 'Earlier report describes Dell/Lenovo laptop plans.'),
    'taiwan': ('https://www.tomshardware.com/tech-industry/semiconductors/us-govt-warned-nvidia-ceo-jensen-huang-tim-cook-and-lisa-su-that-china-could-invade-taiwan-by-2027-apple-ceo-reportedly-said-he-sleeps-with-one-eye-open', '2026-02-24 16:15:41 UTC; search-displayed publication', 'Earlier coverage of investigation into a 2023 briefing.'),
    'earnings': ('https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-fourth-quarter-and-fiscal-2026', '2026-02-25; call scheduled 17:00 EST = 22:00 UTC', 'Sample CFO headlines are timestamped several hours before the scheduled call.'),
    'funding': ('https://openai.com/index/scaling-ai-for-everyone/', '2026-02-27; date only', 'Company announcement confirms funding amounts; minute ordering unresolved here.'),
    'optics': ('https://www.investing.com/news/stock-market-news/nvidia-to-invest-4-billion-in-photonic-product-makers-lumentum-and-coherent-4534799', '2026-03-02; displayed 08:19 AM, timezone not verified', 'Earlier same-day Reuters coverage; exact UTC publication not verified.'),
    'reflection': ('https://www.techmeme.com/260303/h0125', '2026-03-03 01:25 ET archive snapshot', 'Archived page already links matching Financial Times funding report.'),
    'export': ('https://www.streetinsider.com/Investing/Nvidia%2Bsinks%2Bon%2Breport%2Bof%2Bglobal%2BAI%2Bchip%2Bexport%2Brules/26117575.html', '2026-03-05 12:42 EST = 17:42 UTC; displayed syndication time', 'Matching syndicated story precedes saved 18:09 UTC timestamp.'),
    'hp': ('https://www.hp.com/us-en/newsroom/press-releases/2026/hp-pc-local-compute-ai-workloads.html', '2026-03-24; date only', 'Company product release matches sample; exact first-release time unresolved.'),
    'micron': ('https://investors.micron.com/news/press-release/2026/Micron-in-High-Volume-Production-of-HBM4-Designed-for-NVIDIA-Vera-Rubin-PCIe-Gen6-SSD-and-SOCAMM2-03-16-2026/default.aspx', '2026-03-16; date only', 'Company announcement predates sampled March 27 article by eleven days.'),
    'marvell': ('https://nvidianews.nvidia.com/news/nvidia-ai-ecosystem-expands-as-marvell-joins-forces-through-nvlink-fusion', '2026-03-31; date only', 'Company announcement predates sampled April 1 article.'),
}

# id, assessment, confidence in assessment (NOT probability of price response),
# rationale, source key, timing warning. Candidate never means verified first.
ASSESSMENTS = [
 (1,'opinion_background','medium','Fund-positioning/stock-selection headline; summary repeats headline, preventing detailed novelty judgment.','',''),
 (2,'recap','high','Reaction to AMD deal was covered the preceding day; no additional fact identified in saved summary.','huang',''),
 (3,'announcement_candidate','medium','Specific product showcase matches company release; same-day chronology and earlier product disclosures unresolved.','pegatron','Exact first availability unverified'),
 (4,'reactive_reporting','high','Live market movements dominate headline; policy comment could also be new but its timing is not supplied.','','Live coverage may contain later edits'),
 (5,'announcement_candidate','medium','Specific competing-chip launch and customer commitment; source confirms development but dates conflict.','qualcomm','Company page displays October 28 while sample is October 27; do not infer availability'),
 (6,'opinion_background','high','Personal portfolio allocation advice rather than a new Nvidia business development.','',''),
 (7,'recap','high','Same outcome of Trump-Xi talks was publicly reported hours earlier.','blackwell',''),
 (8,'opinion_background','high','Valuation thesis built around an already observed market-cap milestone.','',''),
 (9,'reactive_reporting','high','Explains an observed Entegris decline and broader selling; Nvidia appears as contextual attribution.','',''),
 (10,'opinion_background','medium','Analyst confidence and earnings anticipation; possible fresh analyst action cannot be dated from summary.','',''),
 (11,'reactive_reporting','high','Reports Lilly market capitalization; Nvidia is a comparison, not the event subject.','','Weak substantive NVDA relevance'),
 (12,'reactive_reporting','high','Explicitly discusses falling shares and results released last week.','',''),
 (13,'reactive_reporting','high','Market roundup describes already observed declines.','',''),
 (14,'announcement_candidate','medium','Specific Trainium3 availability development matches same-day company release; first minute unresolved.','amazon','Same-day confirmation is not minute-level novelty verification'),
 (15,'opinion_background','high','Competing bullish/bearish investment opinions rather than a concrete company announcement.','',''),
 (16,'recap','medium','Board-compensation analysis concerns Tesla and had earlier December 15 coverage.','tesla','NVDA is a comparison rather than event subject'),
 (17,'recap','high','Capacity announcement was released by Supermicro January 5.','supermicro',''),
 (18,'recap','high','CoreWeave announced Rubin platform support January 5.','coreweave',''),
 (19,'recap','medium','CES roundup includes Nvidia launch already announced January 5; other-company details not exhaustively checked.','rubin','Truncated saved summary'),
 (20,'recap','high','Saved summary explicitly dates the analyst action December 19, well before January 8 publication.','','Underlying analyst action date from saved summary; not independently confirmed'),
 (21,'reactive_reporting','high','Movers roundup with credit-card-sector decline; no specific new NVDA fact in saved summary.','',''),
 (22,'recap','high','Matching Bloomberg report displayed at 16:15 UTC, before sampled 17:22 UTC publication.','h200','Earliest located source, not proven first report'),
 (23,'announcement_candidate','medium','Specific Microsoft chip/software launch matches dated company announcement; precise ordering unverified.','maia','Exact first availability unverified'),
 (24,'reactive_reporting','high','Live index and stock movement reporting; no specific new NVDA development in saved text.','','Live coverage may contain later edits'),
 (25,'opinion_background','high','Trading-range interpretation mixes observed prices with supplier revenue news; fresh detail timing unresolved.','','Mixed price commentary and supplier development'),
 (26,'opinion_background','high','Interprets valuation and past stock performance against aggregate capex expectations.','',''),
 (27,'opinion_background','high','Interprets a research scenario as bullish for Nvidia; original report time not verified.','','New commentary is not necessarily a new business fact'),
 (28,'recap','medium','Same Dell/Lenovo laptop plans appeared in earlier February 22 coverage; no incremental detail identified.','pc','Full sampled text unavailable'),
 (29,'opinion_background','high','Earnings preview and expectations rather than reported results.','',''),
 (30,'recap','medium','Earlier same-day reporting describes the investigation; historical briefing date is distinct from public disclosure date.','taiwan','Do not equate the 2023 briefing with first public disclosure'),
 (31,'unclear','high','Specific CFO/result details require publication-history verification; saved time is before scheduled earnings call.','earnings','17:33:58 UTC sample versus 22:00 UTC scheduled call; quarantine pending provenance'),
 (32,'unclear','high','Explicit conference-call attribution conflicts with saved pre-call timestamp.','earnings','18:15 UTC sample versus 22:00 UTC scheduled call; quarantine pending provenance'),
 (33,'unclear','low','Vague interpretive headline and empty summary cannot establish content or novelty.','','No summary or full text'),
 (34,'announcement_candidate','medium','Funding amounts match same-day company announcement; sampled item may be a later recap.','funding','Minute ordering unresolved; do not use as verified first-release example'),
 (35,'reactive_reporting','high','Live market coverage reports observed declines and references existing funding/earnings stories.','','Live coverage may contain later edits'),
 (36,'recap','medium','Midday roundup covers optics investment already in earlier same-day reporting.','optics','Earlier source timezone not verified'),
 (37,'reactive_reporting','high','Reports gains and combines several announcements; same investment is already covered by sample 36.','optics','Local sample 36 already reports development at 16:32 UTC'),
 (38,'recap','medium','Matching funding-seeking report was linked in an early-morning archive before sampled afternoon item.','reflection','Archive evidence, not direct Financial Times publication timestamp'),
 (39,'reactive_reporting','high','Explicit observed price declines attributed to Bloomberg report; matching syndication predates saved time.','export','Saved timestamp differs from displayed syndication time'),
 (40,'opinion_background','high','Predicts a possible rally around an upcoming event.','','Headline-only assessment'),
 (41,'opinion_background','high','Pre-event analyst expectations rather than a completed new announcement.','',''),
 (42,'opinion_background','high','Investment interpretation and retrospective discussion of a conference.','',''),
 (43,'announcement_candidate','medium','Specific workstation product release matches company announcement; intraday ordering unresolved.','hp','Date-only company evidence cannot verify 13:30 UTC first availability'),
 (44,'opinion_background','high','Investment decision framework and geopolitical interpretation.','',''),
 (45,'recap','high','HBM4 production was announced March 16; sampled March 27 headline adds investment framing.','micron',''),
 (46,'recap','high','Marvell investment and partnership were announced March 31, before April 1 item.','marvell',''),
 (47,'reactive_reporting','high','Describes an already observed winning streak with demand narrative.','',''),
 (48,'reactive_reporting','high','Live market coverage and winning-streak reporting.','','Live coverage may contain later edits'),
 (49,'reactive_reporting','medium','Mixes Nvidia market-cap movement with potentially new Alphabet investment reporting; no new NVDA business fact established.','','Other-company news novelty not verified'),
 (50,'opinion_background','high','Retrospective platform search-popularity statistics, not an NVDA business announcement.','','Could be new audience data; not the target business-event type'),
]

def main():
    with (HERE/'sample.csv').open(newline='') as f:
        sample = list(csv.DictReader(f))
    assert len(sample) == len(ASSESSMENTS) == 50
    by_id = {int(r['review_id']): r for r in sample}
    rows = []
    for i, label, confidence, reason, key, warning in ASSESSMENTS:
        source = SOURCES.get(key, ('','',''))
        r = dict(by_id[i])
        r.update(assessment=label, assessment_confidence=confidence,
                 rationale=reason, external_evidence_url=source[0],
                 earliest_located_evidence_time=source[1],
                 external_evidence_note=source[2], timing_warning=warning,
                 verification_scope='targeted_external_check' if key else 'saved_headline_summary_only',
                 verified_first_release='not_established',
                 label_origin='assistant_provisional', review_date='2026-10-26')
        rows.append(r)
    with (HERE/'review.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    counts=dict(Counter(r['assessment'] for r in rows))
    summary={'reviewed':50,'assessment_counts':counts,
             'articles_with_external_evidence':sum(bool(r['external_evidence_url']) for r in rows),
             'confirmed_first_releases':0,
             'returns_examined':False,'human_validated':False}
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Blank independent sheet intentionally omits assistant judgments and outcomes.
    independent_fields=['review_id','article_id','published_at_utc','headline','summary','source','url',
                        'human_assessment','earliest_source_url','earliest_source_time','supporting_evidence','notes']
    independent=HERE/'independent_review.csv'
    if not independent.exists():
        with independent.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=independent_fields);writer.writeheader()
            writer.writerows({k:r.get(k,'') for k in independent_fields} for r in rows)
    e=html.escape
    blocks=[]
    for r in rows:
        evidence=(f'<a href="{e(r["external_evidence_url"],quote=True)}">External evidence</a>'
                  f' — {e(r["earliest_located_evidence_time"])}<p>{e(r["external_evidence_note"])}</p>'
                  if r['external_evidence_url'] else 'Saved headline/summary only; earlier-source history not verified.')
        blocks.append(f'<article><h2>{r["review_id"]}. {e(r["headline"])}</h2>'
                      f'<p>{e(r["published_at_utc"])} · {e(r["assessment"])} · confidence: {e(r["assessment_confidence"])}</p>'
                      f'<p><b>Saved summary:</b> {e(r["summary"]) or "[empty]"}</p>'
                      f'<p><b>Assessment:</b> {e(r["rationale"])}</p><p>{evidence}</p>'
                      f'<p><b>Timing caveat:</b> {e(r["timing_warning"]) or "No specific conflict identified; original availability remains unverified."}</p>'
                      '</article>')
    (HERE/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>News novelty review</title>'
        '<style>body{max-width:1050px;margin:40px auto;padding:0 20px;font:16px/1.6 system-ui;color:#192532}article{border-top:1px solid #ccd6df;padding:20px 0}h2{font-size:20px}a{color:#155eaa}</style>'
        '<h1>News novelty review — 50 sampled articles</h1>'
        '<p>Provisional assistant review. Random seed 499; direct regular-session representative articles only. '
        'No subsequent returns examined. Targeted external checks cover selected developments; other judgments use saved headlines/summaries. '
        'Announcement candidates are not verified first releases. Do not use these labels as validated ground truth.</p>'
        '<pre>'+e(json.dumps(summary,indent=2))+'</pre>'+''.join(blocks))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
