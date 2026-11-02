"""Materialize a resumable, evidence-backed review ledger.

Only explicit assistant-reviewed records receive labels. All other records stay
pending. No API inference, return data, automatic novelty labels, or source edits.
"""
from pathlib import Path
import csv
import hashlib
import html
import json
from collections import Counter

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
INPUT=ROOT/'news data pull/processed/all_classified_articles.csv'
OUT=HERE/'october_classification'

# Source timestamps describe the evidence actually found, never an inferred
# guarantee of earliest publication. Source observations are paraphrases.
SOURCES={
 'cohere':dict(url='https://newsroom.amd.com/news/amd-and-cohere-expand-global-ai-collaboration-to-p/',time='2025-09-24',precision='date_only',note='AMD expansion release predates October coverage; page also displays a September 23/24 date discrepancy.'),
 'jefferies':dict(url='https://www2.stockmarketwatch.com/stock-market-news/market-snapshot-bps-gulf-investment-nvidias-price-target-hike-and-european-economic-data/53784/',time='2025-09-29',precision='date_only_secondary',note='Earlier coverage of the same target increase; sample says September 27, so exact action date remains disputed.'),
 'aipc':dict(url='https://www.globenewswire.com/news-release/2025/10/01/3159281/28124/en/AI-PC-Research-Report-2025-2035-Market-to-Reach-992-Billion-by-2035-at-a-CAGR-of-32-16-Driven-by-NVIDIA-Apple-AMD-Lenovo-Microsoft-Dell-HP-Fujitsu-ASUSTeK-Intel-and-Huawei.html',time='2025-10-01T09:16:00Z',precision='minute_company_wire',note='Research publisher release matches saved time; report inception/earlier availability unverified.'),
 'intel':dict(url='https://investor.nvidia.com/news/press-release-details/2025/NVIDIA-and-Intel-to-Develop-AI-Infrastructure-and-Personal-Computing-Products/default.aspx',time='2025-09-18',precision='date_only',note='Company release establishes September disclosure of investment and collaboration.'),
 'openai':dict(url='https://blogs.nvidia.com/blog/openai-nvidia/',time='2025-09-22',precision='date_only',note='NVIDIA announcement precedes October discussion of infrastructure partnership.'),
 'nscale':dict(url='https://www.nscale.com/press-releases/nscale-pre-series-c-safe',time='2025-10-01T08:00:00Z',precision='minute_company_release',note='Release states 09:00 BST; saved article is 2h38m later.'),
 'zyphra':dict(url='https://newsroom.ibm.com/2025-10-01-ibm-and-amd-collaborate-with-zyphra-on-next-generation-ai-infrastructure',time='2025-10-01',precision='date_only',note='Company confirms collaboration; exact release minute unverified. Initial deployment date is not disclosure date.'),
 'phaidra':dict(url='https://www.prnewswire.com/news-releases/collaborative-fund-leads-phaidras-50m-series-b-to-build-the-ai-factories-of-the-future-302572418.html',time='2025-10-01T13:00:00Z',precision='minute_company_wire',note='Investor-issued funding release matches sampled publication minute.'),
 'rivos':dict(url='https://www.streetinsider.com/Reuters/Meta%2Bto%2Bbuy%2Bchip%2Bstartup%2BRivos%2Bfor%2BAI%2Beffort%2C%2Bsource%2Bsays/25398390.html',time='2025-09-30T17:53:00Z',precision='minute_wire_syndication',note='Reuters reporting of acquisition predates October 1 articles.'),
 'msftchips':dict(url='https://finance.yahoo.com/news/microsoft-lean-more-own-ai-203459048.html',time='2025-10-01',precision='date_only_same_article',note='Same article names Scott speech at Italian Tech Week; speech timing/first reporting not established.'),
 'coreweave_meta':dict(url='https://www.sec.gov/Archives/edgar/data/1769628/000176962825000050/crwv-20250925.htm',time='2025-09-30',precision='filing_date_only',note='September 30 filing discloses agreement signed September 25; signature date is not public availability.'),
 'shi':dict(url='https://www.shi.com/about/news/shi-releases-508-compliance-solution-with-hpe-nvidia-and-kamiwaza',time='2025-10-02',precision='date_only_corrected_release',note='Company links a correcting/replacing Business Wire release; original and revised historical text must be distinguished.'),
 'shi_correction':dict(url='https://6ix.com/news/correcting-and-replacing-shi-releases-508-compliance-solution-with-hpe-nvidia-and-kamiwaza',time='2025-10-02T15:43:00Z',precision='minute_syndicated_correction',note='Corrected release appears after both sampled SHI timestamps; changed passages not reconstructed.'),
 'nebius':dict(url='https://www.sec.gov/Archives/edgar/data/1513845/000110465925088312/tm2525580d1_6k.htm',time='2025-09-08',precision='filing_date_only',note='September filing establishes previously announced Microsoft capacity agreement.'),
 'nebius_detail':dict(url='https://www.bloomberg.com/news/articles/2025-10-02/microsoft-neocloud-deals-a-strategy-for-ai-supply-crunch',time='2025-10-02T10:00:00Z',precision='minute_original_report_metadata',note='Same report distinguishes newly sourced usage/GPU details from September agreement; full historical article version unavailable.'),
 'stargate_korea':dict(url='https://openai.com/index/samsung-and-sk-join-stargate/',time='2025-10-01',precision='date_only',note='Company announcement precedes October 2 article.'),
 'uipath':dict(url='https://www.uipath.com/newsroom/uipath-and-nvidia-collaborate-to-deliver-trusted-agentic-automation-for-sensitiv-workflows',time='2025-09-30',precision='date_only',note='Company announced NVIDIA collaboration before October 2 retrospective price article.'),
 'valuation':dict(url='https://gazette.com/2025/10/01/openai-hits-500-billion-valuation-after-share-sale-to-softbank-others-source-says/',time='2025-10-01; displayed 5:08pm, timezone unresolved',precision='date_only_secondary',note='Earlier share-sale reporting; completion can be new even if valuation was previously discussed.'),
 'uae_delays':dict(url='https://www.streetinsider.com/Reuters/Delays%2Bto%2BTrump%27s%2BUAE%2Bchips%2Bdeal%2Bfrustrate%2BNvidia%27s%2BJensen%2BHuang%2C%2Bofficials%2C%2BWSJ%2Breports/25415170.html',time='2025-10-03T00:58:00Z',precision='minute_wire_syndication',note='Reuters attributes delay reporting to WSJ. Underlying May deal date must not be mistaken for disclosure of the delay.'),
 'uae_original':dict(url='https://www.livemint.com/ai/artificial-intelligence/nvidia-uae-ai-chip-deal-jensen-huang-us-china-ai-race-nvidia-h20-chip-china-11759479413855.html',time='2025-10-03; syndicated WSJ article, original time unresolved',precision='date_only_later_syndication',note='Supports substantive delay reporting; original WSJ publication history not recovered.'),
 'fujitsu':dict(url='https://global.fujitsu/en-global/pr/news/2025/10/03-01',time='2025-10-03',precision='date_only',note='Company confirms expanded collaboration; exact public-release minute unresolved.'),
 'superx':dict(url='https://www.nasdaq.com/press-release/superx-launches-latest-xn9160-b300-ai-server-blackwell-ultra-delivers-50-more-compute',time='2025-10-03T10:35:00Z',precision='minute_company_wire_syndication',note='Company-issued release displays 06:35 EDT, matching saved time.'),
 'furiosa':dict(url='https://www.benzinga.com/news/topics/25/10/48016595/exclusive-furiosaai-ceo-tells-benzinga-nvidias-greatest-strength-is-also-its-achilles-heel-after-800m-meta-offer-targets-series-d-in-2026',time='2025-10-03; displayed 10:01 AM, timezone unverified',precision='date_only_original_interview',note='Original interview page located. Exclusive label does not prove every statement was new.'),
}

# Every entry below is an explicit assistant assessment after reading that
# record's saved headline and summary. Lists group parallel judgments only.
# Fields: ids, content, novelty, event, rationale, evidence keys, caveat.
GROUPS=[
 ([1,30],'development_reporting','earlier_disclosure_found','amd_cohere_expansion','October wording repeats September AMD-Cohere expansion; no incremental fact identified in saved text.',['cohere'],''),
 ([2,13,21,26,29,31,43,49,50,53,55,59,61,62,65,66,67,79,95,98],'reactive_reporting','not_established','', 'Main content describes already observed prices, market capitalization, or trading conditions. This does not prove absence of new details in unavailable full text.',[],''),
 ([4],'mixed_roundup','not_established','government_shutdown','Mixes shutdown and policy developments with market-cap reporting; earliest disclosure of each item not checked.',[],'Multiple events; no single valid release anchor'),
 ([3],'analyst_action','earlier_disclosure_found','jefferies_target','Target increase was covered before October 1. Saved summary and earlier coverage disagree on action date.',['jefferies'],'September 27 versus September 29 requires provenance check'),
 ([5,6,7,8,9,10,14,16,18,20,24,27,33,34,36,37,39,42,45,51,52,54,63,68,74,75,76,77,81,87,88,89,90,92,94,99,102,103,104],'opinion_background','not_established','', 'Investment thesis, recommendation, valuation, portfolio discussion or background dominates the available text. May include factual context; not certified as having no new information.',[],''),
 ([11],'research_release','release_time_supported','aipc_research','Publisher-issued research announcement matches saved minute; new report availability is distinct from an NVDA business announcement.',['aipc'],'Earlier sale/publication of underlying report unverified; peripheral NVDA relevance'),
 ([12,40],'opinion_background','earlier_disclosure_found','nvidia_intel_investment','Discusses already announced NVIDIA investment in Intel; no new transaction detail established in saved text.',['intel'],''),
 ([15],'opinion_background','earlier_disclosure_found','nvidia_openai_partnership','Investment framing around previously disclosed partnership.',['openai'],''),
 ([17],'development_reporting','earlier_disclosure_found','nscale_safe','Specific funding round, but company page states release 2h38m before saved article; no incremental detail identified.',['nscale'],''),
 ([19],'development_reporting','candidate_timing_unresolved','ibm_amd_zyphra','Specific collaboration confirmed by company announcement on same date; exact intraday first availability unresolved.',['zyphra'],'Deployment in September does not establish public disclosure in September'),
 ([22],'development_reporting','release_time_supported','phaidra_series_b','Investor-issued announcement matches saved funding-release minute.',['phaidra'],'Earliest public disclosure not guaranteed; premarket announcement'),
 ([23],'development_reporting','candidate_timing_unresolved','ibm_amd_zyphra','Follows saved company release record by about two hours; external release has date-only timing, so historical ordering is not independently established.',['zyphra'],'Likely follow-up; no unsupported precise first-release claim'),
 ([25,32],'reactive_reporting','earlier_disclosure_found','meta_rivos','Price reporting references acquisition already covered September 30.',['rivos'],'Price-reactive and factual recap can coexist'),
 ([28],'development_reporting','earlier_disclosure_found','meta_rivos','Specific reported deal matches previous-day Reuters coverage; no new deal detail identifiable in summary.',['rivos'],''),
 ([35],'mixed_opinion_price','earlier_disclosure_found','nvidia_openai_partnership','Buy-zone discussion includes prior-day analyst action and older partnership; no current new business fact established.',['openai'],'Analyst action timing not independently verified'),
 ([38],'speech_reporting','candidate_timing_unresolved','microsoft_custom_chips','Reports executive comments and custom-silicon plans; speech may add information, but speech and first-report timestamps unresolved.',['msftchips'],'Same-article match confirms identity only, not independent timing'),
 ([41],'background_recap','not_established','microsoft_nvidia_background','Saved summary cites July results and an unspecified collaboration; insufficient detail to identify an October announcement.',[],'Do not turn retrospective financial figures into a new event'),
 ([44],'mixed_opinion_price','earlier_disclosure_found','coreweave_meta','Discusses stock performance and recommendations after September disclosure of Meta contract.',['coreweave_meta'],'Contract signature and disclosure dates differ'),
 ([46,58],'development_reporting','revision_history_unresolved','shi_accessibility','Company confirms solution release, but a correcting/replacing version appears later on the same date.',['shi','shi_correction'],'Quarantine for minute-level historical-text use until original version checked'),
 ([47],'development_reporting','new_detail_candidate','microsoft_neocloud_details','Original report adds sourced GPU/usage detail to an already announced agreement; preserve as possible incremental news.',['nebius','nebius_detail'],'Cannot classify whole story as old merely because agreement was announced earlier'),
 ([48],'development_reporting','earlier_disclosure_found','stargate_korea','Korean supplier participation was announced the preceding day.',['stargate_korea'],''),
 ([56],'background_recap','not_established','ai_deal_background','General AI-deal narrative; no specific new fact identified from limited summary.',[],''),
 ([57],'peripheral_company_release','not_established','climb_award','Award concerns distributor; NVIDIA is another company on the list rather than the announcement subject.',[],'Potential relevance-filter false positive; no claim about award first publication'),
 ([60],'reactive_reporting','earlier_disclosure_found','openai_secondary_valuation','Reports chip-stock rally following previously reported share-sale valuation.',['valuation'],'A completed sale and an earlier proposed valuation are different facts'),
 ([64],'reactive_reporting','earlier_disclosure_found','uipath_nvidia','Saved text dates collaborations to late September; company release confirms September 30 NVIDIA announcement.',['uipath'],''),
 ([69],'development_reporting','candidate_timing_unresolved','uae_chip_delays','Substantive reporting about delay in an older deal can itself be new. Original WSJ publication minute was not recovered.',['uae_delays','uae_original'],'Do not label old solely because underlying deal dates to May'),
 ([70],'development_reporting','followup_reporting','uae_chip_delays','Reuters attributes underlying reporting to WSJ; a located same Reuters syndication closely matches saved time.',['uae_delays'],'Original WSJ first availability unresolved'),
 ([71],'development_reporting','candidate_timing_unresolved','fujitsu_nvidia_expansion','Specific expanded collaboration confirmed on same date; exact company disclosure time unresolved.',['fujitsu'],'Empty saved summary; identity supported by headline and company source'),
 ([72,96],'development_reporting','earlier_disclosure_found','microsoft_nebius','Agreement existed in September; GPU detail appears in October 2 original report before these October 3 recaps.',['nebius','nebius_detail'],'No additional fact identified in saved summaries'),
 ([73,82],'development_reporting','candidate_timing_unresolved','fujitsu_nvidia_expansion','Coverage of same-day expanded collaboration. Earlier local record exists, but minute-level original-release ordering is not externally established.',['fujitsu'],'Do not treat each article as an independent announcement'),
 ([78],'company_profile','not_established','moore_threads_ipo','Profile includes previous-week IPO approval. Public approval history not checked, so cannot certify novelty.',[],'Specific underlying IPO source needed if used as event'),
 ([80],'unclear','not_established','', 'Headline signals an investment discussion but empty summary prevents reliable substantive classification.',[],'Insufficient text'),
 ([83,97],'development_reporting','earlier_disclosure_found','uae_chip_delays','Later same-day reporting of delays already covered in located earlier Reuters syndication.',['uae_delays'],''),
 ([84,86,101],'mixed_price_development','earlier_disclosure_found','uae_chip_delays','Mixes observed stock movement with delay reporting already in earlier Reuters coverage; some also reference Fujitsu.',['uae_delays','fujitsu'],'Multiple developments; one anchor cannot attribute whole move'),
 ([85],'development_reporting','release_time_supported','superx_b300','Company-issued product release matches saved minute.',['superx'],'Partner product release; prior product disclosures and impact on NVDA not established'),
 ([91],'mixed_roundup','candidate_timing_unresolved','fujitsu_nvidia_expansion','Roundup includes Fujitsu development plus unrelated company items.',['fujitsu'],'Not a clean single-event document'),
 ([93],'reactive_reporting','earlier_disclosure_found','microsoft_nebius','Saved text explicitly dates agreement to September 8; company filing corroborates earlier disclosure.',['nebius'],''),
 ([100],'original_interview','candidate_timing_unresolved','furiosa_interview','Original interview may introduce plans or commentary; novelty of individual claims remains unverified.',['furiosa'],'Original journalism is not automatically a new material NVDA announcement'),
]

def main():
    for batch_path in sorted((HERE/'classification_batches').glob('*.json')):
        batch=json.loads(batch_path.read_text())
        for key,value in batch.get('sources',{}).items():
            if key in SOURCES and SOURCES[key]!=value:
                raise ValueError(f'Conflicting evidence key {key}')
            SOURCES[key]=value
        GROUPS.extend(batch['groups'])
    OUT.mkdir(exist_ok=True)
    with INPUT.open(newline='') as f:
        full=list(csv.DictReader(f))
    with (HERE/'october_source.csv').open(newline='') as f:
        month=list(csv.DictReader(f))
    annotations={}
    for ids,content,novelty,event,reason,keys,caveat in GROUPS:
        for i in ids:
            if i in annotations: raise ValueError(f'Duplicate assessment {i}')
            annotations[i]=dict(content_type=content,novelty_status=novelty,event_group=event,
                rationale=reason,evidence_keys='|'.join(keys),caveat=caveat)
    assert set(annotations)==set(range(1,max(annotations)+1)), 'Review ledger has an unexplained gap'
    fields=['october_review_id','article_id','published_at_utc','headline','summary','source','url',
            'relevance_type','excluded','exclusion_reason','event_representative','duplicate_group_id',
            'review_status','content_type','novelty_status','event_group','rationale','evidence_keys',
            'external_urls','evidence_times','caveat','assessment_origin','first_release_proven',
            'price_outcomes_used','training_ready']
    rows=[]
    for raw in month:
        i=int(raw['october_review_id'])
        r={k:raw.get(k,'') for k in fields}
        if i in annotations:
            a=annotations[i];r.update(a)
            keys=a['evidence_keys'].split('|') if a['evidence_keys'] else []
            r.update(review_status='assistant_reviewed',assessment_origin=('saved_headline_and_summary' if i<=104 else 'saved_headline_and_available_summary_or_excerpt'),
                     external_urls=' | '.join(SOURCES[k]['url'] for k in keys),
                     evidence_times=' | '.join(SOURCES[k]['time'] for k in keys))
        else:
            r.update(review_status='pending',novelty_status='unreviewed',assessment_origin='none')
        r.update(first_release_proven='false',price_outcomes_used='false',training_ready='false')
        rows.append(r)
    write_csv(OUT/'october_review_ledger.csv',fields,rows)
    reviewed=[r for r in rows if r['review_status']=='assistant_reviewed']
    write_csv(OUT/'reviewed_articles.csv',fields,reviewed)
    # Research queues, not positive/negative ML targets. Keep unresolved cases.
    queues={
        'announcement_candidates.csv': {'release_time_supported','candidate_timing_unresolved','new_detail_candidate'},
        'earlier_disclosure_articles.csv': {'earlier_disclosure_found'},
        'timestamp_and_revision_review.csv': {'timestamp_conflict','revision_history_unresolved'},
        'content_only_unverified.csv': {'not_established'},
        'followup_articles.csv': {'followup_reporting'},
    }
    for filename,statuses in queues.items():
        write_csv(OUT/filename,fields,[r for r in reviewed if r['novelty_status'] in statuses])
    event_rows=[]
    for event in sorted({r['event_group'] for r in reviewed if r['event_group']}):
        members=[r for r in reviewed if r['event_group']==event]
        event_rows.append(dict(event_group=event,article_count=len(members),
            earliest_saved_article_utc=min(r['published_at_utc'] for r in members),
            review_ids='|'.join(r['october_review_id'] for r in members),
            evidence_keys='|'.join(sorted({k for r in members for k in r['evidence_keys'].split('|') if k})),
            first_public_disclosure_proven='false',training_ready='false'))
    write_csv(OUT/'event_groups.csv',list(event_rows[0]),event_rows)
    # Preserve the full collection, including upstream exclusions and relevance
    # rejects, as a transparent backlog. No keyword screen is treated as review.
    reviewed_by_id={r['article_id']:r for r in reviewed}
    inventory=[]
    for raw in full:
        a=reviewed_by_id.get(raw['article_id'])
        inventory.append(dict(article_id=raw['article_id'],published_at_utc=raw['published_at_utc'],headline=raw['headline'],
            relevance_type=raw['relevance_type'],excluded=raw['excluded'],exclusion_reason=raw['exclusion_reason'],
            review_status='assistant_reviewed' if a else 'pending',
            novelty_status=a['novelty_status'] if a else 'unreviewed'))
    write_csv(OUT/'full_collection_inventory.csv',list(inventory[0]),inventory)
    source_rows=[dict(evidence_key=k,**v,checked_on='2026-10-26') for k,v in SOURCES.items()]
    write_csv(OUT/'sources.csv',list(source_rows[0]),source_rows)
    checks=dict(collection_articles=len(full),october_articles=sum(r['published_at_utc'].startswith('2025-10') for r in full),
        october_upstream_direct_indirect=len(rows),reviewed=len(reviewed),
        remaining_october_direct_indirect=len(rows)-len(reviewed),
        reviewed_with_external_evidence=sum(bool(r['external_urls']) for r in reviewed),
        novelty_counts=dict(Counter(r['novelty_status'] for r in reviewed)),
        content_counts=dict(Counter(r['content_type'] for r in reviewed)),
        completed_scope=f'October direct/indirect review IDs 1–{max(annotations)}, including excluded and nonrepresentative rows',
        october_direct_indirect_provisional_pass_complete=len(reviewed)==len(rows),
        full_task_complete=False,input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        source_branch='origin/AVS c99cb7a',price_outcomes_used=False,review_date='2026-10-26')
    (OUT/'progress.json').write_text(json.dumps(checks,indent=2)+'\n')
    independent=OUT/'independent_review.csv'
    hfields=['october_review_id','article_id','published_at_utc','headline','summary','url','human_content_type','human_novelty_status','evidence_url','notes']
    existing_human=[]
    if independent.exists():
        with independent.open(newline='') as f: existing_human=list(csv.DictReader(f))
    existing_ids={r['article_id'] for r in existing_human}
    write_csv(independent,hfields,existing_human+[{k:r.get(k,'') for k in hfields} for r in reviewed if r['article_id'] not in existing_ids])
    e=html.escape
    cards=[]
    for r in reviewed:
        links=''.join(f'<li><a href="{e(SOURCES[k]["url"],quote=True)}">{e(k)}</a>: {e(SOURCES[k]["time"])} — {e(SOURCES[k]["note"])}</li>' for k in r['evidence_keys'].split('|') if k)
        cards.append(f'<article><h2>{r["october_review_id"]}. {e(r["headline"])}</h2>'
            f'<p>{e(r["published_at_utc"])} · {e(r["content_type"])} · <b>{e(r["novelty_status"])}</b></p>'
            f'<p><b>Saved summary:</b> {e(r["summary"]) or "[empty]"}</p>'
            f'<p><b>Assessment:</b> {e(r["rationale"])}</p><ul>{links}</ul>'
            f'<p><b>Caveat:</b> {e(r["caveat"]) or "Historical versions and absolute first public disclosure not established."}</p>'
            f'<p>Upstream excluded: {e(r["excluded"])}; {e(r["exclusion_reason"])}</p></article>')
    (OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>October news classification</title>'
        '<style>body{max-width:1050px;margin:40px auto;padding:0 20px;font:16px/1.6 system-ui}article{border-top:1px solid #ccd6df;padding:20px 0}h2{font-size:20px}</style>'
        '<h1>October news classification</h1>'
        f'<p>{len(reviewed)} provisional assistant reviews of upstream direct/indirect records; {len(rows)-len(reviewed)} pending in that subset. Other October relevance rejects remain unreviewed. Later batches use saved summaries or excerpts, not independently retrieved full articles. No subsequent returns used. '
        'Evidence supports bounded publication-history judgments, not guaranteed first release. Labels are provisional and not training-ready.</p>'
        '<pre>'+e(json.dumps(checks,indent=2))+'</pre>'+''.join(cards))
    print(json.dumps(checks,indent=2))

def write_csv(path,fields,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

if __name__=='__main__':main()
