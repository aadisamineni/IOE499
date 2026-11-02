"""Render explicit review annotations for upstream rejects and a full October ledger.

No classifier is run here: JSON groups contain individual assistant assessments.
The source collection and upstream relevance labels remain unchanged.
Run classify_october.py first to refresh the direct/indirect subset.
"""
import csv
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'october_classification'
INPUT = HERE.parents[1] / 'news data pull/processed/all_classified_articles.csv'


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write(path, fields, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    full = read(INPUT)
    base = read(OUT / 'reviewed_articles.csv')
    sources = {r['evidence_key']: r for r in read(OUT / 'sources.csv')}
    annotations = {}
    for path in sorted((HERE / 'reject_batches').glob('*.json')):
        batch = json.loads(path.read_text())
        for key, value in batch.get('sources', {}).items():
            assert key not in sources, f'Duplicate evidence key {key}'
            sources[key] = dict(evidence_key=key, **value)
        for ids, content, novelty, event, reason, keys, caveat, relevance in batch['groups']:
            for i in ids:
                assert i not in annotations, f'Duplicate reject review {i}'
                assert all(k in sources for k in keys), (i, keys)
                annotations[i] = dict(content_type=content, novelty_status=novelty,
                    event_group=event, rationale=reason, evidence_keys='|'.join(keys),
                    caveat=caveat, relevance_audit=relevance)
    assert set(annotations) == set(range(1, max(annotations) + 1)), 'Gap in explicit assessments'
    fields = list(base[0]) + ['reject_review_id', 'review_subset', 'relevance_audit']
    rejects = read(HERE / 'october_reject_source.csv')
    assert len(rejects) == 4218
    assert not {r['article_id'] for r in rejects} & {r['article_id'] for r in base}
    lookup = {r['article_id']: r for r in full}
    reject_rows = []
    for raw in rejects:
        # Assert the review source still matches the original collection exactly.
        assert all(raw[k] == v for k, v in lookup[raw['article_id']].items())
        r = {k: raw.get(k, '') for k in fields}
        r.update(review_subset='upstream_reject', october_review_id='',
                 first_release_proven='false', price_outcomes_used='false', training_ready='false')
        i = int(raw['reject_review_id'])
        if i in annotations:
            r.update(annotations[i])
            keys = r['evidence_keys'].split('|') if r['evidence_keys'] else []
            r.update(review_status='assistant_reviewed',
                assessment_origin='saved_headline_and_available_summary_or_excerpt',
                external_urls=' | '.join(sources[k]['url'] for k in keys),
                evidence_times=' | '.join(sources[k]['time'] for k in keys))
        else:
            r.update(review_status='pending', novelty_status='unreviewed', assessment_origin='none')
        reject_rows.append(r)
    reviewed_rejects = [r for r in reject_rows if r['review_status'] == 'assistant_reviewed']
    write(OUT / 'reject_review_ledger.csv', fields, reject_rows)
    write(OUT / 'reviewed_rejects.csv', fields, reviewed_rejects)
    # Restore exact original upstream fields (the older monthly extract changed
    # boolean capitalization through pandas). Assessment fields remain intact.
    base = [dict(r, **{k: lookup[r['article_id']][k] for k in r if k in lookup[r['article_id']]}) for r in base]
    base = [dict(r, reject_review_id='', review_subset='upstream_direct_indirect',
                 relevance_audit='not_independently_revalidated') for r in base]
    combined = sorted(base + reject_rows, key=lambda r: (r['published_at_utc'], r['article_id']))
    assert len(combined) == 5232 == len({r['article_id'] for r in combined})
    reviewed = [r for r in combined if r['review_status'] == 'assistant_reviewed']
    by_id = {r['article_id']: r for r in reviewed}
    write(OUT / 'october_all_articles.csv', fields, combined)
    queues = {
        'all_announcement_candidates.csv': {'release_time_supported', 'candidate_timing_unresolved', 'new_detail_candidate'},
        'all_earlier_disclosure_articles.csv': {'earlier_disclosure_found'},
        'all_timestamp_and_revision_review.csv': {'timestamp_conflict', 'revision_history_unresolved'},
        'all_content_only_unverified.csv': {'not_established'},
        'all_followup_articles.csv': {'followup_reporting'},
    }
    for filename, statuses in queues.items():
        write(OUT / filename, fields, [r for r in reviewed if r['novelty_status'] in statuses])
    relevance_flags = [r for r in reviewed_rejects if r['relevance_audit'] != 'no_direct_nvda_link_established']
    write(OUT / 'relevance_reconsideration.csv', fields, relevance_flags)
    events = []
    for event in sorted({r['event_group'] for r in reviewed if r['event_group']}):
        members = [r for r in reviewed if r['event_group'] == event]
        events.append(dict(event_group=event, article_count=len(members),
            earliest_saved_article_utc=min(r['published_at_utc'] for r in members),
            article_ids='|'.join(r['article_id'] for r in members),
            review_references='|'.join(('R' + r['reject_review_id']) if r['reject_review_id'] else ('D' + r['october_review_id']) for r in members),
            evidence_keys='|'.join(sorted({k for r in members for k in r['evidence_keys'].split('|') if k})),
            first_public_disclosure_proven='false', training_ready='false'))
    write(OUT / 'all_event_groups.csv', list(events[0]), events)
    inventory = []
    for raw in full:
        a = by_id.get(raw['article_id'])
        inventory.append(dict(article_id=raw['article_id'], published_at_utc=raw['published_at_utc'],
            headline=raw['headline'], relevance_type=raw['relevance_type'], excluded=raw['excluded'],
            exclusion_reason=raw['exclusion_reason'], review_status='assistant_reviewed' if a else 'pending',
            novelty_status=a['novelty_status'] if a else 'unreviewed'))
    write(OUT / 'full_collection_inventory.csv', list(inventory[0]), inventory)
    write(OUT / 'all_sources.csv', list(next(iter(sources.values()))), list(sources.values()))
    # Separate independent-review sheet for reject records; preserve prior entries.
    independent = OUT / 'independent_reject_review.csv'
    hfields = ['reject_review_id', 'article_id', 'published_at_utc', 'headline', 'summary', 'url',
               'human_content_type', 'human_novelty_status', 'human_relevance', 'evidence_url', 'notes']
    existing = read(independent) if independent.exists() else []
    existing_ids = {r['article_id'] for r in existing}
    write(independent, hfields, existing + [{k: r.get(k, '') for k in hfields}
        for r in reviewed_rejects if r['article_id'] not in existing_ids])
    checks = dict(collection_articles=len(full), october_articles=len(combined),
        october_content_assessed=len(reviewed), october_content_assessed_percent=round(100 * len(reviewed) / len(combined), 2),
        october_pending=len(combined)-len(reviewed), direct_indirect_assessed=len(base),
        upstream_rejects_assessed=len(reviewed_rejects), upstream_rejects_pending=len(reject_rows)-len(reviewed_rejects),
        reject_review_through_id=max(annotations), reject_review_through_utc=reviewed_rejects[-1]['published_at_utc'],
        articles_with_linked_web_evidence=sum(bool(r['external_urls']) for r in reviewed),
        novelty_counts=dict(Counter(r['novelty_status'] for r in reviewed)),
        relevance_reconsideration_records=len(relevance_flags),
        full_collection_pending=len(full)-len(reviewed),
        full_october_novelty_verification_complete=False, full_task_complete=False,
        first_release_proven_count=0, price_outcomes_used=False, input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        assessment_limit='Saved headlines and summaries/excerpts; targeted external checks, not every full article or first disclosure.',
        review_date='2026-10-26')
    (OUT / 'october_review_progress.json').write_text(json.dumps(checks, indent=2)+'\n')
    e = html.escape
    cards = []
    for r in reviewed_rejects:
        links = ''.join(f'<li><a href="{e(sources[k]["url"], quote=True)}">{e(k)}</a> — {e(sources[k]["time"])}: {e(sources[k]["note"])}</li>' for k in r['evidence_keys'].split('|') if k)
        cards.append(f'<article><h2>R{r["reject_review_id"]}. {e(r["headline"])}</h2>'
            f'<p>{e(r["published_at_utc"])} · {e(r["content_type"])} · <b>{e(r["novelty_status"])}</b></p>'
            f'<p>Relevance audit: {e(r["relevance_audit"])}</p>'
            f'<p><b>Saved summary:</b> {e(r["summary"]) or "[empty]"}</p>'
            f'<p>{e(r["rationale"])}</p><ul>{links}</ul><p>{e(r["caveat"])}</p></article>')
    (OUT / 'reject_report.html').write_text('<!doctype html><meta charset="utf-8"><title>October upstream-reject audit</title>'
        '<style>body{max-width:1050px;margin:40px auto;padding:0 20px;font:16px/1.6 system-ui}article{border-top:1px solid #ccd6df;padding:20px 0}h2{font-size:20px}</style>'
        '<h1>October upstream-reject audit</h1><p>Provisional saved-text assessments with targeted web checks. '
        'No direct NVDA link established means unconfirmed relevance, not a certified irrelevant/negative label. '
        'Source checks may only corroborate an event, not establish earlier availability. No returns used; no record is training-ready.</p>'
        '<pre>'+e(json.dumps(checks, indent=2))+'</pre>'+''.join(cards))
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
