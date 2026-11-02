"""Export explicit content assessments. This file does not classify articles."""
from pathlib import Path
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime

HERE = Path(__file__).resolve().parent
INPUT = HERE.parents[2]/'news data pull/processed/all_classified_articles.csv'
LABELS = {'new_information_candidate', 'old_news_recap', 'price_reaction',
          'opinion_background', 'mixed', 'insufficient_information'}


def timestamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def write_csv(path, fields, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    original_hash = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    with INPUT.open(newline='') as handle:
        raw = sorted(csv.DictReader(handle), key=lambda r: (timestamp(r['published_at_utc']), r['article_id']))
    source = {r['article_id']: r for r in raw}
    assert len(source) == len(raw)
    annotations = {}
    for path in sorted((HERE/'assessments').glob('*.json')):
        for item in json.loads(path.read_text())['assessments']:
            aid = item['article_id']
            assert aid in source and aid not in annotations, f'Duplicate/unknown article {aid}'
            assert item['label'] in LABELS
            current = source[aid]
            quote = item['current_supporting_passage']
            assert quote and any(quote in current[field] for field in ['headline', 'summary']), aid
            assert item['explanation']
            assert bool(item['new_information']) == (item['label'] in {'new_information_candidate', 'mixed'}), aid
            annotations[aid] = item
    classified, evidence_rows, inventory = [], [], []
    for row in raw:
        aid = row['article_id']
        item = annotations.get(aid)
        inventory.append(dict(article_id=aid, published_at_utc=row['published_at_utc'],
            review_status='content_assessed' if item else 'pending', label=item['label'] if item else 'unreviewed'))
        if item is None:
            continue
        references = []
        for evidence in item['earlier_evidence']:
            prior = source[evidence['article_id']]
            assert timestamp(prior['published_at_utc']) < timestamp(row['published_at_utc']), (aid, prior['article_id'])
            quote = evidence['supporting_passage']
            assert quote and any(quote in prior[k] for k in ['headline', 'summary']), (aid, quote)
            references.append(prior['article_id'])
            evidence_rows.append(dict(article_id=aid, headline=row['headline'],
                published_at_utc=row['published_at_utc'], claim_being_checked=item['main_factual_claim'],
                earlier_article_id=prior['article_id'], earlier_headline=prior['headline'],
                earlier_published_at_utc=prior['published_at_utc'], earlier_source=prior['source'],
                earlier_url=prior['url'], supporting_passage=quote,
                relationship=evidence['relationship'], explanation=item['explanation']))
        classified.append(dict(row, label=item['label'], main_factual_claim=item['main_factual_claim'],
            new_information=item['new_information'], current_supporting_passage=item['current_supporting_passage'],
            explanation=item['explanation'], evidence_basis=item['evidence_basis'],
            earlier_article_ids='|'.join(references), caveat=item['caveat'],
            assessment_method='assistant_content_reading_and_earlier_dataset_fact_comparison',
            source_text_scope='complete_saved_headline_and_summary_not_unavailable_full_article',
            dataset_start_utc=raw[0]['published_at_utc'],
            first_public_disclosure_verified='false', web_used='false', price_outcomes_used='false'))
    assert classified, 'No explicit content assessments recorded'
    # A completed prefix makes the resume point unambiguous. No automatic labels
    # are assigned to the remainder or imported from the abandoned screen.
    reviewed_ids = set(annotations)
    assert [r['article_id'] for r in raw[:len(classified)]] == [r['article_id'] for r in classified], 'Chronological review has a gap'
    fields = list(classified[0])
    write_csv(HERE/'classified_articles.csv', fields, classified)
    new_articles = [r for r in classified if r['label'] in {'new_information_candidate', 'mixed'}]
    write_csv(HERE/'new_articles.csv', fields, new_articles)
    ef = ['article_id', 'headline', 'published_at_utc', 'claim_being_checked', 'earlier_article_id',
          'earlier_headline', 'earlier_published_at_utc', 'earlier_source', 'earlier_url',
          'supporting_passage', 'relationship', 'explanation']
    write_csv(HERE/'earlier_article_evidence.csv', ef, evidence_rows)
    write_csv(HERE/'review_status.csv', list(inventory[0]), inventory)
    pending = next((r for r in raw if r['article_id'] not in reviewed_ids), None)
    progress = dict(total_articles=len(raw), assessed_articles=len(classified),
        remaining_articles=len(raw)-len(classified), new_article_candidates=len(new_articles),
        label_counts=dict(Counter(r['label'] for r in classified)), evidence_rows=len(evidence_rows),
        assessed_through_utc=classified[-1]['published_at_utc'], next_article=({k: pending[k] for k in
            ['article_id', 'published_at_utc', 'headline']} if pending else None),
        october_total=sum(r['published_at_utc'].startswith('2025-10') for r in raw),
        october_assessed=sum(r['published_at_utc'].startswith('2025-10') for r in classified),
        input_sha256=original_hash, full_task_complete=len(classified)==len(raw),
        previous_provisional_reviews_imported=False, similarity_labels_used=False, web_used=False,
        notes='New means a specific dataset-relative candidate, not first public release. Historical recaps may be self-dated in the target when earlier source records are unavailable. New facts can coexist with recaps or price reaction.')
    (HERE/'progress.json').write_text(json.dumps(progress, indent=2)+'\n')
    assert hashlib.sha256(INPUT.read_bytes()).hexdigest() == original_hash
    print(json.dumps(progress, indent=2))


if __name__ == '__main__':
    main()
