"""Exploratory day-level word/phrase prevalence before existing 1% movements."""
from collections import defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from News_word_frequency.group_word_frequency import STOPWORDS

STOP = STOPWORDS - {'not', 'no', 'nor'}
OUT = Path(__file__).resolve().parent / 'distinctive_terms'
INPUT = ROOT / 'data/matched/2025-09-15_2026-05-15'
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.I)


def candidates(text):
    # URLs/numeric-only tokens are barriers, not bridges between adjacent words.
    text = re.sub(r'https?://\S+|www\.\S+', '\n', str(text), flags=re.I)
    text = text.lower().replace('’', "'")
    text = re.sub(r"\bwon't\b", 'will not', text)
    text = re.sub(r"\bcan't\b", 'can not', text)
    text = re.sub(r"n't\b", ' not', text)
    result = set()
    for sentence in re.split(r'[.!?;:\n\r]+', text):
        previous = None
        for token in TOKEN.findall(sentence):
            if not any(c.isalpha() for c in token) or len(token) < 2:
                previous = None
                continue
            if token not in STOP:
                result.add(token)
            if previous is not None and (previous not in STOP or token not in STOP):
                result.add(previous + ' ' + token)
            previous = token
    return result


def article_terms(row):
    return candidates(row['headline']) | candidates(row['summary'])


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write(path, rows, fields):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prevalence(day_terms, directions, eligible, minimum):
    up = {d for d in eligible if directions[d] == 'up'}
    down = set(eligible) - up
    if not up or not down:
        raise ValueError('Both up and down event days are required')
    support = defaultdict(set)
    for day in eligible:
        for term in day_terms[day]:
            support[term].add(day)
    result = []
    for term, days in support.items():
        if len(days) < minimum:
            continue
        u, d = len(days & up), len(days & down)
        result.append({'term': term, 'type': 'phrase' if ' ' in term else 'word',
            'up_support_days': u, 'down_support_days': d, 'total_support_days': len(days),
            'up_total_days': len(up), 'down_total_days': len(down),
            'up_prevalence_pct': 100*u/len(up), 'down_prevalence_pct': 100*d/len(down),
            'up_minus_down_pp': 100*(u/len(up)-d/len(down))})
    return sorted(result, key=lambda r: (-r['up_minus_down_pp'], r['term']))


def leaders(rows, direction):
    sign = 1 if direction == 'up' else -1
    return sorted([r for r in rows if sign*r['up_minus_down_pp'] > 0],
                  key=lambda r: (-sign*r['up_minus_down_pp'], -r['total_support_days'], r['term']))[:20]


def main():
    OUT.mkdir(exist_ok=True)
    paths = [INPUT/'event_summary.csv', INPUT/'event_article_matches.csv']
    before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    events, matches = map(read, paths)
    directions = {e['event_date']: 'up' if e['price_move_1pct']=='1' else 'down' for e in events}
    assert all(e['price_move_1pct'] in ['1', '-1'] for e in events)
    eligible = {e['event_date'] for e in events if e['collection_window_verified_24h']=='1'}
    day_terms = {d: set() for d in directions}
    evidence = defaultdict(list)
    cache, pairs, day_articles = {}, set(), defaultdict(set)
    article_directions = defaultdict(set)
    for row in matches:
        if row['in_24h'] != '1':
            continue
        day, key = row['event_date'], row['article_match_key']
        pair = (day, key)
        assert pair not in pairs
        pairs.add(pair)
        assert 0 < float(row['hours_before_cutoff']) <= 24
        if key not in cache:
            cache[key] = article_terms(row)
        day_terms[day].update(cache[key])
        day_articles[day].add(key)
        if day in eligible:
            article_directions[key].add(directions[day])
            for term in cache[key]:
                evidence[(term, directions[day])].append(row)
    ranked = prevalence(day_terms, directions, eligible, 5)
    fields = ['term','type','up_support_days','down_support_days','total_support_days',
              'up_total_days','down_total_days','up_prevalence_pct','down_prevalence_pct','up_minus_down_pp']
    write(OUT/'term_prevalence.csv', ranked, fields)
    selected = {}
    for direction in ['up','down']:
        selected[direction] = leaders(ranked, direction)
        write(OUT/f'top20_{direction}.csv', selected[direction], fields)
    examples = []
    for term in sorted({r['term'] for rows in selected.values() for r in rows}):
        for direction in ['up', 'down']:
            seen = set()
            for row in sorted(evidence[(term,direction)], key=lambda r:(r['event_date'],r['article_id'])):
                if row['article_match_key'] in seen:
                    continue
                seen.add(row['article_match_key'])
                examples.append({'term':term,'direction':direction,
                    **{k:row[k] for k in ['event_date','article_id','article_match_key','headline','summary','source','url',
                                         'publication_timestamp_utc','cutoff_utc']}})
                if len(seen)==3:
                    break
    example_fields = ['term','direction','event_date','article_id','article_match_key','headline','summary','source','url',
                      'publication_timestamp_utc','cutoff_utc']
    write(OUT/'supporting_articles.csv',examples,example_fields)
    vocabulary = sorted(r['term'] for r in ranked)
    matrix = [{'event_date':day,'direction':directions[day],'included_in_primary':int(day in eligible),
               'matched_articles_24h':len(day_articles[day]),
               **{'term::'+t:int(t in day_terms[day]) for t in vocabulary}} for day in sorted(directions)]
    write(OUT/'event_term_presence.csv',matrix,['event_date','direction','included_in_primary','matched_articles_24h']+['term::'+t for t in vocabulary])
    audit = [{'event_date':e['event_date'],'direction':directions[e['event_date']],
              'included_in_primary':int(e['event_date'] in eligible),
              'matched_articles_24h':len(day_articles[e['event_date']]),
              'coverage_issues':e['coverage_issues'],
              'reason':'verified_24h_collection' if e['event_date'] in eligible else 'incomplete_or_uncertain_24h_collection'}
             for e in events]
    write(OUT/'event_audit.csv',audit,list(audit[0]))
    sensitivity = []
    for scope, days in [('verified_24h',eligible),('all_events',set(directions))]:
        for minimum in [3,5,10]:
            rows = prevalence(day_terms,directions,days,minimum)
            for direction in ['up','down']:
                sensitivity.extend({'scope':scope,'minimum_support_days':minimum,'favored_direction':direction,
                                    'rank':i,**r} for i,r in enumerate(leaders(rows,direction),1))
    write(OUT/'sensitivity_top20.csv',sensitivity,['scope','minimum_support_days','favored_direction','rank']+fields)
    assert all(len({d for d in eligible if r['term'] in day_terms[d]})==r['total_support_days'] for r in ranked)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==before[str(p.relative_to(ROOT))] for p in paths)
    metadata = {'source_sha256':before,'sources_unchanged':True,'event_days':len(events),
        'primary_days':len(eligible),'primary_up_days':sum(directions[d]=='up' for d in eligible),
        'primary_down_days':sum(directions[d]=='down' for d in eligible),
        'excluded_coverage_days':sorted(set(directions)-eligible),'candidates_passing_5_day_minimum':len(ranked),
        'articles_matching_both_directions_primary':sum(len(ds)>1 for ds in article_directions.values()),
        'primary_unique_articles':len(article_directions), 'stopwords':sorted(STOP),
        'rules':'headline/summary separately; lowercase; unigrams and original adjacent bigrams; sentence boundaries; URLs/numbers are barriers; keep alphanumeric terms and negation; minimum five combined distinct days',
        'limitations':['Exploratory full-sample vocabulary, not validated predictive features or significance tests.',
            'Retrospective article availability remains uncertain; repeated stories and overlapping windows remain dependent.',
            'Verified collection windows do not guarantee complete publisher coverage.',
            'No stemming; numbers removed; contractions with n\u2019t expanded to not; sentence splitting is conservative punctuation-based.',
            'Missing September stock flags and news after May 1 remain unavailable.'],
        'validation':{'unique_pairs':True,'strict_pre_cutoff_24h':True,'day_support_counts_verified':True}}
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    page=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Distinctive words before NVDA moves</title>',
          '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}details{margin:12px 0}.note{background:#fff3d0;padding:15px}</style>',
          '<h1>Words and phrases before NVDA up/down days</h1>',
          f'<p>{metadata["primary_up_days"]} up days and {metadata["primary_down_days"]} down days with verified 24-hour collection windows. Each term counts once per day. Minimum support: five distinct days.</p>',
          '<p class="note">Ranked by difference in daily prevalence, not statistical significance or predictive power. Historical article availability is uncertain. Phrases may be boilerplate; inspect examples.</p>']
    for direction, rows in selected.items():
        page.append(f'<h2>More common before {direction} days</h2><table><tr><th>Term</th><th>Up days</th><th>Down days</th><th>Up % − down %</th></tr>')
        for r in rows:
            page.append(f'<tr><td>{html.escape(r["term"])}</td><td>{r["up_support_days"]}/{r["up_total_days"]} ({r["up_prevalence_pct"]:.1f}%)</td><td>{r["down_support_days"]}/{r["down_total_days"]} ({r["down_prevalence_pct"]:.1f}%)</td><td>{r["up_minus_down_pp"]:+.1f} pp</td></tr>')
        page.append('</table>')
        for r in rows:
            term=r['term']
            page.append(f'<details><summary>Examples: {html.escape(term)}</summary><ul>')
            for ex in examples:
                if ex['term']!=term: continue
                headline=html.escape(ex['headline'])
                if ex['url'].startswith(('https://','http://')):
                    headline=f'<a href="{html.escape(ex["url"],quote=True)}">{headline}</a>'
                page.append(f'<li>{ex["event_date"]} ({ex["direction"]}): {headline}<br>{html.escape(ex["summary"])}<br><small>Published {ex["publication_timestamp_utc"]}; cutoff {ex["cutoff_utc"]}</small></li>')
            page.append('</ul></details>')
    page.append('</html>')
    (OUT/'report.html').write_text('\n'.join(page))
    print(json.dumps({k:v for k,v in metadata.items() if k not in ['stopwords','source_sha256']},indent=2))
    for direction, rows in selected.items():
        print(direction.upper(), [(r['term'],round(r['up_minus_down_pp'],1),r['total_support_days']) for r in rows[:10]])


if __name__=='__main__': main()
