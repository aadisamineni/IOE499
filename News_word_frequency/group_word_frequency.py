"""Compare existing preceding-news matches for inclusive 1%/3% up/down groups."""
from collections import Counter
import csv
from decimal import Decimal
import hashlib
import html
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / 'movement_groups'
STOCK = ROOT / 'data/processed/NVDA_daily_features_2025-10-01_2026-05-31.csv'
MATCHED = ROOT / 'data/matched/2025-09-15_2026-05-15'
GROUPS = ['1pct_up', '3pct_up', '1pct_down', '3pct_down']
# Explicit, reproducible common-English stopword list; domain words are retained.
STOPWORDS = set('''a an the and or but if while as at by for from in into of on onto to with without
is are was were be been being am has have had having do does did doing will would shall should can could may might must
i me my we us our you your he him his she her it its they them their this that these those there here
what which who whom whose when where why how not no nor so than too very just also only own same such
all any both each few more most other some up down out over under again further then once about between through during before after
above below off until against because otherwise even ever every much many another yet however now today
s t re ve ll d m'''.split())
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def read(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write(path, rows, fields):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def group_membership(row):
    change = Decimal(row['adjusted_close_return_1d'])
    return {'1pct_up': row['price_move_1pct'] == '1',
            '1pct_down': row['price_move_1pct'] == '-1',
            '3pct_up': change >= Decimal('0.03'),
            '3pct_down': change <= Decimal('-0.03')}


def frequencies(articles, filtered):
    counts, documents = Counter(), Counter()
    for row in articles:
        words = TOKEN.findall((row['headline'] + ' ' + row['summary']).lower())
        if filtered:
            words = [w for w in words if w.isalpha() and len(w) > 1 and w not in STOPWORDS]
        counts.update(words)
        documents.update(set(words))
    total = sum(counts.values())
    return [{'rank': rank, 'word': word, 'occurrences': count,
             'articles_containing_word': documents[word],
             'percent_of_group_articles': 100 * documents[word] / len(articles),
             'occurrences_per_1000_tokens': 1000 * count / total}
            for rank, (word, count) in enumerate(sorted(counts.items(), key=lambda x: (-x[1], x[0])), 1)], total


def main():
    OUT.mkdir(exist_ok=True)
    inputs = [STOCK, MATCHED / 'event_summary.csv', MATCHED / 'event_article_matches.csv']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    stocks = {r['trading_date']: r for r in read(STOCK)}
    events = read(inputs[1])
    matches = read(inputs[2])
    event_by_date = {r['event_date']: r for r in events}
    memberships = {day: group_membership(row) for day, row in stocks.items()
                   if '2025-09-15' <= day <= '2026-05-15'}
    for day, flags in memberships.items():
        assert not flags['3pct_up'] or flags['1pct_up']
        assert not flags['3pct_down'] or flags['1pct_down']
        if any(flags.values()):
            assert day in event_by_date
    top, raw_top, full, summaries, article_audit, event_audit = [], [], [], [], [], []
    for group in GROUPS:
        days = {d for d, flags in memberships.items() if flags[group]}
        linked = [r for r in matches if r['event_date'] in days and r['in_7d'] == '1']
        unique = {}
        for row in linked:
            key = row['article_match_key']
            if key in unique:
                assert (unique[key]['headline'], unique[key]['summary']) == (row['headline'], row['summary'])
            unique[key] = row
        articles = list(unique.values())
        counted, total = frequencies(articles, True)
        raw_counted, raw_total = frequencies(articles, False)
        top.extend({'group': group, **r} for r in counted[:10])
        raw_top.extend({'group': group, **r} for r in raw_counted[:10])
        full.extend({'group': group, **r} for r in counted)
        no_news = sorted(days - {r['event_date'] for r in linked})
        summaries.append({'group': group, 'flagged_days': len(days), 'days_with_matches': len(days)-len(no_news),
                          'unique_articles': len(articles), 'event_article_pairs': len(linked),
                          'filtered_tokens': total, 'unfiltered_tokens': raw_total,
                          'no_match_days': '|'.join(no_news),
                          'days_with_coverage_issues': sum(event_by_date[d]['collection_window_verified_7d'] == '0' for d in days)})
        for day in sorted(days):
            event_audit.append({'group': group, 'event_date': day,
                'price_move_1pct': stocks[day]['price_move_1pct'],
                'saved_adjusted_close_return_1d': stocks[day]['adjusted_close_return_1d'],
                'cutoff_utc': event_by_date[day]['cutoff_utc']})
        for key, row in sorted(unique.items()):
            article_audit.append({'group': group, 'article_match_key': key, 'article_id': row['article_id'],
                                  'headline': row['headline'], 'published_at_utc': row['published_at_utc'],
                                  'url': row['url']})
    fields = ['group', 'rank', 'word', 'occurrences', 'articles_containing_word',
              'percent_of_group_articles', 'occurrences_per_1000_tokens']
    write(OUT/'top10_words.csv', top, fields)
    write(OUT/'top10_words_unfiltered.csv', raw_top, fields)
    write(OUT/'all_word_counts.csv', full, fields)
    write(OUT/'group_summary.csv', summaries, list(summaries[0]))
    write(OUT/'group_events.csv', event_audit, list(event_audit[0]))
    write(OUT/'group_articles.csv', article_audit, list(article_audit[0]))
    for p in inputs:
        assert hashes[str(p.relative_to(ROOT))] == hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'metadata.json').write_text(json.dumps({'source_sha256': hashes,
        'source_files_unchanged': True, 'window': 'existing strict preceding 7-calendar-day matches',
        'text': 'headline plus summary', 'counting_unit': 'unique article_match_key within each group',
        'one_percent_groups': 'existing price_move_1pct signed flags',
        'three_percent_groups': 'saved adjusted_close_return_1d >= 0.03 or <= -0.03; no returns recomputed',
        'stopwords': sorted(STOPWORDS), 'tokenization': TOKEN.pattern,
        'filtered_view': 'alphabetic tokens of length >1 excluding listed stopwords; no stemming',
        'ranking': 'total occurrences descending, alphabetical ties',
        'overlap': '3% groups overlap 1% groups; an article can belong to up and down groups',
        'corpus': 'all matched direct/indirect articles, retaining original upstream excluded rows',
        'limitations': 'Missing September flags; news ends May 1; historical availability uncertain. Associations only.'}, indent=2)+'\n')
    page = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Words before NVDA up/down days</title>',
        '<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#182533}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:24px}section{border:1px solid #ddd;border-radius:12px;padding:20px}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:7px;border-bottom:1px solid #eee}.bar{background:#d9e9f5}small{color:#536273}</style>',
        '<h1>Top 10 words before NVDA movement days</h1><p>Seven-day preceding-news matches; each article counted once per group. Headline + summary. Common English words and numeric tokens removed. Counts are total word occurrences, not unique articles.</p>',
        '<p>Inclusive 1%/3% thresholds; groups overlap. Existing 1% flags and saved returns are used. Missing September flags and news after May 1 limit coverage. This comparison does not establish prediction or causation.</p><div class="grid">']
    for summary in summaries:
        group = summary['group']
        rows = [r for r in top if r['group']==group]
        page.append(f'<section><h2>{group.replace("pct", "%").replace("_", " ")}</h2><p>{summary["flagged_days"]} flagged days · {summary["unique_articles"]:,} unique articles</p><table><tr><th>Word</th><th>Occurrences</th><th>Articles (%)</th></tr>')
        maximum = rows[0]['occurrences'] if rows else 1
        for r in rows:
            width=100*r['occurrences']/maximum
            page.append(f'<tr><td>{html.escape(r["word"])}</td><td style="background:linear-gradient(to right,#d9e9f5 {width}%,transparent {width}%)">{r["occurrences"]:,}</td><td>{r["percent_of_group_articles"]:.1f}%</td></tr>')
        page.append(f'</table><p><small>Days with no matches: {summary["no_match_days"] or "none"}. Days with coverage warnings: {summary["days_with_coverage_issues"]}.</small></p></section>')
    page.append('</div></html>')
    (OUT/'report.html').write_text('\n'.join(page))
    print(json.dumps(summaries, indent=2))
    for group in GROUPS:
        print(group+': '+', '.join(f'{r["word"]} ({r["occurrences"]})' for r in top if r['group']==group))


if __name__ == '__main__':
    main()
