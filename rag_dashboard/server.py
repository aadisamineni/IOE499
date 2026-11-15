"""Sentry News: local dashboard for the news CSV and Qwen allocation workflow."""
import argparse
import csv
import hashlib
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import ProxyHandler, build_opener

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import local_qwen

SYMBOLS = ('NVDA', 'SPY', 'META', 'PFE')
MODEL = local_qwen.MODEL


def build_index(csv_path, cache, refresh=False):
    csv_path = Path(csv_path).resolve()
    stat = csv_path.stat()
    identity = f'{csv_path}:{stat.st_size}:{stat.st_mtime_ns}:v1'
    cache.mkdir(parents=True, exist_ok=True)
    db = cache / ('merged-' + hashlib.sha256(identity.encode()).hexdigest()[:16] + '.sqlite')
    if db.exists() and not refresh:
        return db
    temporary = db.with_suffix('.building.sqlite')
    with sqlite3.connect(temporary) as conn, csv_path.open(encoding='utf-8-sig', newline='') as source:
        conn.executescript('''DROP TABLE IF EXISTS articles; DROP TABLE IF EXISTS metadata;
            CREATE TABLE articles (id INTEGER PRIMARY KEY, day TEXT, provider TEXT,
                symbols TEXT, published TEXT, headline TEXT, summary TEXT, record TEXT);
            CREATE TABLE metadata (source TEXT, indexed_at TEXT);''')
        reader = csv.DictReader(source, strict=True)
        required = {'date_utc', 'provider', 'article_id', 'headline', 'summary', 'query_symbols',
                    'published_at_utc', *(f'weight_{s}' for s in SYMBOLS)}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError('The news CSV is missing required fields.')
        batch = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'Invalid CSV row ending at line {reader.line_num}.')
            try:
                datetime.strptime(row['date_utc'], '%Y-%m-%d')
            except ValueError:
                raise ValueError(f'Invalid date at CSV line {reader.line_num}.')
            record = {key: value for key, value in row.items() if key != 'content_html'}
            symbols = ','.join(sorted(set((row.get('query_symbols', '') + ',' + row.get('related_symbols', '')).upper().split(','))))
            batch.append((row['date_utc'], row['provider'].lower(), symbols,
                          row['published_at_utc'], row['headline'], row['summary'], json.dumps(record, ensure_ascii=False)))
            if len(batch) >= 1000:
                conn.executemany('INSERT INTO articles(day,provider,symbols,published,headline,summary,record) VALUES (?,?,?,?,?,?,?)', batch)
                batch.clear()
        if batch:
            conn.executemany('INSERT INTO articles(day,provider,symbols,published,headline,summary,record) VALUES (?,?,?,?,?,?,?)', batch)
        conn.executescript('CREATE INDEX article_day ON articles(day); CREATE INDEX article_provider ON articles(provider);')
        conn.execute('INSERT INTO metadata VALUES (?,?)', (str(csv_path), datetime.now(timezone.utc).isoformat()))
    temporary.replace(db)
    return db


def overview(db, model=MODEL):
    with sqlite3.connect(db) as conn:
        rows = conn.execute('''SELECT day, count(*), sum(provider='alpaca'), sum(provider='finnhub')
                               FROM articles GROUP BY day ORDER BY day''').fetchall()
        source, indexed = conn.execute('SELECT source,indexed_at FROM metadata').fetchone()
    return {'days': [{'date_utc': d, 'article_count': n, 'alpaca_article_count': a,
                      'finnhub_article_count': f} for d, n, a, f in rows],
            'source': source, 'indexed_at': indexed, 'total': sum(r[1] for r in rows), 'model': model}


def search_articles(db, day, symbol='', provider='', query='', page=1, limit=8):
    datetime.strptime(day, '%Y-%m-%d')
    if symbol and symbol not in SYMBOLS:
        raise ValueError('Unknown company or fund.')
    if provider and provider not in ('alpaca', 'finnhub'):
        raise ValueError('Unknown news provider.')
    if not 1 <= page <= 100000 or not 1 <= limit <= 100:
        raise ValueError('Invalid page or page size.')
    if len(query) > 200:
        raise ValueError('Search is limited to 200 characters.')
    clauses, params = ['day = ?'], [day]
    if symbol:
        clauses.append("(',' || symbols || ',') LIKE ?")
        params.append(f'%,{symbol},%')
    if provider:
        clauses.append('provider = ?')
        params.append(provider)
    for word in query.split()[:12]:
        clauses.append("(lower(headline) LIKE ? ESCAPE '\\' OR lower(summary) LIKE ? ESCAPE '\\')")
        term = '%' + word.lower().replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_') + '%'
        params.extend((term, term))
    where = ' AND '.join(clauses)
    with sqlite3.connect(db) as conn:
        total = conn.execute(f'SELECT count(*) FROM articles WHERE {where}', params).fetchone()[0]
        rows = conn.execute(f'SELECT record FROM articles WHERE {where} ORDER BY id LIMIT ? OFFSET ?',
                            [*params, limit, (page - 1) * limit]).fetchall()
    return {'total': total, 'page': page, 'page_size': limit, 'articles': [json.loads(r[0]) for r in rows]}


def analyze(db, request, model=MODEL):
    if not isinstance(request, dict):
        raise ValueError('Expected a JSON object.')
    day = request.get('date', '')
    result = search_articles(db, day, request.get('symbol', ''), request.get('provider', ''),
                             request.get('query', ''), limit=1)
    if not result['articles']:
        raise ValueError('No articles match this date and these filters.')
    records = result['articles']
    question = 'Recommend target portfolio weights for the supplied news and explain the allocation.'
    payload = local_qwen.build_request(records, question)
    payload['model'] = model
    # The cluster's Qwen 3 32B can spend the entire small output budget in
    # thinking mode, leaving no JSON answer for the dashboard to display.
    if model == 'qwen3:32b':
        payload['think'] = False
        payload['format'] = 'json'
        payload['options']['num_predict'] = 1024
    raw = local_qwen.ask_ollama(payload)
    try:
        answer = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError('Qwen returned invalid JSON. Please try again.') from exc
    if not isinstance(answer, dict):
        raise ValueError('Qwen returned an unexpected result. Please try again.')
    weights = answer.get('target_weights')
    reasoning = answer.get('reasoning')
    citations = answer.get('supporting_article_ids')
    if not isinstance(weights, dict) or not isinstance(reasoning, dict) or not isinstance(citations, list):
        raise ValueError('Qwen returned an incomplete result. Please try again.')
    if any(isinstance(weights.get(s), bool) or not isinstance(weights.get(s), (int, float)) or not math.isfinite(weights[s]) for s in SYMBOLS):
        raise ValueError('Qwen returned invalid portfolio weights. Please try again.')
    warnings = []
    if answer.get('date_utc') != day:
        warnings.append('The model returned a different date than the selected date.')
    if abs(sum(weights[s] for s in SYMBOLS) - 1) > 0.0001 or any(weights[s] < 0 for s in SYMBOLS):
        warnings.append('The model weights do not form a valid 100% portfolio.')
    if weights['SPY'] < .35 or weights['NVDA'] > .30 or weights['META'] > .25 or weights['NVDA'] + weights['META'] > .45:
        warnings.append('The model weights do not satisfy all of the pipeline allocation constraints.')
    allowed = {str(r['article_id']) for r in records}
    if any(str(c) not in allowed for c in citations):
        warnings.append('The model cited an article that was not included in its input.')
    if not citations:
        warnings.append('The model did not cite an article.')
    if any(not isinstance(reasoning.get(s), str) or not reasoning[s].strip() for s in SYMBOLS):
        warnings.append('The model did not explain every asset.')
    return {'output': answer, 'evidence': records, 'matching_articles': result['total'],
            'model': model, 'warnings': warnings}


def handler_class(db, model):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, value, status=200):
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            params = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == '/api/overview':
                    return self.send_json(overview(db, model))
                if url.path == '/api/articles':
                    return self.send_json(search_articles(db, params['date'], params.get('symbol', ''),
                        params.get('provider', ''), params.get('q', ''), int(params.get('page', 1))))
                if url.path == '/api/status':
                    try:
                        with build_opener(ProxyHandler({})).open(local_qwen.ENDPOINT.replace('/api/chat', '/api/tags'), timeout=2) as response:
                            tags = json.load(response)
                        available = model in [x.get('name') for x in tags.get('models', [])]
                        return self.send_json({'online': True, 'model_available': available, 'model': model})
                    except Exception:
                        return self.send_json({'online': False, 'model_available': False, 'model': model})
                files = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'), '/styles.css': ('styles.css', 'text/css')}
                if url.path not in files:
                    return self.send_json({'error': 'Not found'}, 404)
                name, mime = files[url.path]
                data = (HERE / name).read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', mime + '; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'")
                self.end_headers()
                self.wfile.write(data)
            except (ValueError, KeyError, TypeError) as exc:
                self.send_json({'error': str(exc)}, 400)

        def do_POST(self):
            if self.path != '/api/analyze':
                return self.send_json({'error': 'Not found'}, 404)
            host = self.headers.get('Host', '')
            if host != f'127.0.0.1:{self.server.server_port}' or self.headers.get('Origin') != f'http://{host}' or not self.headers.get('Content-Type', '').startswith('application/json'):
                return self.send_json({'error': 'Invalid request origin or content type.'}, 403)
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 8192:
                    raise ValueError('Request too large or empty.')
                self.send_json(analyze(db, json.loads(self.rfile.read(length)), model))
            except (ValueError, KeyError, TypeError) as exc:
                self.send_json({'error': str(exc)}, 400)
            except (RuntimeError, OSError) as exc:
                self.send_json({'error': str(exc)}, 503)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=local_qwen.DEFAULT_CSV, help='Merged news and weights CSV')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    print('Indexing local news; the first launch may take a moment.', flush=True)
    db = build_index(args.csv, HERE / '.cache', args.refresh)
    print(f"Indexed {overview(db)['total']:,} news records.", flush=True)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_class(db, args.model))
    print(f'Sentry News: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
