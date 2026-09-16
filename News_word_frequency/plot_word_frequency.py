"""Build lagged news-frequency data and exploratory Plotly charts."""
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
STOPWORDS = {"the", "and", "to", "of", "a", "in", "for", "on", "is", "with", "that", "as", "at", "by", "from", "it", "this", "be", "are", "an", "was", "or", "will", "has", "have", "its", "more", "about", "their", "they", "but", "not", "can", "into", "than", "also", "after", "over", "new", "up", "down", "how", "why", "what", "s", "t"}


def rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    overall = rows(HERE / "overall_word_frequency.csv")
    daily = rows(HERE / "daily_word_frequency.csv")
    stock = rows(ROOT / "data/processed/NVDA_daily_features_2025-10-01_2026-05-31.csv")
    stock_dates = [r["trading_date"] for r in stock]
    stock_by_date = {r["trading_date"]: r for r in stock}
    next_session = {stock_dates[i]: stock_dates[i + 1] for i in range(len(stock_dates) - 1)}
    top = [r["word"] for r in overall if r["word"] not in STOPWORDS and len(r["word"]) > 1][:12]

    # One row per word per New York publication date, joined to next NASDAQ session.
    grouped = defaultdict(lambda: {"occ": 0, "articles": 0, "utc_dates": set(), "categories": defaultdict(int)})
    for r in daily:
        if r["word"] in top:
            key = (r["article_new_york_date"], r["word"])
            g = grouped[key]
            g["occ"] += int(r["word_occurrences"])
            g["articles"] += int(r["articles_containing_word"])
            g["utc_dates"].add(r["article_utc_date"])
            g["categories"][r["top_event_category"]] += int(r["word_occurrences"])

    out = []
    for (news_date, word), g in sorted(grouped.items()):
        future = next((d for d in stock_dates if d > news_date), "")
        sr = stock_by_date.get(future, {})
        out.append({"news_date_new_york": news_date, "word": word, "word_occurrences": g["occ"], "articles_containing_word": g["articles"], "publication_utc_dates": "|".join(sorted(g["utc_dates"])), "next_trading_date": future, "next_day_close_to_close_return": sr.get("adjusted_close_return_1d", ""), "next_day_open_to_close_return": sr.get("adjusted_open_to_close_return", ""), "next_day_price_move_1pct": sr.get("price_move_1pct", ""), "next_day_price_move_5pct": sr.get("price_move_5pct", ""), "top_event_category": max(g["categories"], key=g["categories"].get)})

    with (HERE / "lagged_word_stock_dataset.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out[0]))
        writer.writeheader(); writer.writerows(out)

    chart_rows = [r for r in out if r["next_day_close_to_close_return"]]
    series = {}
    for word in top:
        wr = [r for r in chart_rows if r["word"] == word]
        series[word] = {"x": [int(r["word_occurrences"]) for r in wr], "y": [100 * float(r["next_day_close_to_close_return"]) for r in wr], "text": [r["news_date_new_york"] for r in wr]}
    trend = defaultdict(lambda: [0, 0])
    for r in chart_rows:
        trend[r["news_date_new_york"]][0] += 100 * float(r["next_day_close_to_close_return"])
        trend[r["news_date_new_york"]][1] += 1
    dates = sorted(trend)
    avg = [trend[d][0] / trend[d][1] for d in dates]
    js = json.dumps({"top": top, "series": series, "dates": dates, "avg": avg}, separators=(",", ":"))
    page = f'''<html><head><meta charset="utf-8"><title>NVDA news word frequency plots</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><style>body{{font-family:Arial,sans-serif;margin:32px;color:#15202b}}h1{{margin-bottom:4px}}.note{{color:#5b6570;max-width:900px}}.chart{{width:100%;height:520px;margin:24px 0}}</style></head><body><h1>NVDA news word frequency → next-session return</h1><p class="note">Exploratory plots using New York publication dates. News on date t is paired with the next NASDAQ trading session. Returns are associations, not evidence of causality.</p><div id="scatter" class="chart"></div><div id="trend" class="chart"></div><script>const d={js}; const colors=['#76b900','#2b6cb0','#c0392b','#8e44ad','#16a085','#d35400','#34495e','#f39c12','#7f8c8d','#27ae60','#2980b9','#8e44ad']; const traces=d.top.map((w,i)=>({{x:d.series[w].x,y:d.series[w].y,text:d.series[w].text,mode:'markers',name:w,marker:{{color:colors[i%colors.length],size:7,opacity:.65}}}})); Plotly.newPlot('scatter',traces,{{title:'Word frequency versus next-session NVDA close-to-close return',xaxis:{{title:'Word occurrences on news date'}},yaxis:{{title:'Next-session return (%)'}},hovermode:'closest',legend:{{orientation:'h'}}}},{{responsive:true}}); Plotly.newPlot('trend',[{{x:d.dates,y:d.avg,mode:'lines+markers',line:{{color:'#76b900',width:2}},marker:{{size:5}},name:'Average next-session return'}}],{{title:'Average next-session return across news publication dates',xaxis:{{title:'News publication date (New York)'}},yaxis:{{title:'Average next-session return (%)'}}}},{{responsive:true}});</script></body></html>'''
    (HERE / "word_frequency_plots.html").write_text(page, encoding="utf-8")
    print(f"Top words plotted: {', '.join(top)}")
    print(f"Lagged rows: {len(out)}")
    print(f"Wrote: {HERE / 'lagged_word_stock_dataset.csv'}")
    print(f"Wrote: {HERE / 'word_frequency_plots.html'}")


if __name__ == "__main__":
    main()
