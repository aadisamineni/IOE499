# Sentry News

Sentry News is a local dashboard for exploring the merged news archive and generating an illustrative portfolio analysis with the existing `local_qwen.py` workflow. It uses Python's standard library and needs no frontend build.

From the IOE499 workspace root:

```sh
python3 rag_dashboard/server.py
```

Open <http://127.0.0.1:8765>. Start Ollama with `qwen3:30b-instruct` installed before generating an analysis. The news CSV uses Git LFS; install Git LFS and run `git lfs pull` after cloning. The first launch builds a local SQLite index in `.cache`; subsequent launches reuse it. Stop the server with Ctrl+C. Use `--port 8766` to change the port, `--csv /path/to/news.csv` to select a different merged CSV, `--model MODEL` for another installed Ollama model, and `--refresh` to rebuild the cache.

Choose a date and optionally filter by asset, provider, and search. Generate analysis sends the **first matching record in CSV order** to the existing Qwen request builder and Ollama client, excluding `content_html`, just like the default command-line script. The result shows the model's target weights, per-asset explanations, overall reasoning, and the supplied article. The app checks the weight total, allocation constraints, and citation IDs and flags problems; it does not independently verify model claims. Export saves the result and supplied record as JSON.

The archive uses **simulated starting weights**. It was collected later than the historical dates and has no verified point-in-time cutoff. It has no historical returns or covariance matrices. The output is a news-based heuristic from one article, not an optimized portfolio, a return forecast, or investment advice. Filtered results can omit relevant news. Prompt-size limits may prevent some records from being analyzed; no records are silently truncated.

Run offline checks with:

```sh
python3 -m unittest discover -s rag_dashboard -p 'test_*.py' -v
node --check rag_dashboard/app.js
```
