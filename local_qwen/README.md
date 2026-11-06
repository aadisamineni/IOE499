# Local CSV → Qwen starter

`ask_csv.py` reads a small CSV sample and sends the actual records to the local
Ollama chat API at `http://127.0.0.1:11434/api/chat`, using the installed
`qwen3:30b-instruct` model. It uses only Python's standard library; no packages
or API keys are required. Open Ollama before sending a request.

From the project root, first inspect what the reader selects:

```sh
python3 local_qwen/ask_csv.py /path/to/your.csv --preview
```

Send those records to Qwen with a generic request to describe them:

```sh
python3 local_qwen/ask_csv.py /path/to/your.csv
```

Supply a specific question whenever ready:

```sh
python3 local_qwen/ask_csv.py /path/to/your.csv --limit 5 --question "Your question here"
```

The reader accepts any nonempty, unique CSV column names and preserves their
values. Use UTF-8 (an optional BOM is supported); properly quoted commas and
multiline fields are supported. Missing or extra fields produce an error.
It reads the first five data records by default, or up to `--limit` records;
records beyond that limit are neither read nor validated. The input file is
never modified. `--preview` makes no API call.

This starter passes all columns in the selected records unchanged. It does not
filter by ticker or date, rank articles, or remove future-return labels. Use a
small, appropriate input sample when moving beyond a connection test.

Requests use an 8,192-token context and at most 512 output tokens. A conservative
6,000-byte prompt limit keeps this initial version small; oversized input is
rejected with guidance instead of silently truncating article text. Preview
still works on larger selected records. The fixed instructions ask Qwen to use
the supplied records and cite `article_id` when present, but this starter does
not validate the factual accuracy of generated answers or their citations.

The endpoint is fixed to this computer and ignores HTTP proxy settings. The
script sends one independent request per run; it does not train Qwen or keep
chat history. If Ollama is unavailable, open the app. If the model is missing,
run `ollama pull qwen3:30b-instruct` once while online.

API reference: [Ollama chat endpoint](https://docs.ollama.com/api/chat).
