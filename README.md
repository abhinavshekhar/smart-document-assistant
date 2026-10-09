# Smart Document Knowledge Assistant

A free study app for first-year students. Upload lecture PDFs and text notes, ask a question, and get an answer that uses **only those files**. Every reply lists the snippets it used, so you can check it.

The live app is on [Streamlit Community Cloud](https://share.streamlit.io). Open the app you created from this repository (main file: `doc-assistant/app.py`).

Code: [github.com/abhinavshekhar/smart-document-assistant](https://github.com/abhinavshekhar/smart-document-assistant)

## What it does

Ctrl+F cannot follow meaning, synonyms, or links across pages. This app embeds your notes locally, finds the closest chunks by cosine similarity, and asks Groq to answer from that context only.

If the answer is not in the files, it replies exactly:

`I couldn't find this in the uploaded documents.`

```
  browser
     |
     v
 +----------------+    upload pdf/txt     +------------------+
 | app.py         | --------------------> | session index    |
 | Streamlit UI   |                       | chunks + vectors |
 +----------------+                       +------------------+
     |                                              |
     |  question                                    | top 3-5 chunks
     v                                              v
 +----------------+   rewrite follow-up    +------------------+
 | rag.py         | ---------------------> | MiniLM embeddings|
 |                |   local, CPU only      | cosine search    |
 | generate_text  | ---------------------> | Groq free tier   |
 +----------------+                        +------------------+
```

Nothing is saved to disk. The index and the chat live in `st.session_state`, because Streamlit Cloud resets the filesystem.

## Features

- Upload one or many `.pdf` and `.txt` files
- Skip duplicates, delete a file from the index, warn on empty files and scanned PDFs (no OCR)
- Semantic search with `all-MiniLM-L6-v2` (normalized embeddings)
- Streamed answers with a **Referenced snippets** expander
- Follow-up questions (the last ~6 chat turns are sent; the query is rewritten first)
- Summarize a file, generate 5 quiz questions
- Friendly messages for a missing Groq key, network errors, and rate limits

## Repo layout

```
.
├── README.md
├── requirements.txt          # Streamlit Cloud installs this
├── runtime.txt               # python-3.12
├── .streamlit/config.toml
├── .streamlit/secrets.toml.example
├── index.html                # Fastshot landing mock (optional)
└── doc-assistant/
    ├── app.py                # UI only
    ├── rag.py                # extract, chunk, retrieve, generate
    ├── sample_docs/
    └── tests/test_rag.py
```

## Free Groq key

1. Create an account at [console.groq.com](https://console.groq.com). The free plan does not need a credit card.
2. Create an API key.
3. **Local:** set `GROQ_API_KEY` or copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
4. **Cloud:** app settings → Advanced → Secrets:

```toml
GROQ_API_KEY = "your-key-here"
```

`llama-3.3-70b-versatile` was removed from Groq's free tier on 16 August 2026. Generation uses `openai/gpt-oss-120b` (`GROQ_MODEL` in `doc-assistant/rag.py`).

Never commit a real key. `.streamlit/secrets.toml` and `.env` are gitignored.

## Run locally

Python 3.10 or newer. From the repository root:

### Windows (PowerShell)

```powershell
cd doc-assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ..\requirements.txt
$env:GROQ_API_KEY = "your-key-here"
streamlit run app.py
```

If `python` is not on PATH, use `py -3` instead.

### Mac / Linux

```bash
cd doc-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
export GROQ_API_KEY="your-key-here"
streamlit run app.py
```

The root `requirements.txt` installs CPU-only PyTorch (`torch==2.14.1+cpu`) from `https://download.pytorch.org/whl/cpu` so the app fits Streamlit's free ~1 GB machine.

## Tests

```bash
cd doc-assistant
python -m unittest tests/test_rag.py
```

These tests do not call the LLM. Retrieval tests download `all-MiniLM-L6-v2` once (~90 MB).

```bash
python rag.py
```

prints a paraphrased retrieval check against the sample notes.

## Deploy on Streamlit Community Cloud

1. This repository is already on GitHub.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. Create app → repository `abhinavshekhar/smart-document-assistant` → branch `master` → main file **`doc-assistant/app.py`**.
4. Advanced settings → Secrets → `GROQ_API_KEY`.
5. Deploy. First boot installs torch and the embedding model and can take several minutes.

## Five-step demo

1. Open the live app and wait for **Loading AI model...** to finish.
2. Upload `doc-assistant/sample_docs/roman_republic.pdf`, `newton_motion.pdf`, and `cell_energy.txt`.
3. Ask: `What do mitochondria produce?` Expect ATP, with snippets from `cell_energy.txt`.
4. Ask: `Which organelle creates the energy currency?` Same mitochondria snippet, even without those words in the file.
5. Ask: `What is the capital of Japan?` Expect exactly `I couldn't find this in the uploaded documents.` Then ask about ATP again, follow with `give an example of that`, then delete `cell_energy.txt` and confirm that file is gone from later answers.

## Sample questions

| Question | Expected |
| --- | --- |
| What do mitochondria produce? | ATP. Snippet from `cell_energy.txt`. |
| Which organelle creates the energy currency? | Same mitochondria snippet. |
| Why does water dissolve table salt? | Water is polar. `water_polarity.txt`. |
| When did the Roman Republic begin? | 509 BCE. `roman_republic.pdf`. |
| What is the capital of Japan? | `I couldn't find this in the uploaded documents.` |
| After an ATP answer: give an example of that | An ATP example still from the notes. |

## Troubleshooting

- **Memory errors.** Load only MiniLM. The CPU torch pin is required; a default `torch` install is larger.
- **Slow first load.** The embedding model downloads once per machine / Cloud wake.
- **Sleeping Cloud app.** Free apps sleep. The first visit wakes them. The index is empty after a restart.
- **Missing key.** The UI points to console.groq.com. No stack traces.
- **Rate limit.** `Free API limit reached, wait a minute and try again.`
- **Scanned PDFs.** No text layer → OCR is not supported.

## Limits

- No OCR.
- Chat and index disappear when the session or Cloud app restarts.
- Long notes are trimmed before Groq so they stay inside the free token budget.
