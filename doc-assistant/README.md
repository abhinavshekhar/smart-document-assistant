# Smart Document Knowledge Assistant

A free study tool for first-year students. Upload lecture PDFs and text notes, ask a question, and get an answer that uses only those files. Each answer lists the snippets it used, so you can check it.

The interface uses the Fastshot visual system: a full-bleed dawn video, glass panels, Inter, and an orange send button. The product behind that frame is this document assistant. Assumption: the landing-page headline was replaced with study copy so the screen matches what the app actually does.

```
  browser
     |
     v
 +---------+    upload pdf/txt     +------------------+
 | app.py  | --------------------> | session index    |
 | chat UI |                       | chunks + vectors |
 +---------+                       +------------------+
     |                                      |
     |  question                            | top 4 chunks
     v                                      v
 +---------+   rewrite follow-up    +------------------+
 | rag.py  | ----------------------> | MiniLM embeddings|
 |         |   local, CPU only       | cosine search    |
 |         | ----------------------> | Groq free tier   |
 +---------+   one function:         +------------------+
               generate_text()
```

Nothing is saved to disk. The index and the chat live in `st.session_state`, because Streamlit Community Cloud resets the filesystem.

## Free Groq key

1. Open [console.groq.com](https://console.groq.com) and create an account. The free developer plan does not need a credit card.
2. Create an API key.
3. Locally, set `GROQ_API_KEY`. On Streamlit Cloud, paste it into Secrets (see below).

`llama-3.3-70b-versatile` was removed from the free tier on 16 August 2026. `rag.py` uses `openai/gpt-oss-120b`, which is Groq's documented replacement. The name is the constant `GROQ_MODEL` at the top of `rag.py`.

## Local setup

Python 3.10 or newer.

### Windows (PowerShell)

```powershell
cd doc-assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:GROQ_API_KEY = "your-key-here"
streamlit run app.py
```

If `python` is not on your PATH, use the launcher instead: `py -3 -m venv .venv`.

### Mac / Linux

```bash
cd doc-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY="your-key-here"
streamlit run app.py
```

`requirements.txt` installs CPU-only PyTorch (`torch==2.14.1+cpu`) from `https://download.pytorch.org/whl/cpu` so the app fits Streamlit's free 1 GB machine. The CPU index no longer publishes the 2.2–2.5 wheels, so the pin is a current CPU build. If that index is unreachable, change the torch line to `torch` and install again.

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` if you prefer a secrets file over the environment variable. The real `secrets.toml` is gitignored.

## Tests

```bash
python -m unittest tests/test_rag.py
```

Chunking tests are instant. Retrieval tests download `all-MiniLM-L6-v2` once (about 90 MB) and do not call Groq.

Terminal retrieval check, from `doc-assistant`:

```bash
python rag.py
```

## Streamlit Community Cloud

This repository is ready to deploy from its root. `requirements.txt` and `runtime.txt` sit next to `doc-assistant/`.

1. Push this folder to GitHub.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with the same GitHub account.
3. Create a new app. Choose this repository, branch `master`, and set the main file path to `doc-assistant/app.py`.
4. Open Advanced settings before you deploy. Paste this into Secrets:

```toml
GROQ_API_KEY = "your-key-here"
```

5. Deploy. The first boot installs CPU PyTorch and downloads the embedding model, so it can take several minutes. The page shows "Loading AI model..." when it is ready to talk.

Python on Cloud is pinned to 3.12 in `runtime.txt` so the CPU torch wheel matches.

## Five-step demo

1. Start the app and wait until the loading spinner finishes.
2. In the sidebar, upload `sample_docs/roman_republic.pdf`, `sample_docs/newton_motion.pdf`, and `sample_docs/cell_energy.txt`. Confirm each file reports chunks indexed.
3. Ask: "What do mitochondria produce?" The answer should mention ATP and open "Referenced snippets" on `cell_energy.txt`.
4. Ask: "Which organelle creates the energy currency?" The wording does not appear in the notes. The top snippet should still be the mitochondria paragraph.
5. Ask: "What is the capital of Japan?" The reply should be exactly `I couldn't find this in the uploaded documents.` Then ask "give an example of that" after the ATP answer (ask the ATP question again first). The follow-up should talk about ATP, not a new topic. Delete `cell_energy.txt` and ask the mitochondria question again. It should no longer answer from that file.

## Six check questions

| Question | Expected |
| --- | --- |
| What do mitochondria produce? | ATP, the molecule cells spend to do work. Snippet from `cell_energy.txt`. |
| Which organelle creates the energy currency? | Same mitochondria snippet, even though those words are not in the file. |
| Why does water dissolve table salt? | Water is polar: oxygen holds the shared electrons more tightly, and the bent shape leaves uneven charge. Snippet from `water_polarity.txt`. |
| When did the Roman Republic begin? | 509 BCE, after the last king was driven out. Snippet from `roman_republic.pdf`. |
| What is the capital of Japan? | Exactly: I couldn't find this in the uploaded documents. |
| After an ATP answer, "give an example of that" | An example of ATP use, still grounded in the uploaded notes. |

## Sample files

- `sample_docs/cell_energy.txt` — mitochondria and ATP
- `sample_docs/water_polarity.txt` — why water is polar
- `sample_docs/study_habits.txt` — retrieval practice and sleep
- `sample_docs/roman_republic.pdf` — 509 BCE and one-year consuls
- `sample_docs/newton_motion.pdf` — a force changes motion

## Troubleshooting

- **Memory errors on Streamlit Cloud.** The free machine has about 1 GB of RAM. This app loads only MiniLM. Do not add a second model. If a deploy still dies during install, the CPU torch wheel is the first thing to check. Falling back to a default `torch` install often pulls a larger build.
- **Slow first load.** The embedding model downloads on the first run. Later runs reuse it until the app sleeps.
- **Sleeping app.** Free apps sleep after a period of no visitors. The first visit wakes them and shows the loading spinner again. The index is empty after a wake because it is not stored on disk.
- **Missing key.** The page explains how to get a key at console.groq.com and where to put it in Streamlit secrets. No stack trace is shown.
- **Rate limits.** The free Groq plan throttles bursts. The page says: Free API limit reached, wait a minute and try again.
- **Scanned PDFs.** Pages that are only images have no text layer. The sidebar warns that OCR is not supported.

## Limits

- OCR is not included.
- The chat and the index disappear when the browser session or the Cloud app restarts.
- Very long notes are trimmed before they are sent to Groq so the request stays inside the free token budget.
