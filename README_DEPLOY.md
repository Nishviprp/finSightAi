# Deploying FinSight to HuggingFace Spaces

This guide walks through publishing FinSight as a live Streamlit app on
[HuggingFace Spaces](https://huggingface.co/spaces) — free hosting, no credit card.

---

## Prerequisites

- A HuggingFace account ([sign up free](https://huggingface.co/join))
- A Gemini API key ([get one free](https://aistudio.google.com/apikey))
- Git installed locally

---

## Step 1 — Create a new Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in the form:
   - **Owner**: your HF username
   - **Space name**: `finsight`
   - **License**: MIT
   - **SDK**: **Streamlit**
   - **Hardware**: CPU Basic (free)
3. Click **Create Space**

HuggingFace will create an empty git repo at:
```
https://huggingface.co/spaces/YOUR_USERNAME/finsight
```

---

## Step 2 — Add your Gemini API key as a secret

1. Open your Space → **Settings** tab → **Repository secrets**
2. Click **New secret**
   - **Name**: `GEMINI_API_KEY`
   - **Value**: `AIzaSy...` (your key from Google AI Studio)
3. Click **Save**

The secret is injected as `os.environ["GEMINI_API_KEY"]` at runtime.
`app.py` reads it and writes a `.env` file before anything else runs.

---

## Step 3 — Push the repo

```bash
# In your local finsight/ directory:

# Add HuggingFace as a remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/finsight

# Push
git push hf main
```

If prompted for credentials, use your HF username and a
[User Access Token](https://huggingface.co/settings/tokens) (write scope).

---

## Step 4 — Watch the build

Open your Space URL:
```
https://huggingface.co/spaces/YOUR_USERNAME/finsight
```

The **Logs** tab shows the pip install and Streamlit startup.
Cold-start takes ~60–90 seconds on CPU Basic.

---

## Step 5 — First-run behaviour

`app.py` automatically seeds two baked-in AAPL transcripts (Q3 + Q4 2023)
into the SQLite DB on first boot. The demo is immediately interactive:

- **Insights** page shows sentiment + risks for both quarters
- **Trend Analysis** shows the Q3→Q4 drift chart

To analyse additional tickers live:
1. Go to **Search & Fetch** → enter a ticker → click **Fetch transcripts**
2. Click **Analyze with AI**

> **Note:** the free-tier SQLite DB resets on each Space restart.
> For persistent data, mount a HuggingFace Dataset as a volume or
> swap `TranscriptStore` for a hosted Postgres instance.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `GEMINI_API_KEY not set` | Check Spaces Settings → Secrets |
| `RuntimeError: Gemini API call failed` | Key may have hit daily quota; wait 24 h or add billing |
| Space builds but shows blank page | Check Logs tab; usually a missing dependency |
| `ModuleNotFoundError: src` | Ensure `conftest.py` exists (adds project root to `sys.path`) |

---

## Space metadata (`README.md` front-matter for HF)

HuggingFace Spaces reads YAML front-matter from the README to configure
the Space card. Add this block at the very top of `README.md` before deploying:

```yaml
---
title: FinSight
emoji: 📈
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
license: mit
---
```

---

## Final Space URL

```
https://huggingface.co/spaces/YOUR_USERNAME/finsight
```

Share this link in your portfolio, LinkedIn, or CV.
