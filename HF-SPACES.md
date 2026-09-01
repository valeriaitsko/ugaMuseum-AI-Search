# Deploying the backend to Hugging Face Spaces (free)

A free HF **Docker Space** (2 vCPU, 16 GB RAM) has enough memory for the semantic
search. This runs the same FastAPI app as `Dockerfile`; you only add a Space
README and two secrets. The **shared-secret guard** (`PROXY_SECRET`) keeps the
public Space URL from being called by anyone but our frontend — important, because
a hit to `/api/ai-search` spends your Anthropic credits.

## 1. Create the Space

On [huggingface.co](https://huggingface.co): **New → Space** → choose **Docker**
(blank template) → make it Public (fine — the guard protects the paid calls) →
create. It's an empty git repo, e.g. `huggingface.co/spaces/<you>/museum-ai-search`.

## 2. Put the code in it

The Space needs the app, the `Dockerfile`, `requirements.txt`, the data files, and
a `README.md` with HF metadata. The data files are big, so they go through **git
LFS** (HF rejects >10 MB otherwise).

```bash
git clone https://huggingface.co/spaces/<you>/museum-ai-search
cd museum-ai-search

# copy from the ugaMuseum-AI-Search repo:
cp -r <repo>/app <repo>/requirements.txt <repo>/Dockerfile .

# LFS-track the large data files, then add them
git lfs install
git lfs track "*.npz" "app/data/*.json"
git add .gitattributes app requirements.txt Dockerfile
```

Create the Space's **`README.md`** with this metadata block at the very top (this
is what tells HF to build the Docker app on port 7860):

```markdown
---
title: Museum AI Search Backend
emoji: 🦋
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

FastAPI backend for the museum AI search. Not a demo UI — an API the museum site
proxies to. See the ugaMuseum-AI-Search repo.
```

Then commit and push:

```bash
git add README.md && git commit -m "Museum AI search backend" && git push
```

HF builds the image (a few minutes) and starts it. When the Space is "Running,"
its URL is `https://<you>-museum-ai-search.hf.space`.

## 3. Set the two secrets on the Space

Space → **Settings → Variables and secrets → New secret** (secrets, not public
variables):

- `ANTHROPIC_API_KEY` = your Anthropic key (for query parsing + the summarizer).
- `PROXY_SECRET` = any long random string (e.g. from a password generator). This
  is the shared secret — the same value goes on Vercel next.

Restart the Space after adding them.

## 4. Point the frontend at it (Vercel)

In the **fMuseum** frontend's Vercel project → **Settings → Environment Variables**:

- `AI_BACKEND_URL` = `https://<you>-museum-ai-search.hf.space`
- `PROXY_SECRET` = **the exact same string** you set on the Space.

Redeploy the frontend. The Express proxy now sends `X-Proxy-Secret` on every call,
and the backend only accepts calls carrying it — so the public Space URL is useless
to anyone else.

## Sanity checks

- `https://<you>-museum-ai-search.hf.space/` returns `{"specimens": ...,
  "semantic_search": true}` (the health route stays open — it leaks nothing).
- A raw `POST /api/ai-search` **without** the header should return **401**. With the
  header (i.e. through the frontend) it works. That's the guard doing its job.

## Notes

- **Cold starts:** free Spaces sleep when idle; the first search after a quiet spell
  waits for the container to wake and load the model (~a minute).
- **Local dev is unchanged:** with `PROXY_SECRET` unset (as in your `.env`), the
  guard is off and everything works exactly as before.
- **Updating data:** rebuild `specimens.json`/`enriched.json` (and `vectors.npz`)
  locally, copy them into the Space repo, and push — LFS handles the big files.
