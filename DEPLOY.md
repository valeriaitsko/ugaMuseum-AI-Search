# Deploying the AI-search backend

Locally you just run `uvicorn app.main:app --port 8001` and the frontend on the
same machine reaches it. For a **hosted** site (a link others can use), the
frontend runs on the internet and can't reach your laptop — so this FastAPI
service has to run somewhere public too. These files deploy it to **Render**; the
`Dockerfile` also works on Railway, Fly, or any Docker host.

## Two builds: full vs lexical-only

| | `Dockerfile` (full) | `Dockerfile.lexical` (free-tier) |
|---|---|---|
| search | semantic (meaning-based) + keyword | keyword only — no "understood what I meant" matches |
| deps | includes torch (~2.5 GB) | no torch |
| RAM needed | ~1–2 GB (paid tier / HF free CPU) | fits **512 MB** (Render **free**) |
| everything else | identical | identical (filters, hints, summaries, archaeology) |

The app **auto-detects** which it is — the lexical build just omits the ML library,
and the code falls back to whole-word matching over the enriched text. You can
start lexical/free and later switch to the full build with **zero code changes**.

Sections 1–2 below use the full build. For the free build, see **§4**.

## 1. Deploy the backend to Render (full semantic search)

1. Push this repo to GitHub.
2. On [render.com](https://render.com): **New → Blueprint → connect this repo.**
   Render reads `render.yaml` and creates a Docker web service.
3. It uses the **Standard** plan (2 GB RAM). This is required — the 512 MB
   free/starter tiers can't hold PyTorch and will crash on boot.
4. In the service's **Environment**, add:
   - `ANTHROPIC_API_KEY` = your Anthropic key (for query parsing + the summarizer).
5. Click deploy. The first boot builds the image and, unless `vectors.npz` is
   committed (see §3), encodes ~31k vectors — a few minutes. It's ready when
   `https://<your-service>.onrender.com/` returns
   `{"specimens": ..., "semantic_search": true}`. Copy that URL.

## 2. Point the frontend at it (Vercel)

In the **fMuseum** frontend's Vercel project → **Settings → Environment Variables**:

    AI_BACKEND_URL = https://<your-service>.onrender.com

Redeploy the frontend. Its Express proxy now forwards `/api/ai-search` to the
hosted backend instead of `localhost:8001`. (No CORS setup needed: the browser
never calls this service directly — the Express proxy does, server-to-server.)

## 3. `vectors.npz` is committed (full build only)

`vectors.npz` is committed to the repo so a full-build boot **loads** the cached
vectors instead of **re-encoding** all 31k on startup — faster boots, lower peak
memory, no health-check timeout during a slow first encode. It stays correct
automatically: if `enriched.json` changes, the fingerprint mismatch re-encodes it.
(The lexical build never loads it — `Dockerfile.lexical` drops it from the image.)

## 4. Free deploy — the lexical-only build (Render free tier)

This build omits the ML library so it fits **512 MB** and runs on **Render's free
plan** — see the table at the top for the trade-off (keyword search, no semantic
matching). It's a normal web service, not a Blueprint:

1. Push this repo to GitHub.
2. On Render: **New → Web Service → connect this repo.**
3. Set **Language: Docker**, and **Dockerfile Path: `Dockerfile.lexical`**.
4. Choose the **Free** instance type.
5. Add environment variables:
   - `ANTHROPIC_API_KEY` = your key (query parsing + summaries still use Claude).
   - `PROXY_SECRET` = a long random string (also set the same on Vercel — see §2).
6. Deploy. Boot is quick (no model, no vectors). `/` returns
   `{"specimens": ..., "semantic_search": false}` — note `false`, confirming the
   lexical fallback is active. Point Vercel's `AI_BACKEND_URL` at this URL (§2).

> **Memory note:** loading ~31k specimens + enrichments into RAM is close to the
> 512 MB ceiling. It should fit, but if the service OOMs on boot, bump to the next
> instance size (or use the full build on a larger tier).

**Upgrading to semantic later:** switch the service's Dockerfile Path back to
`Dockerfile` and pick a ≥2 GB instance (or move to HF free CPU). No code changes.

## Notes

- **Cost / cold starts:** the Standard plan stays warm. Cheaper sleeping tiers
  cold-start (~a minute) on the first request after idle.
- **Rebuilding data:** `specimens.json` and `enriched.json` ship in the repo, so a
  deploy always has current data. Rebuild them locally (`build_snapshot`, `enrich`)
  and push to update the hosted search.
