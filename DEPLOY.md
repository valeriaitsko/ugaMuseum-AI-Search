# Deploying the AI-search backend

Locally you just run `uvicorn app.main:app --port 8001` and the frontend on the
same machine reaches it. For a **hosted** site (a link others can use), the
frontend runs on the internet and can't reach your laptop — so this FastAPI
service has to run somewhere public too. These files deploy it to **Render**; the
`Dockerfile` also works on Railway, Fly, or any Docker host.

## 1. Deploy the backend to Render

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

## 3. (Recommended) Commit `vectors.npz` for fast, reliable boots

`vectors.npz` is gitignored because it's derived. But committing it lets a hosted
boot **load** the vectors instead of **re-encoding** all 31k on startup — much
faster boots, lower peak memory, and no risk of the health check timing out during
a slow first encode. It stays correct automatically: if `enriched.json` later
changes, the fingerprint mismatch triggers a re-encode.

To include it:

```bash
# remove the app/data/vectors.npz line from .gitignore, then:
git add -f app/data/vectors.npz
git commit -m "Commit vectors.npz for hosted deploys"
```

## Notes

- **Cost / cold starts:** the Standard plan stays warm. Cheaper sleeping tiers
  cold-start (~a minute) on the first request after idle.
- **Rebuilding data:** `specimens.json` and `enriched.json` ship in the repo, so a
  deploy always has current data. Rebuild them locally (`build_snapshot`, `enrich`)
  and push to update the hosted search.
