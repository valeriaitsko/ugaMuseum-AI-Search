# Museum AI Search -- FastAPI backend, containerised for a hosted deploy
# (Render, Railway, Fly, any Docker host). Locally you still just run uvicorn;
# this is only for running the search somewhere a deployed frontend can reach.

FROM python:3.12-slim

WORKDIR /app

# Install the CPU-only build of torch FIRST, from PyTorch's CPU wheel index. The
# default `torch` pulls ~2.5 GB of CUDA libraries this app never uses on a CPU
# host -- installing the CPU wheel first means the sentence-transformers install
# below sees torch already satisfied and won't drag them in.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so it is NOT downloaded on first
# request (faster, and no dependency on the HF hub being reachable at runtime).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# The app + its committed data (specimens.json, enriched.json). vectors.npz is
# gitignored, so unless it's committed the index re-encodes on first boot (a few
# minutes) -- see DEPLOY.md to commit it and make boots fast.
COPY app ./app

# Hosts inject $PORT; bind 0.0.0.0 so the service is reachable. main.py loads the
# index and warms the query encoder at startup, so the first real search is quick.
ENV PORT=8001
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
