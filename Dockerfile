# Hugging Face Spaces (Docker SDK). Also runs anywhere Docker does.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as user 1000; give it a home for the model cache.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
# Created as "user" so the app can write sessions/ and ocr_cache/ (WORKDIR alone makes it root-owned).
RUN mkdir -p /home/user/app
WORKDIR /home/user/app

# CPU-only torch keeps the image small (the default wheel bundles CUDA).
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt requirements_b.txt requirements_web.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt -r requirements_b.txt -r requirements_web.txt \
    && python -m spacy download en_core_web_sm

# Download the embedding and reranker models now, not on the first question.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY --chown=user . .

# Build the built-in library's indexes from documents/ (if any were uploaded).
RUN if [ -d documents ] && [ ! -d Documents ]; then mv documents Documents; fi; \
    if [ -n "$(ls -A Documents 2>/dev/null)" ]; then python ingest_documents.py --force; fi

ENV HOST=0.0.0.0 PORT=7860
EXPOSE 7860
CMD ["python", "web_app.py"]
