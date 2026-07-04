# Moderation service image.
# Build from the repo root:
#   docker build -f docker/Dockerfile -t moderation-service .
# Run offline (stub classifier + mock LLM):
#   docker run -p 8000:8000 moderation-service
# Run with a real checkpoint + LLM:
#   docker run -p 8000:8000 \
#     -v $(pwd)/outputs/davidson/RoBERTa-base/3class:/model \
#     -e MODEL_DIR=/model -e MODEL_NAME=roberta-base \
#     -e LLM_BASE_URL=http://host.docker.internal:11434/v1 -e LLM_MODEL=llama3.1 \
#     moderation-service

FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image several GB smaller than the default CUDA wheels.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Serving code + the research package it wraps.
COPY serving/ serving/
COPY rag/ rag/
COPY policies/ policies/
COPY evals/ evals/
COPY hs_generalization/ hs_generalization/
COPY datasets/hatecheck-xr/ datasets/hatecheck-xr/

# Pre-build the policy index and warm the embedding model cache at build time
# so the container serves without network at request time.
RUN python -c "from serving.config import ServingConfig; from rag.policy_store import PolicyStore; \
c = ServingConfig.load(); PolicyStore(c.policy_path, c.chroma_dir, c.collection_name)"

EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
