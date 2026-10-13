# Hosted backup deployment (Render / Railway / any Docker host).
# Uses Gemini embeddings so the image does not need torch (~1.5 GB smaller). Set GEMINI_API_KEY in the host's dashboard.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 PORT=8000 \
    EMBED_PROVIDER=gemini LLM_PROVIDER=gemini LLM_FALLBACKS=extractive NO_BROWSER=1

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY app ./app
COPY static ./static
COPY scripts ./scripts
COPY run.py .

EXPOSE 8000
# Render/Railway inject $PORT; config.py reads it.
CMD ["python", "run.py", "--no-browser"]
