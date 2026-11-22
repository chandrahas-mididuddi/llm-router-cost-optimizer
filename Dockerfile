FROM python:3.11-slim

WORKDIR /app

# scikit-learn needs a C toolchain only at build time for wheels it
# cannot fetch prebuilt; slim images ship without it, so install it
# briefly and remove it again to keep the image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential && apt-get autoremove -y

COPY config/ ./config/
COPY data/ ./data/
COPY src/ ./src/

ENV PYTHONPATH=/app/src
ENV ROUTER_BACKEND=mock
ENV COST_DB=/app/data/costs.db

EXPOSE 8000

CMD ["uvicorn", "llm_router.api:app", "--host", "0.0.0.0", "--port", "8000"]
