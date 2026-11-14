FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY . .
RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
# One worker on purpose: each visitor's agent lives in this process's memory.
CMD ["sh", "-c", "uvicorn backend:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
