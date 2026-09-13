FROM python:3.11.11-slim-bookworm
WORKDIR /srv
COPY requirements.txt requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c requirements-lock.txt
COPY app ./app
RUN useradd --uid 10001 --create-home app && mkdir -p /var/log/inference && chown app:app /var/log/inference
USER app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
