FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MLFLOW_DISABLE_TELEMETRY=true DO_NOT_TRACK=true \
    HOME=/tmp
RUN useradd --uid 10001 --create-home app
WORKDIR /app
COPY serving/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY serving/server.py /app/server.py
USER 10001
EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--no-access-log"]
