FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN useradd --uid 10001 --create-home app
WORKDIR /app
COPY jobs/requirements.txt /app/jobs/requirements.txt
RUN pip install --no-cache-dir -r /app/jobs/requirements.txt
COPY jobs/ /app/jobs/
USER 10001
ENTRYPOINT ["python", "-m"]
CMD ["jobs.ingest"]
