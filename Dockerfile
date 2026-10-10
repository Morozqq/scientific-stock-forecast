FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
RUN mkdir -p /app/outputs

ENTRYPOINT ["python", "-m", "src.forecast"]
CMD ["--ticker", "AAPL", "--period", "5y", "--output", "/app/outputs"]
