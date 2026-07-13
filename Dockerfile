FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p models logs artifacts mlruns

EXPOSE 5000

CMD ["python", "pipeline.py"]
