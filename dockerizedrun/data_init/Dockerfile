FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ ./common/
COPY data_init/download_data.py ./download_data.py

CMD ["python", "download_data.py"]
