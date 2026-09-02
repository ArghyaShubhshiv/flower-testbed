FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py client.py data_utils.py ./

# Default command – will be overridden by docker-compose
CMD ["python", "client.py"]