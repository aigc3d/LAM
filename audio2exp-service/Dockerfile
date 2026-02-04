FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Cloud Run uses PORT environment variable
ENV PORT=8080

# Run the application using uvicorn directly
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
