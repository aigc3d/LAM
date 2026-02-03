FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Cloud Run uses PORT environment variable
ENV PORT=8080
ENV HOST=0.0.0.0

# Run the application
CMD ["sh", "-c", "python app.py --host ${HOST} --port ${PORT}"]
