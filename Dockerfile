# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Add FastAPI and uvicorn for API
RUN pip install --no-cache-dir fastapi uvicorn[standard] python-multipart

# Copy application code
COPY . .

# Create directory for temporary files
RUN mkdir -p /tmp/credentials

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Run the FastAPI application
CMD exec uvicorn api:app --host 0.0.0.0 --port $PORT