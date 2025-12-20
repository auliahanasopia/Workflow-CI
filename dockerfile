# Gunakan base image Python
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements dan kode
COPY requirements.txt .
COPY modelling.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Jalankan script saat container start
CMD ["python", "modelling.py"]
