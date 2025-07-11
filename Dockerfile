FROM python:3.9-slim

WORKDIR /app

# Copy your code and requirements
COPY requirements.txt .
COPY inference.py .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Default command to keep container running, override with args when running
CMD ["python", "inference.py", "--help"]
