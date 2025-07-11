FROM python:3.9-slim

WORKDIR /app

# Copy code and requirements
COPY requirements.txt .
COPY inference.py .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install wget and unzip, download and unzip Vosk model, then cleanup
RUN apt-get update && apt-get install -y wget unzip && \
    wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip && \
    unzip vosk-model-small-en-us-0.15.zip && \
    rm vosk-model-small-en-us-0.15.zip && \
    apt-get remove -y wget unzip && apt-get autoremove -y && apt-get clean

ENV MODEL_PATH=/app/vosk-model-small-en-us-0.15

CMD ["python", "inference.py", "/app/vosk-model-small-en-us-0.15", "--help"]
