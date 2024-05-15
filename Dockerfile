# Base image
FROM python:3.12.3-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# RUN apt-get update && apt-get install -y \
#     python3-dev \
#     gcc \
#     build-essential \
#     libpq-dev \
#     && rm -rf /var/lib/apt/lists/*
RUN apt-get update --fix-missing && apt-get install -y --fix-missing build-essential

# Install dependencies
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir fastapi
RUN pip3 install --no-cache-dir langchain_community langchain_text_splitters langchain langchain_core
RUN pip3 install --no-cache-dir cohere boto3 tiktoken gpt4all bert-extractive-summarizer
RUN pip3 install --no-cache-dir python-dotenv jq scikit-learn
RUn pip3 install --no-cache-dir chromadb
RUN pip3 install --no-cache-dir numpy 
RUN pip3 install --no-cache-dir uvicorn
# RUN pip3 install --no-cache-dir -r requirements.txt

# Copy server code
COPY . .

# # Expose port
# EXPOSE 8000

# Run server
CMD ["fastapi", "run", "app/main.py", "--port", "8000"]