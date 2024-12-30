# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt requirements.txt

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# 1) Install CPU-only Torch:
#RUN pip install --no-cache-dir torch==2.0.0+cpu -f https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch==2.0.0 --extra-index-url https://download.pytorch.org/whl/cpu

# 2) Install the rest
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your files
COPY . .

EXPOSE 7860
CMD ["python", "gradio_ui.py"]
