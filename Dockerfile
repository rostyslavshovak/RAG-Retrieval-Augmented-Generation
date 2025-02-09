FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt_tab

COPY src/ /app/src/

EXPOSE 7861

CMD ["python", "-m", "src.gradio_ui"]