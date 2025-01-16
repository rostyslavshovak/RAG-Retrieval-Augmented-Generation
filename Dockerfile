FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader punkt_tab

COPY . .

EXPOSE 7860

CMD ["python", "gradio_ui.py"]