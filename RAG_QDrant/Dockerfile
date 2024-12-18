FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "rag_script.naive_rag", "--pdf_path", "C:/Users/Admin/Desktop/ammazon_annual.pdf", "--query", "What position in the company does Jeffrey P. Bezos hold and since when?"]