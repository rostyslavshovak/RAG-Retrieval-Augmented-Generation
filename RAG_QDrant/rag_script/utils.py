import pdfplumber
from langchain.docstore.document import Document

def extract_tables_as_text(pdf_page):
    """Extract tables from a pdf_page object and convert them into a readable text format."""
    tables = pdf_page.extract_tables()
    table_texts = []
    for table in tables:
        # Each table is a list of rows, where each row is a list of cells.
        row_strings = []
        for row in table:
            # Handle None values and join cells with commas
            cleaned_row = [cell if cell is not None else "" for cell in row]
            row_str = ", ".join(cleaned_row)
            row_strings.append(row_str)
        combined_table_text = "\n".join(row_strings)
        table_texts.append(combined_table_text)
    return "\n\n".join(table_texts)


def load_documents_from_pdf(pdf_path):
    """Load and process PDF documents into a list of LangChain Document objects."""
    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            table_text = extract_tables_as_text(page)

            if table_text.strip():
                # Add a header to separate tables from main text
                text += "\n\n[Table Data]\n" + table_text

            if text.strip():
                documents.append(Document(page_content=text))
    return documents