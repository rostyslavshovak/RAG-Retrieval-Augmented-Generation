import os
import tempfile
import pytest
from dotenv import load_dotenv

#Automatically load environment variables for tests.
@pytest.fixture(scope="session", autouse=True)
def load_env():
    load_dotenv()

#Fixture to create a temporary PDF file for testing indexing functions.
@pytest.fixture
def temp_pdf_file():
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 55 >>
stream
BT
/F1 12 Tf
72 712 Td
(Hello, this is a test page.) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000010 00000 n 
0000000061 00000 n 
0000000111 00000 n 
0000000178 00000 n 
trailer
<< /Root 1 0 R /Size 5 >>
startxref
256
%%EOF
"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_content)
        tmp_path = tmp.name
    yield tmp_path
    os.remove(tmp_path)