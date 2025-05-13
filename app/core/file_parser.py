import PyPDF2
from docx import Document
from typing import List, Tuple, Dict

def pares_pdf(file_path: str):
    chunks_with_pages = []
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            chunks_with_pages.append({
                "text" : page.extract_text() or "",
                "page_number" : page_num + 1
            })
        return chunks_with_pages

def pares_docx(file_path: str):
    doc = Document(file_path)
    return [{"text": p.text, "page_number" : None} for p in doc.paragraphs if p.text.strip()]

def parse_txt(file_path : str):
    with open(file_path,"r", encoding="utf-8") as f:
        return [{"text" : f.read(), "page_number" : None}]
