"""
File processing utilities for zrag.
"""

from pathlib import Path
from typing import Optional
import re
from pymupdf4llm import to_markdown
import requests
from urllib.parse import urlparse

def extract_text_from_pdf(filepath: Path) -> str:
    try:
        return to_markdown(filepath)
    except Exception as e:
        print(f"    ⚠ Error extracting PDF {filepath.name}: {e}")
        return ""

def extract_text_from_docx(filepath: Path) -> str:
    try:
        from docx import Document
        doc = Document(filepath)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except ImportError:
        print(f"    ⚠ python-docx not installed, cannot process {filepath.name}")
        return ""
    except Exception as e:
        print(f"    ⚠ Error extracting DOCX {filepath.name}: {e}")
        return ""

def fetch_web_content(url: str, timeout: int = 10) -> str:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"    ⚠ Error fetching {url}: {e}")
        return ""

def html_to_markdown(html: str) -> str:
    try:
        from markdownify import markdownify as md
        return md(html)
    except ImportError:
        return re.sub(r'<[^>]+>', '', html)
    except Exception as e:
        return re.sub(r'<[^>]+>', '', html)

def extract_text_from_url(url: str, timeout: int = 10) -> str:
    html = fetch_web_content(url, timeout)
    return html_to_markdown(html) if html else ""

def extract_text_from_file(filepath: Path) -> str:
    from zrag.chunking import get_file_type
    file_type = get_file_type(filepath)
    if file_type == 'pdf':
        return extract_text_from_pdf(filepath)
    elif file_type == 'docx':
        return extract_text_from_docx(filepath)
    else:
        try:
            return filepath.read_text(encoding='utf-8')
        except Exception as e:
            print(f"    ⚠ Error reading {filepath.name}: {e}")
            return ""
