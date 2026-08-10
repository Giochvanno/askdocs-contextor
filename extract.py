"""
extract.py - extract plain text from an uploaded document.

Works with both a file path and a file-like object
(e.g. the object returned by the Streamlit uploader).
"""

from pypdf import PdfReader


def extract_text(file, filename: str) -> str:
    """
    file     - a path (str) or a file-like object (BytesIO/UploadedFile).
    filename - the file name, used to detect the format.
    """
    ext = filename.lower().rsplit(".", 1)[-1]

    if ext == "pdf":
        return _from_pdf(file)
    if ext == "docx":
        return _from_docx(file)
    if ext in ("txt", "md"):
        return _from_txt(file)

    raise ValueError(f"Unsupported format: .{ext} (supported: pdf, docx, txt, md)")


def _from_pdf(file) -> str:
    reader = PdfReader(file)
    parts = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            parts.append(f"[page {i}]\n{text}")
    return "\n\n".join(parts)


def _from_docx(file) -> str:
    from docx import Document
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _from_txt(file) -> str:
    if isinstance(file, str):
        with open(file, "rb") as f:
            data = f.read()
    else:
        data = file.read()
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="ignore")
    return data


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (~4 characters per token)."""
    return len(text) // 4