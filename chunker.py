import pymupdf
import pytesseract
from PIL import Image
import io


# Tesseract location
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def extract_text_from_pdf(pdf_path):

    pages = []

    document = pymupdf.open(pdf_path)

    for page_number, page in enumerate(document, start=1):

        # First try normal PDF text extraction
        text = page.get_text().strip()

        # If no text, use OCR
        if not text:

            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(3, 3),
                alpha=False
            )

            image = Image.open(
                io.BytesIO(pix.tobytes("png"))
            )

            text = pytesseract.image_to_string(
                image,
                lang="eng"
            ).strip()

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    document.close()

    return pages


def create_chunks(pdf_path, chunk_size=100, overlap=20):

    pages = extract_text_from_pdf(pdf_path)

    chunks = []

    for page in pages:

        text = " ".join(page["text"].split())

        words = text.split()

        start = 0

        while start < len(words):

            end = start + chunk_size

            chunk_text = " ".join(
                words[start:end]
            )

            if chunk_text.strip():

                chunks.append({
                    "page": page["page"],
                    "text": chunk_text
                })

            start = end - overlap

            if end >= len(words):
                break

    return chunks


# This function is used by rag_engine.py
def process_pdf(pdf_path):

    return create_chunks(pdf_path)


if __name__ == "__main__":

    pdf_path = "data/documents/sample.pdf"

    print("Processing PDF...")

    chunks = process_pdf(pdf_path)

    print("PDF processed successfully!")
    print("Number of chunks:", len(chunks))

    print("\n===== CHUNKS =====")

    for i, chunk in enumerate(chunks, start=1):

        print("\nChunk", i)
        print("Page:", chunk["page"])
        print("Text:", chunk["text"])