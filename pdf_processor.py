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

        print(f"Processing page {page_number}...")

        # Try normal text extraction
        text = page.get_text().strip()

        if text:
            print("Normal text found.")

        else:
            print("No normal text found. Running OCR...")

            # Render PDF page as high-resolution image
            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(3, 3),
                alpha=False
            )

            image = Image.open(
                io.BytesIO(pix.tobytes("png"))
            )

            # OCR
            text = pytesseract.image_to_string(
                image,
                lang="eng"
            ).strip()

            if text:
                print("OCR successful.")
            else:
                print("OCR could not find text.")

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    document.close()

    return pages


if __name__ == "__main__":

    pdf_path = "data/documents/sample.pdf"

    print("Reading PDF...")

    pages = extract_text_from_pdf(pdf_path)

    print("\nPDF processed successfully!")
    print("Number of pages with text:", len(pages))

    for page in pages:

        print("\n--------------------")
        print("Page:", page["page"])
        print(page["text"][:1500])