import fitz
import logging
import json

def parse_pdf(pdf_path, output_json_path="book.json"):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logging.error(f"Failed to open PDF '{pdf_path}': {e}")
        return None

    pages_dict = {}
    for i, page in enumerate(doc):
        try:
            text = page.get_text()
            if not text.strip():
                raise ValueError("Empty text")
            pages_dict[str(i + 1)] = text
        except Exception as e:
            error_msg = "[Error: Could not parse this page]"
            pages_dict[str(i + 1)] = error_msg
            logging.warning(f"Could not parse page {i + 1}: {e}")

    # Save as JSON
    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(pages_dict, json_file, ensure_ascii=False, indent=2)

    logging.info(f"Parsing complete. JSON saved to '{output_json_path}'.")
    return pages_dict
