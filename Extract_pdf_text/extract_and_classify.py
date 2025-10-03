import os
import shutil
import subprocess
import signal
from typing import Optional

import fitz  # PyMuPDF
from transformers import pipeline

# -------- CONFIG --------
INPUT_DIR = "./sorted/pdfs"
REPAIRED_TEXT_DIR = "./sorted_data/repaired_texts"
OCR_TEXT_DIR = "./sorted_data/ocr_texts"
REPAIRED_PDF_DIR = "./sorted_data/repaired"
ERROR_DIR = "./sorted_data/error"
LIMIT = None  # set a number to limit files for testing

# ------------------ Zero-Shot Classifier ------------------
print("🔎 Loading zero-shot classification model (facebook/bart-large-mnli)...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
LABELS = ["static data", "dynamic data"]

def classify_text(text: str) -> str:
    if not text.strip():
        return "unknown"
    try:
        result = classifier(text, LABELS, multi_label=False)
        return result["labels"][0]
    except Exception as e:
        print(f"[WARN] classification failed: {e}. Defaulting to 'dynamic data'.")
        return "dynamic data"

# ------------------ OCR with Timeout ------------------
class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()

def ocr_extract(pdf_path: str, ocr_dir: str, reader, dpi: int = 200, timeout_per_page: int = 60) -> Optional[str]:
    os.makedirs(ocr_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    txt_path = os.path.join(ocr_dir, f"{base}_ocr.txt")

    try:
        with fitz.open(pdf_path) as doc, open(txt_path, "w", encoding="utf-8") as txt_file:
            for page_num in range(doc.page_count):
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout_per_page)
                try:
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap(dpi=dpi)
                    ocr_result = reader.readtext(pix.tobytes(), detail=0, paragraph=True)
                    txt_file.write(f"Page {page_num + 1}:\n")
                    txt_file.write(" ".join(ocr_result) + "\n")
                    txt_file.write("-" * 80 + "\n")
                except TimeoutException:
                    print(f"[WARN] OCR timeout on page {page_num + 1} of {base}")
                    txt_file.write(f"Page {page_num + 1}: ⚠️ OCR timeout\n")
                    txt_file.write("-" * 80 + "\n")
                except Exception as e:
                    print(f"[WARN] OCR failed on page {page_num + 1} of {base}: {e}")
                    txt_file.write(f"Page {page_num + 1}: ⚠️ OCR failed\n")
                    txt_file.write("-" * 80 + "\n")
                finally:
                    signal.alarm(0)
        return txt_path
    except Exception as e:
        print(f"[ERROR] OCR completely failed for {base}: {e}")
        return None

# ------------------ PDF Processing ------------------
def is_digital(pdf_path: str) -> bool:
    try:
        with fitz.open(pdf_path) as doc:
            text_pages = sum(1 for page in doc if page.get_text().strip())
            return text_pages >= max(1, doc.page_count // 2)
    except Exception:
        return False

def ghostscript_repair(pdf_path: str, out_dir: str) -> Optional[str]:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(pdf_path))
    gs_candidates = ["gswin64c", "gswin32c", "gs"]
    gs = None
    for cand in gs_candidates:
        try:
            subprocess.run([cand, "-v"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            gs = cand
            break
        except Exception:
            continue
    if gs is None:
        print("[INFO] Ghostscript not found; skipping repair.")
        return None
    try:
        subprocess.run([gs, "-o", out_path, "-sDEVICE=pdfwrite", "-dPDFSETTINGS=/prepress", pdf_path],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_path
    except Exception as e:
        print(f"[WARN] Ghostscript repair failed for {pdf_path}: {e}")
        return None

def extract_text_from_pdf(pdf_path: str) -> str:
    texts = []
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                texts.append(page.get_text())
    except Exception as e:
        print(f"[WARN] Failed to extract text from {pdf_path}: {e}")
    return "\n".join(texts)

def save_text(text: str, out_dir: str, base_filename: str):
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{base_filename}.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(text)
    return out_file

def process_pdf(pdf_path: str, repaired_text_dir: str, ocr_text_dir: str, repaired_pdf_dir: str, error_dir: str, reader=None):
    base = os.path.basename(pdf_path)
    name_no_ext = os.path.splitext(base)[0]
    try:
        print(f"🔄 Processing: {base}")
        digital = is_digital(pdf_path)
        if digital:
            print("📄 Detected as digital PDF — extracting text.")
            repaired_pdf = ghostscript_repair(pdf_path, repaired_pdf_dir)
            source_pdf_for_extract = repaired_pdf if repaired_pdf else pdf_path
            text = extract_text_from_pdf(source_pdf_for_extract)
            if not text.strip() and reader:
                print("[WARN] Text empty — attempting OCR fallback.")
                ocr_path = ocr_extract(pdf_path, ocr_text_dir, reader)
                text = ""
                if ocr_path:
                    with open(ocr_path, "r", encoding="utf-8") as fo:
                        text = fo.read()
            txt_path = save_text(text, repaired_text_dir, name_no_ext)
            label = classify_text(text)
            classified_dir = os.path.join(repaired_pdf_dir, label.replace(" ", "_"))
            os.makedirs(classified_dir, exist_ok=True)
            shutil.copy(pdf_path, os.path.join(classified_dir, base))
            print(f"✅ Digital processed — text saved to {txt_path} and PDF copied to {classified_dir}")
        else:
            print("🖼️ Detected as scanned PDF — running OCR.")
            if reader is None:
                import easyocr
                reader = easyocr.Reader(["hi", "en"], gpu=False)
            ocr_txt = ocr_extract(pdf_path, ocr_text_dir, reader)
            if not ocr_txt:
                print(f"[ERROR] OCR failed for {base}. Moving to error dir.")
                os.makedirs(error_dir, exist_ok=True)
                shutil.copy(pdf_path, os.path.join(error_dir, base))
                return
            with open(ocr_txt, "r", encoding="utf-8") as fo:
                text = fo.read()
            label = classify_text(text)
            classified_dir = os.path.join(ocr_text_dir, label.replace(" ", "_"))
            os.makedirs(classified_dir, exist_ok=True)
            shutil.move(ocr_txt, os.path.join(classified_dir, os.path.basename(ocr_txt)))
            print(f"✅ OCR processed — saved to {classified_dir}")
    except Exception as e:
        print(f"[ERROR] Processing failed for {base}: {e}")
        os.makedirs(error_dir, exist_ok=True)
        try:
            shutil.copy(pdf_path, os.path.join(error_dir, base))
        except Exception:
            pass

# ------------------ Main ------------------
def main(input_dir=INPUT_DIR, repaired_text_dir=REPAIRED_TEXT_DIR, ocr_text_dir=OCR_TEXT_DIR,
         repaired_pdf_dir=REPAIRED_PDF_DIR, error_dir=ERROR_DIR, limit=LIMIT):

    if not os.path.isdir(input_dir):
        print(f"[WARN] No PDFs to process at {input_dir}")
        return

    os.makedirs(repaired_text_dir, exist_ok=True)
    os.makedirs(ocr_text_dir, exist_ok=True)
    os.makedirs(repaired_pdf_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    # Initialize EasyOCR lazily
    reader = None

    files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith(".pdf")]
    if limit:
        files = files[:limit]
    if not files:
        print(f"[INFO] No PDF files found in {input_dir}.")
        return

    print(f"[INFO] Found {len(files)} PDFs — starting processing.")
    for i, fname in enumerate(files, 1):
        path = os.path.join(input_dir, fname)
        print(f"[{i}/{len(files)}] {fname}")
        try:
            process_pdf(path, repaired_text_dir, ocr_text_dir, repaired_pdf_dir, error_dir, reader=reader)
        except Exception as e:
            print(f"[ERROR] Fatal error on {fname}: {e}")
            os.makedirs(error_dir, exist_ok=True)
            shutil.copy(path, os.path.join(error_dir, fname))
            continue

    print("✅ All PDF processing complete.")

def run_extract():
    main()


if __name__ == "__main__":
    run_extract()
