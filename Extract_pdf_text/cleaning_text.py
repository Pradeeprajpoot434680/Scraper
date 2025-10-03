# import os
# import shutil
# import tempfile

# def clean_file_in_place(filepath):
#     """
#     Clean a single text file in place:
#     - Replace header blocks (--- or Page X) with '#'
#     - Remove empty lines
#     """
#     temp_fd, temp_path = tempfile.mkstemp()
    
#     try:
#         with os.fdopen(temp_fd, 'w', encoding='utf-8') as temp_file:
#             with open(filepath, 'r', encoding='utf-8') as original_file:
#                 is_header_block = False
#                 for line in original_file:
#                     stripped_line = line.strip()
                    
#                     if '---' in stripped_line or stripped_line.startswith('Page '):
#                         if not is_header_block:
#                             temp_file.write('#\n')
#                             is_header_block = True
#                     elif stripped_line:
#                         temp_file.write(line)
#                         is_header_block = False
        
#         shutil.move(temp_path, filepath)
#         return True

#     except Exception as e:
#         print(f"Error processing {os.path.basename(filepath)}: {e}")
#         if os.path.exists(temp_path):
#             os.remove(temp_path)
#         return False

# def clean_directory(directory_path):
#     """
#     Clean all .txt files in a directory
#     """
#     if not os.path.isdir(directory_path):
#         print(f"Error: Directory not found at '{directory_path}'")
#         return

#     files_found = 0
#     for filename in os.listdir(directory_path):
#         if filename.endswith(".txt"):
#             files_found += 1
#             full_path = os.path.join(directory_path, filename)
#             clean_file_in_place(full_path)
    
#     if files_found == 0:
#         print("No .txt files were found to clean.")

# def run_cleaning():
#     """
#     Determine the data folder dynamically and clean all text files
#     """
#     # Assume data folder is inside project: project_root/data/texts
#     project_root = os.path.dirname(os.path.abspath(__file__))  # folder of this script
#     target_directory = os.path.join(project_root, 'data', 'texts')

#     print(f"Cleaning text files in: {target_directory}")
#     clean_directory(target_directory)


# cleaning_text.py
import os
import shutil
import tempfile

REPAIRED_TEXT_DIR = "./sorted_data/repaired_texts"
OCR_TEXT_DIR = "./sorted_data/ocr_texts"
CLEANED_DIR = "./sorted_data/cleaned_texts"

def clean_text_content(text: str) -> str:
    # simple cleaning: strip repeated blank lines and header markers,
    # and normalize whitespace
    lines = []
    prev_empty = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if not prev_empty:
                lines.append("")
            prev_empty = True
        else:
            # collapse header markers
            if line.startswith("---") or line.startswith("Page "):
                lines.append("#")
            else:
                lines.append(line)
            prev_empty = False
    return "\n".join(lines).strip() + "\n"

def process_directory(in_dir: str, out_dir: str):
    if not os.path.isdir(in_dir):
        print(f"[WARN] Directory {in_dir} not found — skipping.")
        return 0
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for root, _, files in os.walk(in_dir):
        for fname in files:
            if not fname.lower().endswith(".txt"):
                continue
            src = os.path.join(root, fname)
            rel = os.path.relpath(src, in_dir)
            out_path = os.path.join(out_dir, rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            try:
                with open(src, "r", encoding="utf-8") as fi:
                    text = fi.read()
                cleaned = clean_text_content(text)
                with open(out_path, "w", encoding="utf-8") as fo:
                    fo.write(cleaned)
                print(f"🧹 Cleaned: {src} -> {out_path}")
                count += 1
            except Exception as e:
                print(f"[ERROR] cleaning {src}: {e}")
    return count

def run_cleaning():
    print("🧹 Starting cleaning step...")
    total = 0
    total += process_directory(REPAIRED_TEXT_DIR, CLEANED_DIR)
    total += process_directory(OCR_TEXT_DIR, CLEANED_DIR)
    if total == 0:
        print("[WARN] No text files found to clean.")
    else:
        print(f"✅ Cleaning completed. {total} files cleaned into {CLEANED_DIR}.")
