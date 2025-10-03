import os
import shutil
import subprocess
from dotenv import load_dotenv

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

def run_scrape():
    # Website URL
    URL = "https://curaj.ac.in/"
    BASE_DIR = "curaj.ac.in"
    SORTED_DIR = "sorted"

    # Step 1: Download website with a 10-minute timeout (600 seconds)
    print("Downloading website data for max 10 minutes...")

    try:
        subprocess.run(
            [
                "wget",
                "--mirror",
                "--convert-links",
                "--adjust-extension",
                "--page-requisites",
                "--no-parent",
                "-e",
                "robots=on",
                URL,
            ],
            timeout=900  # 
        )
    except subprocess.TimeoutExpired:
        print("⏱️ Download stopped after 10 minutes.")

    # Step 2: Create sorted folder
    os.makedirs(SORTED_DIR, exist_ok=True)

    # Step 3: Function to move files by extension if count > 10
    def move_files(exts, folder):
        files = []
        for root, _, filenames in os.walk(BASE_DIR):
            for fname in filenames:
                if any(fname.lower().endswith(f".{ext}") for ext in exts):
                    files.append(os.path.join(root, fname))

        count = len(files)
        if count > 10:
            target_dir = os.path.join(SORTED_DIR, folder)
            os.makedirs(target_dir, exist_ok=True)
            print(f"Found {count} {exts} files → moving to {target_dir}/")
            for f in files:
                try:
                    shutil.move(f, target_dir)
                except shutil.Error:
                    pass  # ignore if already moved
        else:
            print(f"Only {count} {exts} files → skipping.")

    # Step 4: Call function for major file types
    move_files(["pdf"], "pdfs")
    move_files(["html"], "html")
    move_files(["jpg", "jpeg", "png", "gif", "svg"], "images")
    move_files(["doc", "docx", "xlsx", "txt"], "docs")
    move_files(["css"], "css")
    move_files(["js"], "js")

    print("✅ Sorting completed!")

if __name__ == "__main__":
    run_scrape()
