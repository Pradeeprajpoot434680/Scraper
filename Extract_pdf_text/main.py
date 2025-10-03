# main.py
from scrape import run_scrape
from extract_and_classify import run_extract
from cleaning_text import run_cleaning
from llama_local import run_llama  # use the file name you saved (llama_local.py or run_lamma_locally.py)

def main():
    print("\n========== STEP 1: SCRAPE ==========")
    # run_scrape()
    print("\n========== STEP 2: EXTRACT & CLASSIFY ==========")
    run_extract()
    print("\n========== STEP 3: CLEAN TEXT ==========")
    # run_cleaning()
    print("\n========== STEP 4: INDEX & QUERY ==========")
    run_llama()
    print("\n========== ALL DONE ==========")

if __name__ == "__main__":
    main()
