# CURAJ Data Processing Pipeline
**Automated Web Scraping, PDF Extraction, Cleaning, and LLM-based Indexing**

---

## Overview

This project is an **end-to-end pipeline** designed to **scrape**, **extract**, **clean**, and **index** data from the [Central University of Rajasthan](https://curaj.ac.in/) website. The pipeline is modular and orchestrated by the `main.py` script, and consists of the following four main steps:

1. **Scrape Website & Sort Files**
2. **Extract & Classify PDF Content (OCR Included)**
3. **Clean Text**
4. **Index Content & Query using LLM + Weaviate**

---

## Directory Structure

```plaintext
project/
├── main.py                  # Main orchestrator script
├── scrape.py                # Web scraping & file sorting
├── extract_and_classify.py  # PDF extraction, OCR, and classification
├── cleaning_text.py         # Text cleaning & normalization
├── llama_local.py           # LlamaIndex + Weaviate indexing & query
├── requirements.txt         # Python dependencies
├── .env                     # API keys and Weaviate URL
├── sorted/                  # Sorted files (PDFs, HTML, images)
├── sorted_data/             # Processed text files
│   ├── repaired_texts/      # Repaired PDF text
│   ├── ocr_texts/           # OCR extracted text
│   └── cleaned_texts/       # Final cleaned text
└── raw_pages/               # Raw HTML pages saved during scraping
