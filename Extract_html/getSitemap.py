import os
import time
import json
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import google.generativeai as genai

# -------------------------------
# 1. Configure Gemini API
# -------------------------------
genai.configure(api_key="AIzaSyAq4ErUni6wtiM-RYFvq0mgFN0U8a1YAEA")

# Make sure the raw_pages folder exists
os.makedirs("raw_pages", exist_ok=True)

# -------------------------------
# 2. Summarize content using Gemini
# -------------------------------
def summarize_with_gemini(text, url):
    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        prompt = f"""
        Summarize the following webpage content clearly and concisely for a sitemap entry.
        Include: page purpose, key topics, and what the user can learn from it.

        URL: {url}

        Content:
        {text[:8000]}  # limit for safety
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print("❌ Summarization error:", e)
        return "Summary not available."

# -------------------------------
# 3. Extract main page content
# -------------------------------
def extract_main_content(html):
    soup = BeautifulSoup(html, "html.parser")

    # Extract basic metadata
    title = soup.title.string.strip() if soup.title else ""
    meta_desc = soup.find("meta", attrs={"name": "description"})
    description = meta_desc["content"].strip() if meta_desc else ""

    # Remove unnecessary tags
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    paragraphs = soup.find_all("p")
    headings = soup.find_all(["h1", "h2", "h3"])
    body_text = " ".join([p.get_text(strip=True) for p in paragraphs])
    headings_text = " ".join([h.get_text(strip=True) for h in headings])

    return title, description, headings_text, body_text

# -------------------------------
# 4. Parse sitemap.xml
# -------------------------------
tree = ET.parse("sitemap.xml")
root = tree.getroot()
namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}

urls = [url.find("ns:loc", namespace).text for url in root.findall("ns:url", namespace)]
print(f"✅ Found {len(urls)} URLs in sitemap.xml")

results = []

# -------------------------------
# 5. Crawl and summarize each URL
# -------------------------------
for i, url in enumerate(urls, 1):
    print(f"[{i}/{len(urls)}] Crawling: {url}")

    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()

        # Extract and summarize content
        title, description, headings, body = extract_main_content(r.text)

        # Build final content structure (with URL at top)
        content = (
            f"URL: {url}\n"
            f"Title: {title}\n"
            f"Description: {description}\n"
            f"Headings: {headings}\n"
            f"Body: {body}"
        )

        summary = summarize_with_gemini(content, url)

        results.append({
            "url": url,
            "summary": summary,
            "raw_text": content
        })

        # Save structured text
        with open(f"raw_pages/page_{i}.txt", "w", encoding="utf-8") as f:
            f.write(content)

        time.sleep(2)  # polite delay

    except Exception as e:
        print(f"❌ Failed to process {url}: {e}")
        continue

# -------------------------------
# 6. Save all results as JSON
# -------------------------------
with open("sitemap_db.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("✅ Sitemap processing complete! Data saved to sitemap_db.json and raw_pages/")
