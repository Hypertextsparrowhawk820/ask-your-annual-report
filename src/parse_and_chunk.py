# src/parse_and_chunk.py
import os
import pymupdf
from langchain.text_splitter import RecursiveCharacterTextSplitter

FILINGS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sec_filings")

# Map filename keywords to company names
COMPANY_MAP = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "TSLA": "Tesla"
}

def extract_text_from_pdf(pdf_path):
    """Extract text page by page using PyMuPDF."""
    doc = pymupdf.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if len(text.strip()) > 100:   # skip mostly-empty pages
            pages.append({
                "text": text,
                "page": page_num
            })
    doc.close()
    return pages

def get_company_and_year(filename):
    """Extract company ticker and year from filename like AAPL_10K_2025.pdf"""
    parts = filename.replace(".pdf", "").split("_")
    ticker = parts[0]
    year = parts[2] if len(parts) >= 3 else "unknown"
    company = COMPANY_MAP.get(ticker, ticker)
    return company, ticker, year

def chunk_pages(pages, company, ticker, year):
    """Split pages into chunks and attach metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )

    all_chunks = []
    for page_data in pages:
        chunks = splitter.split_text(page_data["text"])
        for chunk in chunks:
            if len(chunk.strip()) < 50:   # skip tiny chunks
                continue
            all_chunks.append({
                "text": chunk,
                "metadata": {
                    "company": company,
                    "ticker": ticker,
                    "year": year,
                    "page": page_data["page"],
                    "source": f"{ticker}_10K_{year}.pdf"
                }
            })
    return all_chunks

def process_all_filings():
    all_chunks = []
    pdf_files = [f for f in os.listdir(FILINGS_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDFs found in", FILINGS_DIR)
        return []

    for filename in pdf_files:
        pdf_path = os.path.join(FILINGS_DIR, filename)
        company, ticker, year = get_company_and_year(filename)

        print(f"Processing {filename}...")
        pages = extract_text_from_pdf(pdf_path)
        print(f"  Extracted {len(pages)} pages")

        chunks = chunk_pages(pages, company, ticker, year)
        print(f"  Created {len(chunks)} chunks")

        all_chunks.extend(chunks)

    print(f"\nTotal chunks across all filings: {len(all_chunks)}")
    return all_chunks

if __name__ == "__main__":
    chunks = process_all_filings()

    # Preview 3 random chunks to verify quality
    import random
    print("\n--- Sample chunks ---")
    for chunk in random.sample(chunks, min(3, len(chunks))):
        print(f"\nCompany: {chunk['metadata']['company']} | Year: {chunk['metadata']['year']} | Page: {chunk['metadata']['page']}")
        print(f"Text preview: {chunk['text'][:200]}...")
        print("-" * 60)