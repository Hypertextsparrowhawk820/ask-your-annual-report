# src/ingest.py
import os
import requests

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sec_filings")

# SEC requires a user-agent header — use your real name/email
HEADERS = {
    "User-Agent": "YourName youremail@gmail.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "efts.sec.gov"
}

COMPANIES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "TSLA": "Tesla"
}

def get_10k_urls(ticker, count=2):
    """Get the URLs of the most recent 10-K filing documents for a ticker."""
    search_url = (
        f"https://efts.sec.gov/LATEST/search-index?q=%2210-K%22"
        f"&dateRange=custom&startdt=2021-01-01&enddt=2024-12-31"
        f"&entity={ticker}&forms=10-K"
    )
    headers = {**HEADERS, "Host": "efts.sec.gov"}
    resp = requests.get(search_url, headers=headers)
    resp.raise_for_status()
    hits = resp.json().get("hits", {}).get("hits", [])
    urls = []
    for hit in hits[:count]:
        accession = hit["_id"].replace("-", "")
        cik = hit["_source"]["entity_id"].lstrip("0")
        doc = hit["_source"]["file_date"]
        filing_url = (
            f"https://www.sec.gov/Archives/edgar/full-index/"
            f"{doc[:4]}/QTR{((int(doc[5:7])-1)//3)+1}/{accession}-index.htm"
        )
        urls.append(filing_url)
    return urls

def download_filings():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("SEC EDGAR requires manual PDF download for clean results.")
    print("Please download 10-K PDFs manually:\n")
    for ticker, name in COMPANIES.items():
        print(f"  {name} ({ticker}):")
        print(f"  https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-K&dateb=&owner=include&count=5")
        print()
    print("Save PDFs to:", os.path.abspath(SAVE_DIR))
    print("Name them: AAPL_10K_2023.pdf, MSFT_10K_2023.pdf, TSLA_10K_2023.pdf")

if __name__ == "__main__":
    download_filings()