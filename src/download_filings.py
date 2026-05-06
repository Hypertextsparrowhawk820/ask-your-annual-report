# src/download_filings.py
import os
import re
import requests

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sec_filings")

# extracted file IDs from your Google Drive URLs
FILINGS = {
    "AAPL_10K_2025.pdf": "1ZLCqDvVvGL_3KaZP1AO1g3De3lEaykqt",
    "MSFT_10K_2025.pdf": "1RKI7TNIlpPdno8xxVPP0mMtxNWyAM-So",
    "TSLA_10K_2025.pdf": "1ILeeaW_yXSsew4x09jpNa8nFfjzSMeR1",
}

def download_from_drive(file_id, dest_path):
    """Download from Google Drive handling large file confirmation."""
    session = requests.Session()

    url = "https://drive.google.com/uc"
    params = {"export": "download", "id": file_id}
    response = session.get(url, params=params, stream=True)

    # handle virus scan warning for large files (cookie method)
    confirm_token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            confirm_token = value
            break

    # fallback — check HTML content for confirm token
    if not confirm_token:
        match = re.search(rb"confirm=([0-9A-Za-z_\-]+)", response.content)
        if match:
            confirm_token = match.group(1).decode()

    if confirm_token:
        params["confirm"] = confirm_token
        response = session.get(url, params=params, stream=True)

    # write file
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

    # verify it's a real PDF (not an HTML error page)
    size = os.path.getsize(dest_path)
    if size < 10000:
        os.remove(dest_path)
        raise ValueError(
            f"Download failed for {dest_path} — "
            f"got {size} bytes. Check file sharing is set to 'Anyone with the link'."
        )

    print(f"  Saved {dest_path} ({size // 1024} KB)")

def download_all_filings():
    os.makedirs(SAVE_DIR, exist_ok=True)
    for filename, file_id in FILINGS.items():
        dest = os.path.join(SAVE_DIR, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 10000:
            print(f"  {filename} already exists, skipping")
            continue
        print(f"Downloading {filename}...")
        download_from_drive(file_id, dest)

if __name__ == "__main__":
    download_all_filings()
    print("\nAll filings downloaded successfully!")
