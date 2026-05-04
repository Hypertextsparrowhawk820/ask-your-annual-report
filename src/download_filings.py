# src/download_filings.py
import os
import requests

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sec_filings")

# paste your Google Drive file IDs here after uploading
FILINGS = {
    "AAPL_10K_2025.pdf": "https://drive.google.com/file/d/1ZLCqDvVvGL_3KaZP1AO1g3De3lEaykqt/view?usp=sharing",
    "MSFT_10K_2025.pdf": "https://drive.google.com/file/d/1RKI7TNIlpPdno8xxVPP0mMtxNWyAM-So/view?usp=sharing",
    "TSLA_10K_2025.pdf": "https://drive.google.com/file/d/1ILeeaW_yXSsew4x09jpNa8nFfjzSMeR1/view?usp=sharing",
}

def download_from_drive(file_id, dest_path):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()
    response = session.get(url, stream=True)

    # handle Google's virus scan warning for large files
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={value}"
            response = session.get(url, stream=True)
            break

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

def download_all_filings():
    os.makedirs(SAVE_DIR, exist_ok=True)
    for filename, file_id in FILINGS.items():
        dest = os.path.join(SAVE_DIR, filename)
        if not os.path.exists(dest):
            print(f"Downloading {filename}...")
            download_from_drive(file_id, dest)
            print(f"  Saved to {dest}")
        else:
            print(f"  {filename} already exists, skipping")

if __name__ == "__main__":
    download_all_filings()