# app/startup.py
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")

def index_exists():
    return (
        os.path.exists(os.path.join(VECTORSTORE_DIR, "index.faiss")) and
        os.path.exists(os.path.join(VECTORSTORE_DIR, "index.pkl"))
    )