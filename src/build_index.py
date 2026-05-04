# src/build_index.py
import os
import pickle
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from parse_and_chunk import process_all_filings

VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")

def build_index():
    # Step 1 — load all chunks
    print("Loading chunks...")
    chunks = process_all_filings()

    # Step 2 — convert to LangChain Document objects
    print("\nConverting to documents...")
    documents = [
        Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"]
        )
        for chunk in chunks
    ]
    print(f"  {len(documents)} documents ready")

    # Step 3 — load embedding model (downloads once, ~80MB)
    print("\nLoading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("  Model loaded")

    # Step 4 — build FAISS index (this takes 1-2 mins)
    print("\nBuilding FAISS index — this takes 1-2 minutes...")
    vectorstore = FAISS.from_documents(
        tqdm(documents, desc="  Embedding"),
        embeddings
    )

    # Step 5 — save to disk
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"\nFAISS index saved to: {os.path.abspath(VECTORSTORE_DIR)}")

    # Step 6 — quick retrieval test
    print("\nTesting retrieval...")
    results = vectorstore.similarity_search(
        "What are Tesla's main risk factors?", k=3
    )
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Company: {doc.metadata['company']} | Page: {doc.metadata['page']}")
        print(f"  Text: {doc.page_content[:150]}...")

if __name__ == "__main__":
    build_index()