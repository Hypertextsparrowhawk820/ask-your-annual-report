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
    print("Converting to documents...")
    documents = [
        Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"]
        )
        for chunk in chunks
    ]
    print(f"  {len(documents)} documents ready")

    # Step 3 — load embedding model
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("Model loaded")

    # Step 4 — build FAISS index
    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Step 5 — save to disk
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)
    print(f"FAISS index saved to: {os.path.abspath(VECTORSTORE_DIR)}")
    return vectorstore

if __name__ == "__main__":
    vs = build_index()

    # quick retrieval test
    print("\nTesting retrieval...")
    results = vs.similarity_search(
        "What are Tesla's main risk factors?", k=3
    )
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Company: {doc.metadata['company']} | Page: {doc.metadata['page']}")
        print(f"  Text: {doc.page_content[:150]}...")
