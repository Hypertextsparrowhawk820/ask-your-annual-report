# src/retriever.py
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

def get_retriever(vectorstore, companies=None, k=5):
    """
    Return a retriever filtered by company list.
    companies = None  → search all companies
    companies = ["Apple"]  → single company
    companies = ["Apple", "Microsoft"]  → comparison mode
    """
    search_kwargs = {"k": k}

    if companies and len(companies) == 1:
        search_kwargs["filter"] = {"company": companies[0]}

    return vectorstore.as_retriever(search_kwargs=search_kwargs)

def get_comparison_docs(vectorstore, question, companies, k=4):
    """
    For comparison mode — retrieve top-k docs from EACH company separately
    and combine them so the LLM sees context from all selected companies.
    """
    all_docs = []
    for company in companies:
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": k,
                "filter": {"company": company}
            }
        )
        docs = retriever.invoke(question)
        all_docs.extend(docs)
    return all_docs