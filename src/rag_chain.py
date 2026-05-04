# src/rag_chain.py
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()

VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")

PROMPT_TEMPLATE = """
You are a financial analyst assistant. Answer the question using ONLY
the context provided from SEC 10-K filings below.

Rules:
- Cite the company name and page number for every fact you state
- If the answer is not in the context, say "I don't have enough information in the provided filings to answer this."
- Be concise and factual — no speculation

Context:
{context}

Question: {question}

Answer:
"""

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

def build_rag_chain(vectorstore, company_filter=None):
    """Build RAG chain, optionally filtered by company."""

    # metadata filter
    search_kwargs = {"k": 5}
    if company_filter:
        search_kwargs["filter"] = {"company": company_filter}

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain

def ask(chain, question):
    """Run a question and print answer with sources."""
    result = chain.invoke({"query": question})

    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{result['result']}")
    print("\nSources:")
    seen = set()
    for doc in result["source_documents"]:
        key = f"{doc.metadata['company']} | Page {doc.metadata['page']}"
        if key not in seen:
            print(f"  - {key} ({doc.metadata['source']})")
            seen.add(key)
    return result

if __name__ == "__main__":
    print("Loading vectorstore...")
    vectorstore = load_vectorstore()
    print("Building RAG chain...")
    chain = build_rag_chain(vectorstore)

    # test questions
    questions = [
        "What are Tesla's main risk factors?",
        "What was Apple's revenue in 2025?",
        "How does Microsoft describe its AI strategy?"
    ]

    for q in questions:
        ask(chain, q)
        print("\n" + "="*60)