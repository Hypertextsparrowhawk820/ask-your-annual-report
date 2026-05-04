# app/app.py
import os
import sys
import subprocess
from startup import index_exists
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from retriever import load_vectorstore, get_retriever, get_comparison_docs
from router import classify_intent, web_search

# ── page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Ask Your Annual Report",
    page_icon="📊",
    layout="wide"
)

# ── casual triggers ───────────────────────────────────────────
CASUAL_TRIGGERS = [
    "hello", "hi", "hey", "how are you", "good morning", "good evening",
    "good afternoon", "what's up", "whats up", "who are you", "what can you do",
    "what are you", "thank you", "thanks", "bye", "goodbye", "help"
]

CASUAL_RESPONSES = {
    "hello": "Hello! I'm your financial analyst assistant. Ask me anything about Apple, Microsoft, or Tesla's 2025 annual reports — or any general question.",
    "hi": "Hi there! I can answer questions from the 10-K filings or search the web for general queries. What would you like to know?",
    "hey": "Hey! Ask me anything — financials from the 10-K filings or general knowledge from the web.",
    "how are you": "Doing great and ready to help! Ask me about Apple, Microsoft, Tesla — or anything else.",
    "good morning": "Good morning! What can I help you with today?",
    "good evening": "Good evening! What would you like to explore?",
    "good afternoon": "Good afternoon! Ask me anything — financial or general.",
    "what's up": "All good! Ask me about revenues, risks, strategies, or anything else you're curious about.",
    "whats up": "All good! Ask me about revenues, risks, strategies, or anything else you're curious about.",
    "who are you": (
        "I'm a hybrid financial analyst assistant. I can:\n\n"
        "- Answer questions about **Apple, Microsoft & Tesla** from their 2025 SEC 10-K filings\n"
        "- Search the **web** for general knowledge questions\n\n"
        "Try asking: *What was Apple's revenue in 2025?* or *Who is the Prime Minister of India?*"
    ),
    "what can you do": (
        "I can do two things:\n\n"
        "**Financial analysis** — Ask about Apple, Microsoft, or Tesla:\n"
        "- Revenue, profits, risk factors\n"
        "- Business strategies and AI investments\n"
        "- Multi-company comparisons\n\n"
        "**General knowledge** — Ask me anything:\n"
        "- Current events, people, science, sports\n"
        "- I'll search the web and give you a grounded answer"
    ),
    "what are you": "I'm an AI-powered assistant that combines financial analysis from SEC 10-K filings with live web search for general questions.",
    "thank you": "You're welcome! Feel free to ask anything else.",
    "thanks": "Happy to help! Any other questions?",
    "bye": "Goodbye! Come back anytime.",
    "goodbye": "Goodbye! Come back anytime.",
    "help": (
        "Here's what I can do:\n\n"
        "**From 10-K filings:**\n"
        "- *What are Tesla's main risk factors?*\n"
        "- *What was Apple's revenue in 2025?*\n"
        "- *Compare Microsoft and Apple's AI strategy*\n\n"
        "**From the web:**\n"
        "- *Who is the Prime Minister of India?*\n"
        "- *What is the latest news on AI?*\n"
        "- *How does RAG work?*\n\n"
        "Use the sidebar to filter by company or enable comparison mode."
    )
}

def is_casual(text):
    text_lower = text.lower().strip().rstrip("?!.,")
    for trigger in CASUAL_TRIGGERS:
        if trigger in text_lower:
            return True, trigger
    return False, None

def get_casual_response(trigger):
    return CASUAL_RESPONSES.get(
        trigger,
        "I'm here to help! Ask me about the 10-K filings or any general question."
    )

# ── prompts ───────────────────────────────────────────────────
PROMPT_TEMPLATE = """
You are a professional financial analyst assistant. Answer the question using ONLY
the context provided from SEC 10-K filings below.

Rules:
- Cite the company name and page number for every fact you state
- If the answer is not in the context, say "I don't have enough information in the provided filings to answer this."
- Be concise and factual — no speculation
- For comparison questions, clearly structure your answer by company

Context:
{context}

Question: {question}

Answer:
"""

EXAMPLE_QUESTIONS = [
    "What are Tesla's main risk factors?",
    "What was Apple's total revenue in 2025?",
    "How does Microsoft describe its AI strategy?",
    "Compare Apple and Microsoft's cloud strategy",
    "What does Tesla say about EV competition?",
    "Who is the Prime Minister of India?",
    "What is retrieval augmented generation?",
]

COMPANIES = ["Apple", "Microsoft", "Tesla"]

# ── cached resources ──────────────────────────────────────────
@st.cache_resource
def load_resources():
    vectorstore = load_vectorstore()
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY")
    )
    return vectorstore, llm

def build_chain(vectorstore, llm, companies):
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    retriever = get_retriever(
        vectorstore,
        companies=companies if companies else None
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

def run_comparison(vectorstore, llm, question, companies):
    docs = get_comparison_docs(vectorstore, question, companies, k=4)
    context = "\n\n".join([
        f"[{d.metadata['company']} | Page {d.metadata['page']}]\n{d.page_content}"
        for d in docs
    ])
    filled_prompt = PROMPT_TEMPLATE.replace(
        "{context}", context
    ).replace("{question}", question)
    response = llm.invoke([HumanMessage(content=filled_prompt)])
    return response.content, docs

def dedupe_sources(source_docs):
    seen = set()
    sources = []
    for doc in source_docs:
        key = f"{doc.metadata['company']}_{doc.metadata['page']}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "company": doc.metadata["company"],
                "page": doc.metadata["page"],
                "source": doc.metadata["source"]
            })
    return sources

# ── sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("Settings")
    st.markdown("---")

    st.subheader("Companies")
    selected_companies = []
    for company in COMPANIES:
        if st.checkbox(company, value=True):
            selected_companies.append(company)

    st.markdown("---")
    st.subheader("Mode")
    compare_mode = False
    if len(selected_companies) > 1:
        compare_mode = st.toggle("Comparison mode", value=False)
        if compare_mode:
            st.info("Retrieves context from each selected company separately.")

    st.markdown("---")
    st.subheader("Try these")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state["prefill"] = q

    st.markdown("---")
    st.caption("Financial source: SEC EDGAR 10-K · FY2025")
    st.caption("General search: Tavily web search")
    st.caption("Model: Llama 3.3 70B via Groq")

# ── main ──────────────────────────────────────────────────────
st.title("Ask Your Annual Report")
st.caption(
    "Financial insights from Apple, Microsoft & Tesla 10-K filings · "
    "General questions answered from the web"
)

if not selected_companies:
    st.warning("Please select at least one company in the sidebar.")
    st.stop()

# init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            source_type = msg.get("source_type", "none")
            sources = msg.get("sources", [])

            if source_type == "financial" and sources:
                with st.expander("Sources — 10-K filings"):
                    for s in sources:
                        st.markdown(
                            f"- **{s['company']}** · Page {s['page']} · `{s['source']}`"
                        )

            elif source_type == "web" and sources:
                with st.expander("Sources — web search"):
                    for s in sources:
                        st.markdown(f"- [{s['title']}]({s['url']})")

# chat input
user_input = st.chat_input("Ask anything — financials or general knowledge...")

# inject example question from sidebar buttons
if "prefill" in st.session_state and not user_input:
    user_input = st.session_state.pop("prefill")

# ── handle input ──────────────────────────────────────────────
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        # 1 — casual / greeting
        casual, trigger = is_casual(user_input)
        if casual:
            answer = get_casual_response(trigger)
            st.markdown(answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": [],
                "source_type": "none"
            })

        # 2 — financial or general via router
        else:
            with st.spinner("Analyzing..."):
                vectorstore, llm = load_resources()
                intent = classify_intent(user_input, llm)

                # ── financial → RAG ───────────────────────────
                if intent == "financial":
                    if compare_mode and len(selected_companies) > 1:
                        answer, source_docs = run_comparison(
                            vectorstore, llm, user_input, selected_companies
                        )
                    else:
                        chain = build_chain(vectorstore, llm, selected_companies)
                        result = chain.invoke({"query": user_input})
                        answer = result["result"]
                        source_docs = result["source_documents"]

                    st.markdown(answer)
                    sources = dedupe_sources(source_docs)

                    if sources:
                        with st.expander("Sources — 10-K filings"):
                            for s in sources:
                                st.markdown(
                                    f"- **{s['company']}** · Page {s['page']} · `{s['source']}`"
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "source_type": "financial"
                    })

                # ── general → web search ──────────────────────
                else:
                    answer, web_sources = web_search(user_input, llm)
                    st.markdown(answer)

                    if web_sources:
                        with st.expander("Sources — web search"):
                            for s in web_sources:
                                st.markdown(f"- [{s['title']}]({s['url']})")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": web_sources,
                        "source_type": "web"
                    })
def setup_on_cold_start():
    """Download PDFs and build index if not present (for cloud deployment)."""
    vectorstore_path = os.path.join(
        os.path.dirname(__file__), "..", "vectorstore"
    )
    faiss_path = os.path.join(vectorstore_path, "index.faiss")

    if not os.path.exists(faiss_path):
        with st.spinner("Setting up for first time — downloading filings and building index (3-5 mins)..."):
            # download PDFs
            sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
            from download_filings import download_all_filings
            download_all_filings()

            # build index
            subprocess.run(
                ["python", os.path.join(
                    os.path.dirname(__file__), "..", "src", "build_index.py"
                )],
                check=True
            )
        st.success("Setup complete! Reloading...")
        st.rerun()
st.set_page_config(
    page_title="Ask Your Annual Report",
    page_icon="📊",
    layout="wide"
)

setup_on_cold_start()   # ← add this line