# src/router.py
import os
import re
import requests
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

CASUAL_TRIGGERS = [
    "hello", "hi", "hey", "hii", "haii", "heyy", "heyyy", "hai",
    "howdy", "sup", "wassup", "whats up", "what's up",
    "how are you", "how r u", "how are u",
    "good morning", "good evening", "good afternoon", "good night",
    "who are you", "what are you", "what can you do",
    "thank you", "thanks", "thx", "ty",
    "bye", "goodbye", "see you", "later",
    "ok", "okay", "cool", "nice", "wow", "great", "awesome",
    "lol", "haha", "lmao", "😊", "👋", "help me", "help"
]

FINANCIAL_KEYWORDS = [
    "revenue", "profit", "loss", "earnings", "sales", "income",
    "apple", "microsoft", "tesla", "aapl", "msft", "tsla",
    "10-k", "annual report", "risk factor", "strategy", "filing",
    "balance sheet", "cash flow", "ebitda", "guidance", "forecast",
    "dividend", "shares", "stock", "quarterly", "fiscal", "10k",
    "sec", "operating", "gross margin", "net income", "ceo", "cfo",
    "compare", "comparison", "versus", "vs", "difference between",
    "three companies", "all companies", "both companies"
]

CASUAL_SYSTEM_PROMPT = """You are a friendly, warm AI assistant — like ChatGPT.
You have a conversational personality and also specialize in financial analysis 
of Apple, Microsoft, and Tesla SEC 10-K filings.

For casual messages:
- Respond naturally and warmly, matching the user's energy
- Keep it short and friendly — 1-2 sentences max
- If someone says "haii" or "heyy", be equally enthusiastic
- Mention your financial analysis capability naturally when relevant
- Never be robotic or redirect rudely"""

def is_casual_fast(text: str) -> bool:
    """Fast keyword check before calling LLM."""
    text_lower = text.lower().strip().rstrip("?!., ")
    # exact match or contained in short message
    for trigger in CASUAL_TRIGGERS:
        if trigger == text_lower or (trigger in text_lower and len(text_lower) < 20):
            return True
    return False

def classify_intent(question: str, llm) -> str:
    """Returns 'casual', 'financial', or 'general'."""

    # fast casual check — no LLM call needed
    if is_casual_fast(question):
        return "casual"

    # fast financial keyword check
    q_lower = question.lower()
    for keyword in FINANCIAL_KEYWORDS:
        if keyword in q_lower:
            return "financial"

    # LLM classification for ambiguous cases
    prompt = f"""Classify this message into exactly one category:

1. casual — greetings, small talk, reactions, very short informal messages
2. financial — questions about Apple, Microsoft, Tesla companies, their revenues, profits, strategies, risks, SEC filings, comparisons between these companies
3. general — factual questions about the world, people, places, science, sports, news, technology NOT about Apple/Microsoft/Tesla

Message: "{question}"

Reply with just one word: casual, financial, or general"""

    response = llm.invoke([HumanMessage(content=prompt)])
    result = response.content.strip().lower()

    if "casual" in result:
        return "casual"
    elif "financial" in result:
        return "financial"
    else:
        return "general"

def casual_chat(question: str, llm) -> str:
    """Handle casual conversation naturally."""
    response = llm.invoke([
        HumanMessage(content=f"{CASUAL_SYSTEM_PROMPT}\n\nUser: {question}\n\nAssistant:")
    ])
    return response.content

def web_search(question: str, llm) -> tuple:
    """Search the web and generate a grounded answer."""
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        results = client.search(
            query=question,
            max_results=5,
            search_depth="basic"
        )

        context = ""
        sources = []
        for r in results.get("results", []):
            context += f"\nSource: {r['url']}\n{r['content']}\n"
            sources.append({
                "title": r.get("title", "Web source"),
                "url": r["url"]
            })

        prompt = f"""Answer the following question based on the web search results.
Be concise, accurate, and cite sources. Use plain text — no LaTeX or math formatting.

Search results:
{context}

Question: {question}

Answer:"""

        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content, sources

    except Exception as e:
        response = llm.invoke([HumanMessage(content=question)])
        return response.content, []
