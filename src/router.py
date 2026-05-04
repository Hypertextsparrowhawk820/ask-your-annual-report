# src/router.py
import os
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

FINANCIAL_KEYWORDS = [
    "revenue", "profit", "loss", "earnings", "sales", "income",
    "apple", "microsoft", "tesla", "aapl", "msft", "tsla",
    "10-k", "annual report", "risk factor", "strategy", "filing",
    "balance sheet", "cash flow", "ebitda", "guidance", "forecast",
    "dividend", "shares", "stock", "quarterly", "fiscal", "10k",
    "sec", "operating", "gross margin", "net income", "ceo", "cfo"
]

def classify_intent(question: str, llm) -> str:
    """
    Returns 'casual', 'financial', or 'general'.
    Uses LLM to classify naturally — like ChatGPT would.
    """
    prompt = f"""You are an intent classifier for a chatbot. Classify the user message into exactly one of these three categories:

1. casual — greetings, small talk, how are you, thanks, bye, expressions, reactions, informal chat. Examples: "hello", "haii", "heyy", "sup", "wassup", "lol", "ok", "cool", "nice", "wow", "thanks", "bye", "who are you", "what can you do", "good morning"

2. financial — questions specifically about Apple, Microsoft, or Tesla companies, their revenues, profits, strategies, risks, products, SEC filings, or stock performance

3. general — factual questions about the world, people, places, science, sports, news, technology, history, or any topic NOT about Apple/Microsoft/Tesla specifically. Examples: "who is PM of India", "how does RAG work", "what is the capital of France", "latest AI news"

User message: "{question}"

Reply with just one word: casual, financial, or general"""

    response = llm.invoke([HumanMessage(content=prompt)])
    result = response.content.strip().lower()

    if "casual" in result:
        return "casual"
    elif "financial" in result:
        return "financial"
    else:
        return "general"


CASUAL_SYSTEM_PROMPT = """You are a friendly, helpful AI assistant — like ChatGPT. 
You have a warm, conversational personality. 
You're also specialized in analyzing Apple, Microsoft, and Tesla's SEC 10-K filings.

For casual conversation:
- Respond naturally and warmly, like a friend
- Keep responses short and conversational
- Don't be robotic or overly formal
- If someone greets you informally (haii, heyy, sup), match their energy
- You can mention your financial analysis capability naturally when relevant

Never say you can't have a conversation. Never redirect rudely."""


def casual_chat(question: str, llm) -> str:
    """Handle casual conversation naturally using the LLM."""
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

        prompt = f"""Answer the following question based on the web search results below.
Be concise, accurate, and helpful. Cite sources where relevant.

Search results:
{context}

Question: {question}

Answer:"""

        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content, sources

    except Exception as e:
        # fallback if Tavily fails
        response = llm.invoke([HumanMessage(content=question)])
        return response.content, []