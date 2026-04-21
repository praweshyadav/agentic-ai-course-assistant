# ============================================================
# agent.py — FINAL code
# ============================================================

from dotenv import load_dotenv
import os

# load_dotenv()
# from dotenv import load_dotenv
# import os

# load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

import warnings
warnings.filterwarnings("ignore")

import os
import re
from typing import TypedDict, List
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from tavily import TavilyClient

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from sentence_transformers import SentenceTransformer
import chromadb

from knowledge_base import DOCUMENTS

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "llama-3.1-8b-instant"
SLIDING_WINDOW = 6

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# ============================================================
# LLM
# ============================================================

def make_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model=MODEL_NAME,
        temperature=0
    )
# ============================================================
# KNOWLEDGE BASE
# ============================================================

def build_knowledge_base():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name="course_assistant_kb",
        metadata={"hnsw:space": "cosine"}
    )

    if collection.count() == 0:
        texts = [doc["text"] for doc in DOCUMENTS]
        ids = [doc["id"] for doc in DOCUMENTS]
        metas = [{"topic": doc["topic"]} for doc in DOCUMENTS]

        embeddings = embedder.encode(texts).tolist()

        collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metas
        )

    return embedder, collection


# ============================================================
# STATE
# ============================================================

class CapstoneState(TypedDict):
    question: str
    messages: List[dict]
    route: str
    retrieved: str
    sources: List[str]
    answer: str
    user_name: str
    tool_result: str
    faithfulness: float

# ============================================================
# MEMORY
# ============================================================

def extract_user_name(text):
    match = re.search(r"(?:my name is|i am)\s+([a-z]+(?:\s+[a-z]+)?)", text)
    return match.group(1).title() if match else None

def memory_node(state: CapstoneState):
    messages = state.get("messages", [])
    user_name = state.get("user_name", "")

    new_name = extract_user_name(state["question"].lower())
    if new_name:
        user_name = new_name

    messages.append({"role": "user", "content": state["question"]})
    messages = messages[-SLIDING_WINDOW:]

    return {"messages": messages, "user_name": user_name}
    
    follow_up = ["explain more", "more detail", "in detail", "elaborate", "tell me more"]

    q = question.strip().lower()

    if any(f in q for f in follow_up):
        last_answer = state.get("messages", [])[-1]["content"] if state.get("messages") else ""

        if last_answer:
            llm = make_llm()

            prompt = f"""
    Explain this in more detail:

{last_answer}
"""
        response = llm.invoke([HumanMessage(content=prompt)])

        return {"answer": response.content.strip()}

# ============================================================
# TOOL NODE
# ============================================================

def tool_node(state: CapstoneState):
    question = state["question"].lower()

    if question.strip() in ["ok", "okay", "hmm"]:
       return {
        "tool_result": "👍 Got it! What would you like to ask next?",
        "route": "memory_only"   # 🔥 force skip retrieval
    }

    # Date
    if "date" in question:
        return {"tool_result": f"📅 Today's date is {datetime.now().strftime('%Y-%m-%d')}"}

    # Time
    if "time" in question:
        return {"tool_result": f"⏰ Current time is {datetime.now().strftime('%H:%M:%S')}"}

    # Calculator
    if any(op in question for op in ["+", "-", "*", "/"]):
        try:
            result = eval(question)
            return {"tool_result": f"🧮 Result: {result}"}
        except:
            pass

    #web search trigger words
    web_keywords = [
    # time / updates
    # "latest", "news", "today", "current", "recent", "update", "updates",
    "latest", "news", "today", "current", "recent",
    "update", "updates", "trending",

    # identity / info
    "who is", "what is", "where is", "tell me about", "information about",

    # general queries
    "about", "details", "explain", "overview",

    # location / places
    "location", "located", "in which country", "in which city",

    # events / real-world
    "happening", "going on", "event", "events",

    # comparisons / facts
    "difference", "compare", "vs",

    # people / organizations
    "ceo", "founder", "company", "organization",

    # trending topics
    "trend", "trending",

    # general fallback words
    "history", "background"
]
#     web_keywords = [
#     "latest", "news", "today", "current", "recent",
#     "who is", "where is", "what is", "tell me about"
# ]
    if any(word in question for word in web_keywords):
    # if any(word in question for word in ["latest", "news", "today", "current", "who is", "recent"]):
        try:
            response = tavily.search(question, max_results=1,timeout=10)
            result = response["results"][0]

            content = result.get("content", "")
            url = result.get("url", "")

        # 🔥 Summarize using LLM
            llm = make_llm()

            summary_prompt = f"""
Write a clear and natural 2-sentence answer to the question.

Question: {state['question']}

Information:
{content[:500]}
"""

            summary = llm.invoke([HumanMessage(content=summary_prompt)]).content.strip()

            return {
                "tool_result": f"🌐 {summary}\n🔗 Source: {url}"
            }

        # except Exception as e:
        #     return {"tool_result": f"Web search failed: {str(e)}"}
        except Exception:
           # fallback → let LLM handle it
            return {}
    

    
# ============================================================
# ROUTER
# ============================================================
def router_node(state: CapstoneState):
    q = state["question"].lower().strip()

    # small talk → skip retrieval
    if q in ["ok", "okay", "hi", "hello", "hey", "hmm"]:
        return {"route": "memory_only"}

    return {"route": "retrieve"}



# ============================================================
# RETRIEVAL
# ============================================================

def make_retrieval_node(embedder, collection):
    def retrieval_node(state: CapstoneState):
        emb = embedder.encode([state["question"]]).tolist()
        results = collection.query(query_embeddings=emb, n_results=3)

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        return {
            "retrieved": "\n\n".join(docs),
            "sources": [m["topic"] for m in metas]
        }

    return retrieval_node

def answer_node(state: CapstoneState):

    question = state["question"].lower().strip()

    # 🔥 FIRST PRIORITY (VERY IMPORTANT)
    if question in ["ok", "okay", "hmm"]:
        return {"answer": "👍 Got it! What would you like to ask next?"}

    # 🔥 SECOND PRIORITY (tool output)
    if state.get("tool_result"):
        return {"answer": state["tool_result"]}
# ============================================================
# ANSWER NODE
# ============================================================

def answer_node(state: CapstoneState):

    question = state["question"].lower().strip()
    # ✅ final safety for small talk
    if question.strip() in ["ok", "okay", "hmm","thanks","thank you"]:
      return {"answer": "👍 Got it! What would you like to ask next?"}
    name = state.get("user_name", "")
    tool_result = state.get("tool_result", "")
    retrieved = state.get("retrieved", "")
    sources = state.get("sources", [])

    # Greeting
   
    # if any(g in question for g in ["hi", "hello", "hey", "hy", "hey buddy"]):
    greetings = ["hi", "hello", "hey", "hy", "hey buddy"]

    if question.strip() in greetings:
     return {
        "answer": "Hey 👋 How can I help you today?\n\nYou can ask me about your course or anything else!"
}

    

    # Tool
    if tool_result:
        return {"answer": tool_result}

    # Memory
    if "what is my name" in question:
        return {"answer": f"Your name is {name} 😊" if name else "I don't know your name yet."}

    # RAG + Web Combined
    llm = make_llm()

    prompt = f"""
Use the context and answer clearly.

Context:
{retrieved}

Question:
{state['question']}
"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content.strip()

        if sources:
            answer += f"\n\n📚 Sources: {', '.join(sources)}"

        return {"answer": answer}

    except Exception as e:
        return {"answer": f"⚠️ Error: {str(e)}"}

# ============================================================
# EVAL NODE
# ============================================================

def eval_node(state: CapstoneState):
    answer = state.get("answer", "")
    context = state.get("retrieved", "")

    if not context:
        return {"faithfulness": 0.0}

    overlap = sum(word in context.lower() for word in answer.lower().split())
    score = overlap / max(len(answer.split()), 1)

    return {"faithfulness": round(score, 2)}

# ============================================================
# SAVE
# ============================================================

def save_node(state: CapstoneState):
    messages = state.get("messages", [])
    messages.append({"role": "assistant", "content": state["answer"]})
    return {"messages": messages}

# ============================================================
# GRAPH
# ============================================================

def build_graph(embedder, collection):
    graph = StateGraph(CapstoneState)

    graph.add_node("memory", memory_node)
    graph.add_node("tool", tool_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", make_retrieval_node(embedder, collection))
    graph.add_node("answer", answer_node)
    graph.add_node("eval", eval_node)
    graph.add_node("save", save_node)

    graph.set_entry_point("memory")

    graph.add_edge("memory", "tool")
    graph.add_edge("tool", "router")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "eval")
    graph.add_edge("eval", "save")
    graph.add_edge("save", END)

    graph.add_conditional_edges("router", lambda s: s["route"], {
        "retrieve": "retrieve",
        "memory_only": "answer"
    })

    return graph.compile(checkpointer=MemorySaver())

# ============================================================
# ASK
# ============================================================

def ask(app, question: str, thread_id="default"):
    config = {"configurable": {"thread_id": thread_id}}

    state = {
        "question": question,
        "messages": [],
        "route": "",
        "retrieved": "",
        "sources": [],
        "answer": "",
        "user_name": "",
        "tool_result": "",
        "faithfulness": 0.0
    }

    return app.invoke(state, config=config)
#..................................................................
#run_tests function 
#..................................................................
def test_retrieval(embedder, collection):
    query = "What is LangChain?"
    
    emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=emb, n_results=3)

    docs = results["documents"][0]

    print("\n🔍 Retrieval Test:")
    for i, doc in enumerate(docs, 1):
        print(f"\nResult {i}:")
        print(doc[:200])

def run_tests(app):
    test_cases = [
        "hello",
        "what is langchain",
        "who is elon musk",
        "what is agentic ai",
        "ok"
    ]

    results = {}

    for i, q in enumerate(test_cases):
        try:
            config = {
                "configurable": {
                    "thread_id": f"test-{i}"
                }
            }

            state = {
                "question": q,
                "messages": [],
                "route": "",
                "retrieved": "",
                "sources": [],
                "answer": "",
                "user_name": "",
                "tool_result": ""
            }

            response = app.invoke(state, config=config)

            results[q] = response["answer"]

        except Exception as e:
            results[q] = f"❌ Error: {str(e)}"

    return results

# ============================================================
# RAGAS EVALUATION
#============================================================
def run_ragas_evaluation(app):
    test_questions = [
        "what is langchain",
        "what is agentic ai",
        "who is elon musk"
    ]

    results = []

    for i, q in enumerate(test_questions):
        try:
            config = {
                "configurable": {
                    "thread_id": f"ragas-{i}"
                }
            }

            state = {
                "question": q,
                "messages": [],
                "route": "",
                "retrieved": "",
                "sources": [],
                "answer": "",
                "user_name": "",
                "tool_result": ""
            }

            response = app.invoke(state, config=config)

            answer = response["answer"]
            context = response.get("retrieved", "")

            # simple faithfulness score
            overlap = sum(word in context.lower() for word in answer.lower().split())
            score = overlap / max(len(answer.split()), 1)

            results.append({
                "question": q,
                "answer": answer,
                "faithfulness": round(score, 2)
            })

        except Exception as e:
            results.append({
                "question": q,
                "error": str(e)
            })

    return results

# from agent import load_agent

# app = load_agent()

# config = {"configurable": {"thread_id": "test"}}

# state = {
#     "question": "who is elon musk",
#     "messages": [],
#     "route": "",
#     "retrieved": "",
#     "sources": [],
#     "answer": "",
#     "user_name": "",
#     "tool_result": ""
# }

# result = app.invoke(state, config=config)

# print(result["answer"])
