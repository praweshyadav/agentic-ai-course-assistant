# 🤖 Agentic AI Course Assistant — Capstone Project

**Student:** [Your Name] | **Roll No:** [Your Roll No] | **Batch:** Agentic AI 2026

---

## What This Project Does

A complete agentic AI chatbot that answers questions about the Agentic AI Hands-On Course.
Built with LangGraph + ChromaDB + Groq LLM + Streamlit.

**6 Mandatory Capabilities:**
1. ✅ LangGraph StateGraph with 8 nodes
2. ✅ ChromaDB RAG with 12 documents
3. ✅ MemorySaver + thread_id multi-turn memory
4. ✅ Self-reflection eval node (faithfulness scoring)
5. ✅ Tool use — DateTime + Calculator + Web Search
6. ✅ Streamlit deployment

---

## Project Structure

```
course_assistant/
├── knowledge_base.py        ← 12 knowledge base documents
├── agent.py                 ← State, nodes, graph, tests, RAGAS
├── capstone_streamlit.py    ← Streamlit UI
├── day13_capstone.ipynb     ← Jupyter notebook submission
├── requirements.txt         ← Python dependencies
└── README.md                ← This file
```

---

## Setup Instructions

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Get a Groq API Key
1. Go to https://console.groq.com
2. Sign up (free)
3. Create an API key
4. Copy the key

### Step 3 — Set your API key
**Option A — Environment variable (recommended):**
```bash
# Mac/Linux
export GROQ_API_KEY="your_key_here"

# Windows
set GROQ_API_KEY=your_key_here
```

**Option B — In the notebook:**
Edit this line in `day13_capstone.ipynb`:
```python
os.environ['GROQ_API_KEY'] = 'your_groq_api_key_here'
```

### Step 4 — Run the Jupyter Notebook
```bash
jupyter notebook day13_capstone.ipynb
```
Run cells in order (Kernel → Restart & Run All before submitting)

### Step 5 — Launch Streamlit UI
```bash
streamlit run capstone_streamlit.py
```
Open your browser to http://localhost:8501

---

## Architecture

```
User Question
    ↓
[memory_node]      → stores history, sliding window (last 6), extract name
    ↓
[router_node]      → LLM → retrieve / tool / memory_only
    ↓
[retrieval_node / tool_node / skip_node]
    ↓
[answer_node]      → grounded LLM response
    ↓
[eval_node]        → faithfulness score → retry if < 0.7 (max 2 retries)
    ↓
[save_node]        → save answer to history → END
```

---

## Submission Checklist

- [ ] `day13_capstone.ipynb` — all cells run without error
- [ ] `capstone_streamlit.py` — UI launches correctly
- [ ] `agent.py` — all functions complete
- [ ] Project ZIP file ready
- [ ] GitHub repository created (public)
- [ ] PDF documentation (4–5 pages) ready
- [ ] Name, Roll No, Batch filled in notebook

**Deadline: April 21, 2026 at 11:59 PM**
