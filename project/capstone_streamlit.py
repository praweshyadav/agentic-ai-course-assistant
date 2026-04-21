
# this is new code proper ..............................................................................................


# ============================================================
# capstone_streamlit.py — FINAL CLEAN VERSION
# ============================================================

import streamlit as st
import os
from dotenv import load_dotenv
from agent import build_knowledge_base, build_graph, ask

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Agentic AI Course Assistant",
    page_icon="🤖",
    layout="wide",
)

# ============================================================
# LOAD AGENT
# ============================================================
@st.cache_resource
def load_agent():
    groq_key = os.environ.get("GROQ_API_KEY")

    if not groq_key:
        st.error("❌ GROQ_API_KEY not found. Check your .env file.")
        st.stop()

    os.environ["GROQ_API_KEY"] = groq_key

    embedder, collection = build_knowledge_base()
    app = build_graph(embedder, collection)
    return app


with st.spinner("🔧 Loading AI Assistant (first run takes ~30s)..."):
    app = load_agent()

# ============================================================
# SESSION STATE
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())

# ============================================================
# SIDEBAR
# ============================================================
# with st.sidebar:
#     st.title("🤖 Course Assistant")
#     st.markdown("Agentic AI Course")
name = st.session_state.get("user_name", "Prawesh Yadav")

with st.sidebar:
    st.markdown("## 🤖 Course Assistant")
    st.markdown("**Agentic AI Hands-On Course**")
    st.markdown(f"*👨‍💻 Built by {name.title()}*")

    st.divider()

    if st.button("🔄 New Conversation"):
        import uuid
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    # Download chat
    if st.button("📥 Download Chat"):
        st.download_button(
            "Download",
            "\n".join([m["content"] for m in st.session_state.messages]),
            file_name="chat.txt"
        )

# ============================================================
# MAIN UI
# ============================================================
# st.title("🤖 Agentic AI Course Assistant")
# st.markdown("##### 👨‍💻 Built by Prawesh Yadav")
# st.markdown("I can explain concepts, help debug your code logic, and answer course questions.")
# st.markdown("""
# <div style="
#     text-align:center;
#     padding:25px;
#     border-radius:15px;
#     background: linear-gradient(90deg, #1f4037, #99f2c8);
#     color:#2c2c2c;
# ">
#     <h1 style="margin-bottom:10px;">🤖 Agentic AI Course Assistant</h1>
#     <p style="font-size:18px;">👨‍💻 Built by <b>Prawesh Yadav</b></p>
# </div>
# """, unsafe_allow_html=True)

st.markdown("""
<div style="
    text-align:center;
    padding:20px;
    border-radius:15px;
    background: linear-gradient(90deg, #1f4037, #99f2c8);
    color:#2c2c2c;
">
    <h1 style="margin-bottom:8px; font-size:28px;">🤖 Agentic AI Course Assistant</h1>
    <p style="font-size:14px;">👨‍💻 Built by <b>Prawesh Yadav</b></p>
</div>
""", unsafe_allow_html=True)

# st.markdown("I can explain concepts, help debug your code logic, and answer course questions.")

st.markdown("""
<div style="
    text-align:center;
    background: linear-gradient(90deg,#eef2f3,#dfe9f3);
    padding:14px;
    border-radius:12px;
    margin-top:10px;
    font-size:16px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
">
    💡 <b>I can explain concepts, debug your code logic, and answer course questions.</b>
</div>
""", unsafe_allow_html=True)

if not st.session_state.messages:
    st.info("Ask me anything about Agentic AI 🚀")
    st.info("I can do websearch in real-time and answer your querry based on the latest information available on the web")

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask your question...")

# ============================================================
# CHAT LOGIC (WITH MEMORY)
# ============================================================
if user_input:

    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # =========================
    # 🔥 MEMORY: Detect & save name
    # =========================
    text = user_input.lower()
    name = None

    if "my name is" in text:
        name = text.split("my name is")[-1].strip()

    elif "i am" in text:
        name = text.split("i am")[-1].strip()

    elif "i'm" in text:
        name = text.split("i'm")[-1].strip()

    if name:
        st.session_state["user_name"] = name

    # =========================
    # 🔥 MEMORY: Answer name question
    # =========================
    if "my name" in text:
        saved_name = st.session_state.get("user_name", None)

        if saved_name:
            answer = f"Your name is {saved_name.title()} 😊"
        else:
            answer = "I don't know your name yet. Please tell me!"

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.stop()
    
    # =========================
    # 🤖 NORMAL AI RESPONSE
    # =========================
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask(app, user_input, thread_id=st.session_state.thread_id)

            answer = result.get("answer", "No answer generated.")
            route = result.get("route", "")
            faith = result.get("faithfulness", 0.0)
            sources = result.get("sources", [])

            # st.markdown(answer)
            st.markdown(f"""
<div style="background-color:#f0f2f6; padding:12px; border-radius:10px;">
<h4>🤖 Answer</h4>
<p>{answer}</p>
<hr>
<small>✨ Powered by Agentic AI</small>
</div>
""", unsafe_allow_html=True)

            with st.expander("🔍 Details"):
                st.write(f"Route: {route}")
                st.write(f"Faithfulness: {faith:.2f}")
                st.write(f"Sources: {sources}")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )