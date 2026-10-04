import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from langchain_core.messages import HumanMessage

# Load .env for local fallback (database creds, etc.)
load_dotenv()

# ============================================================
# Helpers
# ============================================================
def extract_text(msg):
    """Normalize AIMessage.content (str or list of blocks) to plain text."""
    content = msg.content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b)
            for b in content
        )
    return content or ""


def run_agent(question: str, extra_context: str = ""):
    # CRITICAL FIX: Import the agent here (lazy loading). 
    # This ensures the agent is initialized AFTER the user has set their API key 
    # in the Streamlit sidebar, preventing initialization errors.
    from agents.data_agent import data_agent

    full_question = f"{question}\n{extra_context}" if extra_context else question
    response = data_agent.invoke({
        "messages": [HumanMessage(content=full_question)],
        "route_response": "",
    })
    return extract_text(response["messages"][-1])


def get_postgres_engine():
    """Build a SQLAlchemy engine from .env — handles special chars in password."""
    url = URL.create(
        drivername="postgresql",
        username=os.getenv("user"),
        password=os.getenv("password"),
        host=os.getenv("host"),
        port=int(os.getenv("port", "5432")),
        database=os.getenv("database"),
    )
    return create_engine(url)


# ============================================================
# Page config & Styling
# ============================================================
st.set_page_config(
    page_title="Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        :root {
            --accent: #6366f1;
            --accent-light: #eef2ff;
            --accent-dark: #4338ca;
            --ink: #0f172a;
            --muted: #64748b;
            --border: #e2e8f0;
            --surface: #ffffff;
            --bg: #f8fafc;
            --success: #059669;
            --danger: #dc2626;
        }

        /* App background */
        .stApp { background: var(--bg); }
        #MainMenu, footer, header {visibility: hidden;}
        .block-container { padding-top: 1.5rem; max-width: 1100px; }

        /* ---------- Hero header ---------- */
        .hero {
            background: linear-gradient(135deg, #1e1b4b 0%, #4338ca 55%, #6366f1 100%);
            border-radius: 20px;
            padding: 2.1rem 2.4rem;
            margin-bottom: 1.6rem;
            box-shadow: 0 10px 30px -12px rgba(67, 56, 202, 0.45);
            position: relative;
            overflow: hidden;
        }
        .hero::after {
            content: ""; position: absolute;
            top: -60px; right: -60px;
            width: 220px; height: 220px;
            background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 70%);
            border-radius: 50%;
        }
        .hero-eyebrow {
            color: #c7d2fe; font-size: 0.72rem; font-weight: 600;
            letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 0.5rem;
        }
        .hero-title {
            color: #ffffff; font-size: 1.9rem; font-weight: 800;
            margin: 0; letter-spacing: -0.02em;
        }
        .hero-sub {
            color: #e0e7ff; font-size: 0.95rem; margin-top: 0.4rem;
            font-weight: 400; max-width: 640px; line-height: 1.5;
        }

        /* ---------- Capability cards ---------- */
        .cap-card {
            background: var(--surface); border: 1px solid var(--border);
            border-radius: 14px; padding: 1.1rem 1.3rem; height: 100%;
            transition: box-shadow 0.15s ease, transform 0.15s ease;
        }
        .cap-card:hover {
            box-shadow: 0 8px 20px -10px rgba(15, 23, 42, 0.15);
            transform: translateY(-1px);
        }
        .cap-icon {
            width: 34px; height: 34px; border-radius: 9px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.05rem; margin-bottom: 0.6rem;
        }
        .cap-icon.sql { background: var(--accent-light); }
        .cap-icon.etl { background: #ecfdf5; }
        .cap-title { font-weight: 700; font-size: 0.92rem; color: var(--ink); margin-bottom: 0.25rem; }
        .cap-desc { font-size: 0.82rem; color: var(--muted); line-height: 1.45; margin: 0; }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: var(--surface); border-right: 1px solid var(--border);
        }
        section[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
        .sb-section-label {
            font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em;
            text-transform: uppercase; color: var(--muted);
            margin: 1.1rem 0 0.5rem 0; display: flex; align-items: center; gap: 0.4rem;
        }
        .sb-brand { display: flex; align-items: center; gap: 0.55rem; margin-bottom: 0.2rem; }
        .sb-brand-mark {
            width: 30px; height: 30px; border-radius: 8px;
            background: linear-gradient(135deg, var(--accent), var(--accent-dark));
            display: flex; align-items: center; justify-content: center;
            color: white; font-weight: 800; font-size: 0.85rem;
        }
        .sb-brand-name { font-weight: 700; font-size: 1.02rem; color: var(--ink); }
        .status-pill {
            display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.74rem;
            font-weight: 600; padding: 0.28rem 0.6rem; border-radius: 999px; margin-top: 0.4rem;
        }
        .status-pill.on { background: #ecfdf5; color: var(--success); }
        .status-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }

        /* ---------- Buttons & inputs ---------- */
        .stButton > button {
            border-radius: 9px; font-weight: 600; font-size: 0.85rem;
            border: 1px solid var(--accent); background: var(--accent); color: white;
            padding: 0.45rem 1rem; transition: all 0.15s ease;
        }
        .stButton > button:hover { background: var(--accent-dark); border-color: var(--accent-dark); }
        div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] > div {
            border-radius: 8px !important;
        }

        /* ---------- Chat & Misc ---------- */
        .stChatMessage {
            background: var(--surface); border: 1px solid var(--border);
            border-radius: 14px; padding: 0.3rem 0.4rem; margin-bottom: 0.6rem;
        }
        div[data-testid="stChatInput"] textarea { border-radius: 12px; }
        hr { border: none; border-top: 1px solid var(--border); margin: 0.9rem 0; }
        div[data-testid="stExpander"] {
            border: 1px solid var(--border); border-radius: 14px; background: var(--surface);
            box-shadow: 0 4px 14px -8px rgba(15, 23, 42, 0.1); margin-bottom: 1.4rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Hero header & Capabilities
# ============================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">DATA WORKSPACE</div>
        <h1 class="hero-title">Analyst Agent</h1>
        <p class="hero-sub">
            Ask questions in plain English. Your request is routed automatically to a
            SQL analyst for database queries or an ETL analyst for extract / transform
            workflows — no query writing required.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("🧭  What can this agent do?", expanded=True):
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(
            """
            <div class="cap-card">
                <div class="cap-icon sql">🗄️</div>
                <div class="cap-title">SQL Analyst</div>
                <p class="cap-desc">Answers questions about your PostgreSQL database. Curates the question, generates SQL, runs it past a safety judge, executes it, and explains the result.</p>
            </div>
            """, unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            """
            <div class="cap-card">
                <div class="cap-icon etl">🔧</div>
                <div class="cap-title">ETL Analyst</div>
                <p class="cap-desc">Extracts data from an API and saves it as CSV/JSON/Parquet, or transforms an uploaded CSV with a Pandas operation and saves the result.</p>
            </div>
            """, unsafe_allow_html=True
        )

# ============================================================
# Sidebar Configuration
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sb-brand">
            <div class="sb-brand-mark">A</div>
            <div class="sb-brand-name">Agent Setup</div>
        </div>
        """, unsafe_allow_html=True,
    )
    
    # --- BYOK API Key Section ---
    st.markdown("<div class='sb-section-label'>🔑 API Configuration</div>", unsafe_allow_html=True)
    provider = st.selectbox("Select LLM Provider", ["Groq", "Google Gemini", "Mistral"])
    api_key = st.text_input(f"{provider} API Key", type="password", placeholder="Paste key here...")

    st.markdown("<div class='sb-section-label'>🌐 Extract from API</div>", unsafe_allow_html=True)
    api_url = st.text_input("API URL", placeholder="https://pokeapi.co/api/v2/pokemon", label_visibility="collapsed")
    api_format = st.selectbox("Output format", ["csv", "json", "parquet"], label_visibility="collapsed")

    st.markdown("<div class='sb-section-label'>📄 Transform a CSV</div>", unsafe_allow_html=True)
    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    st.markdown("<div class='sb-section-label'>🗃️ Load CSV into Database</div>", unsafe_allow_html=True)
    db_upload = st.file_uploader("Upload CSV for Postgres", type=["csv"], key="db_upload", label_visibility="collapsed")

    if db_upload is not None:
        table_name_input = st.text_input(
            "Table name",
            value=db_upload.name.replace(".csv", "").lower().replace(" ", "_"),
        )
        if st.button("Load into database", use_container_width=True):
            try:
                df = pd.read_csv(db_upload)
                engine = get_postgres_engine()
                df.to_sql(table_name_input, con=engine, if_exists="replace", index=False)
                st.success(f"Loaded into table `{table_name_input}`")
            except Exception as e:
                st.error(f"Failed to load: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Configure data sources above, then ask your question in the chat.")

# ============================================================
# API Key Validation (Renders in Main Area)
# ============================================================
if not api_key:
    st.info(f"👈 **Please enter your {provider} API key in the sidebar to unlock the chat.**")
    st.stop()
    
# Dynamically inject the key into environment variables
if provider == "Groq":
    os.environ["GROQ_API_KEY"] = api_key
elif provider == "Google Gemini":
    os.environ["GOOGLE_API_KEY"] = api_key
elif provider == "Mistral":
    os.environ["MISTRAL_API_KEY"] = api_key

# ============================================================
# Build Context & Chat UI
# ============================================================
# ... rest of your code (extra_context logic, chat UI, etc.) ...

# ============================================================
# Build Context & Chat UI
# ============================================================
extra_context = ""
active_sources = []

if api_url:
    extra_context += f"\n[API to extract from: {api_url}]\n[Output format: {api_format}]"
    active_sources.append(f"API → `{api_url}`")

if uploaded_csv is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
        f.write(uploaded_csv.getbuffer())
        csv_path = f.name.replace("\\", "/")
    extra_context += f"\n[Uploaded CSV path: {csv_path}]"
    active_sources.append(f"CSV → `{uploaded_csv.name}`")

if active_sources:
    st.markdown(
        "".join(
            f"<span class='status-pill on' style='margin-right:0.4rem;'>"
            f"<span class='status-dot'></span>{s}</span>"
            for s in active_sources
        ),
        unsafe_allow_html=True,
    )
    st.write("")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown(
        "<p style='color:var(--muted); font-size:0.9rem; text-align:center; margin-top:2rem;'>"
        "No messages yet — ask a question about your data to get started.</p>",
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your data..."):
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                answer = run_agent(prompt, extra_context)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                err = f"Something went wrong: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})