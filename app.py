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
try:
    for k in ("host", "port", "user", "password", "database"):
        if k in st.secrets:
            os.environ[k] = str(st.secrets[k])
except Exception:
    pass


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
    # Lazy loading so the agent initializes AFTER the API key is set in os.environ
    from agents.data_agent import data_agent

    full_question = f"{question}\n{extra_context}" if extra_context else question
    response = data_agent.invoke({
        "messages": [HumanMessage(content=full_question)],
        "route_response": "",
    })
    return extract_text(response["messages"][-1])


def get_env_or_secret(key: str, default: str = None):
    """Fetch from Streamlit Cloud secrets FIRST, then fallback to os.environ."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    val = os.getenv(key)
    if val:
        return val
    return default


def get_postgres_engine():
    """Build a SQLAlchemy engine from .env or Streamlit Secrets."""
    url = URL.create(
        drivername="postgresql",
        username=get_env_or_secret("user"),
        password=get_env_or_secret("password"),
        host=get_env_or_secret("host"),
        port=int(get_env_or_secret("port", "5432")),
        database=get_env_or_secret("database"),
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

        :root {
            --accent: #6366f1;
            --accent-light: #eef2ff;
            --accent-dark: #4338ca;
            --ink: #0f172a;
            --muted: #475569;
            --border: #cbd5e1;
            --surface: #ffffff;
            --surface-2: #f1f5f9;
            --bg: #f8fafc;
            --success: #059669;
            --danger: #dc2626;
            color-scheme: light only;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: var(--ink);
        }

        /* ---------- App shell ---------- */
        .stApp { background: var(--bg) !important; color: var(--ink) !important; }
        #MainMenu, footer,
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"] { visibility: hidden; }
        /* keep header (sidebar toggle lives here) but make it invisible */
        header[data-testid="stHeader"] { background: transparent !important; }
        header[data-testid="stHeader"] button,
        header[data-testid="stHeader"] svg,
        button[data-testid="stExpandSidebarButton"],
        button[data-testid="stSidebarCollapseButton"],
        div[data-testid="stSidebarCollapseButton"] button,
        div[data-testid="stSidebarCollapsedControl"] button {
            color: var(--ink) !important;
            fill: var(--ink) !important;
            visibility: visible !important;
        }
        .block-container { padding-top: 1.5rem; max-width: 1100px; }

        /* ---------- Chat text ---------- */
        div[data-testid="stChatMessageContent"] p,
        div[data-testid="stChatMessageContent"] div,
        div[data-testid="stChatMessageContent"] span,
        div[data-testid="stChatMessageContent"] li {
            color: var(--ink) !important;
        }
        div[data-testid="stChatMessageContent"] code {
            color: var(--accent-dark) !important;
            background: var(--accent-light) !important;
            padding: 0.1rem 0.35rem;
            border-radius: 5px;
        }

        /* ---------- Hero ---------- */
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
        .hero .hero-eyebrow {
            color: #c7d2fe !important; font-size: 0.72rem; font-weight: 600;
            letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 0.5rem;
        }
        .hero h1.hero-title {
            color: #ffffff !important; font-size: 1.9rem; font-weight: 800;
            margin: 0; padding: 0; letter-spacing: -0.02em;
        }
        .hero h1.hero-title a { display: none !important; }
        .hero p.hero-sub {
            color: #e0e7ff !important; font-size: 0.95rem; margin-top: 0.4rem;
            font-weight: 400; max-width: 640px; line-height: 1.5;
        }

        /* ---------- Expander ---------- */
        div[data-testid="stExpander"] {
            border: 1px solid var(--border) !important;
            border-radius: 14px !important;
            background: var(--surface) !important;
            box-shadow: 0 4px 14px -8px rgba(15, 23, 42, 0.08);
            margin-bottom: 1.4rem;
            overflow: hidden;
        }
        div[data-testid="stExpander"] details,
        div[data-testid="stExpander"] summary {
            background: var(--surface) !important;
            color: var(--ink) !important;
        }
        div[data-testid="stExpander"] summary p,
        div[data-testid="stExpander"] summary span,
        div[data-testid="stExpander"] summary svg {
            color: var(--ink) !important;
            fill: var(--ink) !important;
            font-weight: 600 !important;
        }
        div[data-testid="stExpander"] summary:hover { background: var(--surface-2) !important; }

        /* ---------- Capability cards ---------- */
        .cap-card {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
            height: 100%;
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
        .cap-card .cap-title {
            font-weight: 700; font-size: 0.92rem;
            color: var(--ink) !important; margin-bottom: 0.25rem;
        }
        .cap-card p.cap-desc {
            font-size: 0.82rem; color: var(--muted) !important;
            line-height: 1.45; margin: 0;
        }

        /* ============================================================
           SIDEBAR
           ============================================================ */
        section[data-testid="stSidebar"] {
            background: var(--surface) !important;
            border-right: 1px solid var(--border) !important;
        }
        section[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] label p,
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: var(--ink) !important;
        }

        /* --- Text / password inputs: the outer baseweb wrapper owns border + bg --- */
        section[data-testid="stSidebar"] div[data-baseweb="input"],
        section[data-testid="stSidebar"] div[data-baseweb="base-input"] {
            background-color: var(--surface-2) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            box-shadow: none !important;
        }
        /* inner wrapper must not draw its own border (this caused the black outline) */
        section[data-testid="stSidebar"] div[data-baseweb="base-input"] {
            border: none !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="input"]:focus-within {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.18) !important;
        }
        section[data-testid="stSidebar"] input {
            background-color: transparent !important;
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
            caret-color: var(--accent) !important;
        }
        section[data-testid="stSidebar"] input::placeholder {
            color: #94a3b8 !important;
            -webkit-text-fill-color: #94a3b8 !important;
            opacity: 1 !important;
        }
        /* password show/hide eye button (the dark block on the right) */
        section[data-testid="stSidebar"] div[data-baseweb="input"] button,
        section[data-testid="stSidebar"] div[data-baseweb="input"] > div:last-child {
            background-color: transparent !important;
            border: none !important;
            color: var(--muted) !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="input"] button svg {
            fill: var(--muted) !important;
            color: var(--muted) !important;
        }

        /* --- Selectbox (closed state) --- */
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background-color: var(--surface-2) !important;
            color: var(--ink) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            box-shadow: none !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.18) !important;
        }
        /* inner parts (selected value + arrow area) must stay transparent */
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div > div,
        section[data-testid="stSidebar"] div[data-baseweb="select"] div[value],
        section[data-testid="stSidebar"] div[data-baseweb="select"] span {
            background-color: transparent !important;
            color: var(--ink) !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
            fill: var(--muted) !important;
            color: var(--muted) !important;
        }

        /* --- Dropdown popup: rendered in a portal OUTSIDE the sidebar --- */
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div,
        div[data-baseweb="popover"] ul,
        ul[data-testid="stSelectboxVirtualDropdown"] {
            background-color: var(--surface) !important;
            color: var(--ink) !important;
        }
        div[data-baseweb="popover"] li,
        div[data-baseweb="popover"] li * {
            background-color: var(--surface) !important;
            color: var(--ink) !important;
        }
        div[data-baseweb="popover"] li:hover,
        div[data-baseweb="popover"] li[aria-selected="true"] {
            background-color: var(--accent-light) !important;
        }
        div[data-baseweb="popover"] li:hover *,
        div[data-baseweb="popover"] li[aria-selected="true"] * {
            background-color: var(--accent-light) !important;
            color: var(--accent-dark) !important;
        }

        /* --- File uploader --- */
        section[data-testid="stSidebar"] section[data-testid="stFileUploaderDropzone"] {
            background-color: var(--surface-2) !important;
            border: 1px dashed var(--border) !important;
            border-radius: 10px !important;
            color: var(--ink) !important;
        }
        section[data-testid="stSidebar"] section[data-testid="stFileUploaderDropzone"] * {
            color: var(--muted) !important;
        }
        /* the "Upload" / "Browse files" button */
        section[data-testid="stSidebar"] section[data-testid="stFileUploaderDropzone"] button {
            background-color: var(--surface) !important;
            color: var(--accent-dark) !important;
            border: 1px solid var(--accent) !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        section[data-testid="stSidebar"] section[data-testid="stFileUploaderDropzone"] button *,
        section[data-testid="stSidebar"] section[data-testid="stFileUploaderDropzone"] button svg {
            color: var(--accent-dark) !important;
            fill: var(--accent-dark) !important;
        }
        section[data-testid="stSidebar"] section[data-testid="stFileUploaderDropzone"] button:hover {
            background-color: var(--accent-light) !important;
        }
        /* uploaded file chip */
        section[data-testid="stSidebar"] div[data-testid="stFileUploaderFile"],
        section[data-testid="stSidebar"] div[data-testid="stFileUploaderFile"] * {
            background-color: transparent !important;
            color: var(--ink) !important;
        }

        /* --- Sidebar brand / labels --- */
        .sb-section-label {
            font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em;
            text-transform: uppercase; color: var(--muted) !important;
            margin: 1.1rem 0 0.5rem 0; display: flex; align-items: center; gap: 0.4rem;
        }
        .sb-brand { display: flex; align-items: center; gap: 0.55rem; margin-bottom: 0.2rem; }
        .sb-brand-mark {
            width: 30px; height: 30px; border-radius: 8px;
            background: linear-gradient(135deg, var(--accent), var(--accent-dark));
            display: flex; align-items: center; justify-content: center;
            color: #ffffff !important; font-weight: 800; font-size: 0.85rem;
        }
        .sb-brand-name { font-weight: 700; font-size: 1.02rem; color: var(--ink) !important; }
        .status-pill {
            display: inline-flex; align-items: center; gap: 0.35rem; font-size: 0.74rem;
            font-weight: 600; padding: 0.28rem 0.6rem; border-radius: 999px; margin-top: 0.4rem;
        }
        .status-pill.on { background: #ecfdf5; color: var(--success) !important; }
        .status-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }

        /* ---------- Buttons (primary action buttons) ---------- */
        .stButton > button {
            border-radius: 9px; font-weight: 600; font-size: 0.85rem;
            border: 1px solid var(--accent); background: var(--accent) !important;
            color: #ffffff !important;
            padding: 0.45rem 1rem; transition: all 0.15s ease;
        }
        .stButton > button p, .stButton > button span { color: #ffffff !important; }
        .stButton > button:hover { background: var(--accent-dark) !important; border-color: var(--accent-dark); }

        /* ---------- Alerts (st.info / success / error) ---------- */
        div[data-testid="stAlert"] p,
        div[data-testid="stAlert"] span,
        div[data-testid="stAlert"] div { color: var(--ink) !important; }

        /* ---------- Spinner ---------- */
        div[data-testid="stSpinner"] p,
        div[data-testid="stSpinner"] span { color: var(--muted) !important; }

        /* ---------- Chat messages & input ---------- */
        .stChatMessage {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 14px;
            padding: 0.6rem 0.8rem;
            margin-bottom: 0.6rem;
            color: var(--ink) !important;
        }
        div[data-testid="stBottom"],
        div[data-testid="stBottom"] > div {
            background-color: var(--bg) !important;
        }
        div[data-testid="stChatInput"] {
            background-color: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
        }
        div[data-testid="stChatInput"] > div,
        div[data-testid="stChatInput"] div[data-baseweb="textarea"],
        div[data-testid="stChatInput"] div[data-baseweb="base-input"] {
            background-color: var(--surface) !important;
            border: none !important;
        }
        div[data-testid="stChatInput"] textarea {
            background-color: var(--surface) !important;
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
            caret-color: var(--accent) !important;
        }
        div[data-testid="stChatInput"] textarea::placeholder {
            color: #94a3b8 !important;
            -webkit-text-fill-color: #94a3b8 !important;
        }
        div[data-testid="stChatInput"] button {
            background-color: transparent !important;
            color: var(--accent) !important;
        }
        div[data-testid="stChatInput"] button svg { fill: var(--accent) !important; }
        hr { border: none; border-top: 1px solid var(--border); margin: 0.9rem 0; }
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
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="cap-card">
                <div class="cap-icon etl">🔧</div>
                <div class="cap-title">ETL Analyst</div>
                <p class="cap-desc">Extracts data from an API and saves it as CSV/JSON/Parquet, or transforms an uploaded CSV with a Pandas operation and saves the result.</p>
            </div>
            """,
            unsafe_allow_html=True,
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
        """,
        unsafe_allow_html=True,
    )

    # --- BYOK API Key Section ---
    st.markdown("<div class='sb-section-label'>🔑 API Configuration</div>", unsafe_allow_html=True)
    provider = st.selectbox("Select LLM Provider", ["Groq", "Google Gemini", "Mistral"])
    api_key = st.text_input(f"{provider} API Key", type="password", placeholder="Paste key here...")

    st.markdown("<div class='sb-section-label'>🌐 Extract from API</div>", unsafe_allow_html=True)
    api_url = st.text_input(
        "API URL",
        placeholder="https://pokeapi.co/api/v2/pokemon",
        label_visibility="collapsed",
    )
    api_format = st.selectbox(
        "Output format", ["csv", "json", "parquet"], label_visibility="collapsed"
    )

    st.markdown("<div class='sb-section-label'>📄 Transform a CSV</div>", unsafe_allow_html=True)
    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    st.markdown("<div class='sb-section-label'>🗃️ Load CSV into Database</div>", unsafe_allow_html=True)
    db_upload = st.file_uploader(
        "Upload CSV for Postgres", type=["csv"], key="db_upload", label_visibility="collapsed"
    )

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
os.environ["LLM_PROVIDER"] = provider
if provider == "Groq":
    os.environ["GROQ_API_KEY"] = api_key
elif provider == "Google Gemini":
    os.environ["GOOGLE_API_KEY"] = api_key
elif provider == "Mistral":
    os.environ["MISTRAL_API_KEY"] = api_key

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