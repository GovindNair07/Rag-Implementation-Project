import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Book Assistant",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --ink:       #1a1410;
    --paper:     #faf7f2;
    --cream:     #f0ebe1;
    --warm-mid:  #c4a882;
    --accent:    #8b5e3c;
    --accent-lt: #d4956a;
    --muted:     #7a6e63;
    --border:    #e0d8cc;
    --success:   #4a7c59;
    --card-bg:   #fffcf7;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--ink);
    background-color: var(--paper);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1100px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--ink);
    border-right: 1px solid #2e2620;
}
[data-testid="stSidebar"] * { color: var(--cream) !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Lora', serif !important;
    color: var(--warm-mid) !important;
}
[data-testid="stSidebar"] .stMarkdown p { color: #b0a898 !important; font-size: 0.85rem; }
[data-testid="stSidebar"] hr { border-color: #3a322b !important; }

/* Hero */
.hero {
    background: linear-gradient(135deg, var(--ink) 0%, #2d2118 60%, #3d2e1e 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(196,168,130,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: rgba(196,168,130,0.15);
    border: 1px solid rgba(196,168,130,0.3);
    color: var(--warm-mid);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Lora', serif;
    font-size: 2.2rem;
    font-weight: 600;
    color: var(--cream);
    margin: 0 0 0.4rem;
    letter-spacing: -0.02em;
}
.hero-sub {
    font-size: 1rem;
    color: var(--warm-mid);
    margin: 0;
    font-weight: 300;
}

/* Section labels */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.5rem;
}

/* Cards */
.card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 4px rgba(26,20,16,0.05);
}
.card-accent { border-left: 3px solid var(--accent); }

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--cream);
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 0.5rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent-lt); }

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: var(--cream) !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.55rem 1.6rem !important;
    transition: background 0.2s, transform 0.1s !important;
    box-shadow: 0 2px 8px rgba(139,94,60,0.3) !important;
}
.stButton > button:hover {
    background: #a06b45 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(139,94,60,0.4) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Text input */
.stTextInput > div > div > input {
    background: var(--card-bg) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1rem !important;
    color: var(--ink) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(139,94,60,0.12) !important;
}
.stTextInput label { font-weight: 500 !important; font-size: 0.88rem !important; color: var(--muted) !important; }

/* Alerts */
.stAlert { border-radius: 10px !important; font-size: 0.88rem !important; }

/* Spinner */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* Divider */
hr { border-color: var(--border) !important; margin: 2rem 0 !important; }

/* Answer box */
.answer-box {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-top: 3px solid var(--accent);
    border-radius: 12px;
    padding: 1.6rem 2rem;
    margin-top: 1.2rem;
    font-family: 'Lora', serif;
    font-size: 1rem;
    line-height: 1.8;
    color: var(--ink);
}

/* Source chips */
.source-chip {
    display: inline-block;
    background: var(--cream);
    border: 1px solid var(--border);
    border-radius: 20px;
    font-size: 0.75rem;
    color: var(--muted);
    padding: 3px 10px;
    margin: 3px 3px 3px 0;
}

/* Stats */
.stat-grid { display: flex; gap: 1rem; margin-top: 0.8rem; }
.stat-item {
    flex: 1;
    background: var(--cream);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.stat-num {
    font-family: 'Lora', serif;
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--accent);
}
.stat-lbl { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--paper); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--warm-mid); }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "db_ready" not in st.session_state:
    st.session_state.db_ready = os.path.exists("chroma_db")
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {"pages": 0, "chunks": 0}
if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📖 RAG Assistant")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
1. **Upload** your PDF document
2. **Index** it into a vector database
3. **Ask** questions in natural language
4. Get **context-aware answers** instantly
""")
    st.markdown("---")
    st.markdown("### Models")
    st.markdown("""
- 🟢 **Embeddings** — HuggingFace (all-MiniLM-L6-v2)
- 🟠 **LLM** — Mistral Small
- 🗄 **Vector DB** — ChromaDB (MMR)
""")
    st.markdown("---")
    st.markdown("### Session")
    q_count = len(st.session_state.history)
    st.markdown(f"Questions asked: **{q_count}**")
    if st.button("🗑 Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI · RAG · Document Intelligence</div>
    <div class="hero-title">📖 Book Assistant</div>
    <div class="hero-sub">Upload any PDF — ask anything inside it.</div>
</div>
""", unsafe_allow_html=True)

# ── Two-column layout ──────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.6], gap="large")

# ─── LEFT: Upload & Index ─────────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="section-label">Step 1 — Upload Document</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop your PDF here",
        type="pdf",
        label_visibility="collapsed",
    )

    if uploaded_file:
        fname = uploaded_file.name
        fsize = round(uploaded_file.size / 1024, 1)
        st.markdown(f"""
<div class="card card-accent" style="margin-top:0.8rem">
    <div style="font-weight:600;font-size:0.95rem">📄 {fname}</div>
    <div style="color:var(--muted);font-size:0.8rem;margin-top:2px">{fsize} KB · PDF</div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:1rem">Step 2 — Build Index</div>', unsafe_allow_html=True)

        if st.button("⚡ Create Vector Database", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                file_path = tmp.name

            with st.spinner("Loading & chunking document…"):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)

            with st.spinner("Embedding & storing vectors…"):
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="chroma_db",
                )
                vectorstore.persist()

            st.session_state.db_ready = True
            st.session_state.doc_stats = {"pages": len(docs), "chunks": len(chunks)}
            st.success("Vector database ready!")

    # Stats card
    if st.session_state.db_ready:
        p = st.session_state.doc_stats["pages"]
        c = st.session_state.doc_stats["chunks"]
        pg_display = p if p else "—"
        ch_display = c if c else "—"
        st.markdown(f"""
<div class="card" style="margin-top:1rem">
    <div class="section-label">Index Stats</div>
    <div class="stat-grid">
        <div class="stat-item">
            <div class="stat-num">{pg_display}</div>
            <div class="stat-lbl">Pages</div>
        </div>
        <div class="stat-item">
            <div class="stat-num">{ch_display}</div>
            <div class="stat-lbl">Chunks</div>
        </div>
        <div class="stat-item">
            <div class="stat-num">4</div>
            <div class="stat-lbl">Top-K</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div style="display:flex;align-items:center;gap:8px;margin-top:0.5rem">
    <span style="width:8px;height:8px;border-radius:50%;background:#4a7c59;display:inline-block"></span>
    <span style="font-size:0.82rem;color:#4a7c59;font-weight:600">Index active · MMR retrieval enabled</span>
</div>
""", unsafe_allow_html=True)

# ─── RIGHT: Q&A ───────────────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="section-label">Step 3 — Ask Questions</div>', unsafe_allow_html=True)

    if not st.session_state.db_ready:
        st.markdown("""
<div class="card" style="text-align:center;padding:3rem 2rem;color:var(--muted)">
    <div style="font-size:2.5rem;margin-bottom:1rem">💬</div>
    <div style="font-weight:500">No index yet</div>
    <div style="font-size:0.85rem;margin-top:0.4rem">Upload a PDF and build the vector database first.</div>
</div>
""", unsafe_allow_html=True)
    else:
        query = st.text_input(
            "Your question",
            placeholder="e.g. What is the main argument in chapter 3?",
            label_visibility="collapsed",
        )

        if query:
            with st.spinner("Retrieving relevant context…"):
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectorstore = Chroma(
                    persist_directory="chroma_db",
                    embedding_function=embeddings,
                )
                retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
                )
                docs = retriever.invoke(query)
                context = "\n\n".join([doc.page_content for doc in docs])

            with st.spinner("Generating answer…"):
                llm = ChatMistralAI(model="mistral-small-2506")
                prompt = ChatPromptTemplate.from_messages([
                    ("system",
                     "You are a helpful AI assistant.\n\n"
                     "Use ONLY the provided context to answer the question.\n\n"
                     "If the answer is not present in the context, "
                     'say: "I could not find the answer in the document."'),
                    ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
                ])
                final_prompt = prompt.invoke({"context": context, "question": query})
                response = llm.invoke(final_prompt)

            answer = response.content
            pages = [
                doc.metadata.get("page", "?") + 1
                if isinstance(doc.metadata.get("page"), int) else "?"
                for doc in docs
            ]
            st.session_state.history.insert(0, {"q": query, "a": answer, "pages": pages})

        # Conversation history
        for item in st.session_state.history:
            page_chips = "".join(f'<span class="source-chip">p.{p}</span>' for p in item["pages"])
            st.markdown(f"""
<div class="card" style="margin-bottom:1.4rem">
    <div style="font-weight:600;font-size:0.95rem;margin-bottom:0.8rem;display:flex;align-items:flex-start;gap:8px">
        <span style="color:var(--accent);font-size:1rem;flex-shrink:0">Q</span>
        <span>{item["q"]}</span>
    </div>
    <div class="answer-box">{item["a"]}</div>
    <div style="margin-top:0.75rem">
        <span style="font-size:0.72rem;color:var(--muted);font-weight:600;letter-spacing:0.08em;text-transform:uppercase">Sources · page </span>
        {page_chips}
    </div>
</div>
""", unsafe_allow_html=True)

        if not st.session_state.history:
            st.markdown("""
<div class="card" style="text-align:center;padding:2.5rem 2rem;color:var(--muted)">
    <div style="font-size:2rem;margin-bottom:0.8rem">🔍</div>
    <div style="font-weight:500">Ready to answer</div>
    <div style="font-size:0.85rem;margin-top:0.4rem">Type your question above to get started.</div>
</div>
""", unsafe_allow_html=True)
