"""streamlit_demo.py – clean wording, same visual style.
Run:
    streamlit run streamlit_demo.py
"""
from __future__ import annotations

import json
import streamlit as st

# Try import according to project layout
try:
    from qa_pipeline import QAPipeline
except ModuleNotFoundError:
    from py_files.QA_Pipeline import QAPipeline

# ─── Page config ──────────────────────────────────────────────────────────── #
st.set_page_config(
    page_title="AI Assistant Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS (unchanged) ──────────────────────────────────────────────── #
st.markdown(
    """
    <style>
    html, body, .main {
        background: linear-gradient(135deg, #191e2b 0%, #232d39 100%) !important;
        color: #f2f2f2 !important;
        font-family: 'Inter', 'Segoe UI', 'Roboto', 'sans-serif' !important;
        min-height: 100vh;
    }
    .block-container { padding-top: 20px !important; padding-bottom: 12px !important; max-width: 960px; }
    .fancy-header-bar {
        display: flex; align-items: center; justify-content: center; gap: 12px;
        background: linear-gradient(90deg,#5cfbf1 0%,#009efd 100%);
        border-radius: 19px; padding: 1.05em 1.8em; margin-bottom: 1.6em;
        box-shadow: 0 8px 40px #0fa47a30; max-width: 760px; margin-left: auto; margin-right: auto;
    }
    .fancy-header-title { font-size: 2em; font-weight: 700; letter-spacing: 0.03em; color: #232d39; }
    @keyframes float-in {0%{opacity:0;transform:translateY(40px) scale(.96);}100%{opacity:1;transform:translateY(0) scale(1);}}
    .custom-card{
        background: rgba(35,38,49,.97); border-radius:24px;
        padding:2em 2.2em 1.5em; margin:0.8em 0 1.5em;
        box-shadow:0 16px 64px #0fa47a22,0 2px 12px #161a2180;
        border:2px solid #232d39; font-size:1.2em;
        backdrop-filter: blur(10px); transition: box-shadow .21s;
    }
    .custom-card:hover{box-shadow:0 4px 64px #3ad4f6aa,0 2px 14px #0fa47a55;border:2px solid #009efd;}
    .stTextInput>div>div>input{
        background:linear-gradient(90deg,#232631 70%,#222c37 100%);
        color:#fafbfc !important; border-radius:15px; border:2px solid #0fa47a;
        font-size:1.1em; padding:1.1em 1.8em !important; margin:.4em 0 1em;
    }
    .stTextInput>div>div>input:focus{border:2px solid #009efd !important;}
    .stButton>button{
        background:linear-gradient(90deg,#0fa47a 10%,#009efd 80%); color:#fff;
        font-weight:600; border-radius:15px; padding:0.9em 3em; margin:.5em 0 1.8em;
        font-size:1.15em; border:none; transition:transform .12s,box-shadow .13s,background .18s;
    }
    .stButton>button:hover{background:linear-gradient(90deg,#5cfbf1 10%,#009efd 100%); color:#222c37 !important; transform:translateY(-2px) scale(1.05);}
    .stExpander{background:linear-gradient(90deg,#1c222e,#232d39 120%); border-radius:17px !important; border:1.6px solid #0fa47a55 !important; margin:.6em 0 0.9em;}
    .stExpanderHeader{font-size:1.05em !important; color:#00e0d6 !important; font-weight:700 !important;}
    .my-json-block{
        background:linear-gradient(100deg,#161b24 60%,#232d39 100%);
        border-radius:13px; border:2px solid #009efd33; box-shadow:0 2px 24px #009efd11,0 1.5px 10px #0fa47a12;
        padding:1em 1.2em; overflow-x:auto; font-size:0.98em; font-family:'Fira Mono','Consolas','Roboto Mono',monospace; color:#50e6ff;
    }
    .my-json-block-title{color:#4ee3c5; font-weight:600; font-size:0.95em; margin-bottom:0.6em;}
    .custom-footer{text-align:center; font-size:0.95em; color:#b0b8c7; margin:2.4em 0 0.2em;}
    [data-testid="stAppViewContainer"]{background:none !important;}
    .main .block-container{max-width:980px !important; padding-top:16px !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Header ──────────────────────────────────────────────────────────────── #
st.markdown(
    '<div class="fancy-header-bar"><span class="fancy-header-title">AI Assistant Q&A Pipeline</span></div>',
    unsafe_allow_html=True,
)

# ─── Pipeline cache ──────────────────────────────────────────────────────── #
@st.cache_resource(show_spinner="Loading pipeline …")
def get_pipeline():
    return QAPipeline()

pipeline = get_pipeline()

# ─── Input ───────────────────────────────────────────────────────────────── #
user_query = st.text_input(
    "Enter your question",
    value="",
    placeholder="Type your question here…",
)

# ─── Helper: JSON block ──────────────────────────────────────────────────── #

def json_block(obj, title="") -> str:
    pretty = json.dumps(obj, indent=4, ensure_ascii=False)
    title_html = f'<div class="my-json-block-title">{title}</div>' if title else ""
    return f"<div class='my-json-block'>{title_html}<pre>{pretty}</pre></div>"

# ─── Main action ─────────────────────────────────────────────────────────── #
if st.button("Get Answer", use_container_width=True):
    if not user_query.strip():
        st.warning("Please enter a question first.")
        st.stop()
    with st.spinner("Generating answer …"):
        answer, rag_payload = pipeline.run_with_payload(user_query)

    st.markdown(
        f'<div class="custom-card"><b>Answer:</b><br>'
        f'<div style="font-size:1.1em; margin-top:12px;">{answer}</div></div>',
        unsafe_allow_html=True,
    )

    with st.expander("RAG Payload", expanded=False):
        try:
            payload_obj = json.loads(rag_payload) if isinstance(rag_payload, str) else rag_payload
        except Exception:
            payload_obj = rag_payload
        st.markdown(json_block(payload_obj, "Context provided to the model"), unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="custom-card" style="background:#171c25;">'
        '<i>Enter a question above and click <b>Get Answer</b>.</i></div>',
        unsafe_allow_html=True,
    )

# ─── Footer ──────────────────────────────────────────────────────────────── #
st.markdown(
    '<div class="custom-footer">© 2025 QAPipeline &nbsp;|&nbsp; Streamlit Demo</div>',
    unsafe_allow_html=True,
)
