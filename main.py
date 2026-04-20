import streamlit as st
from DocumentExtraction import Extractedfiles
from procecssor import InvertedIndex
from Queries import Queries

st.set_page_config(
    page_title="VSM Speech Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

.stApp {
    background-color: #0d0d0d;
    color: #e8e3d5;
}

.stTextInput > div > div > input {
    background-color: #1a1a1a !important;
    border: 1px solid #333 !important;
    border-radius: 4px !important;
    color: #e8e3d5 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 1rem !important;
    padding: 0.6rem 1rem !important;
}

.stTextInput > div > div > input:focus {
    border-color: #c8a96e !important;
    box-shadow: 0 0 0 2px rgba(200, 169, 110, 0.15) !important;
}

.stButton > button {
    background-color: #c8a96e !important;
    color: #0d0d0d !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.5rem 1.5rem !important;
    transition: background-color 0.2s ease !important;
}

.stButton > button:hover {
    background-color: #e0c080 !important;
}

[data-testid="metric-container"] {
    background-color: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 1rem;
}

[data-testid="metric-container"] label {
    color: #888 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #c8a96e !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
}

.result-card {
    background-color: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-left: 3px solid #c8a96e;
    border-radius: 4px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-family: 'DM Mono', monospace;
    color: #e8e3d5;
}

.result-card span {
    color: #c8a96e;
    font-weight: 500;
}

.header-strip {
    background: linear-gradient(90deg, #c8a96e22, transparent);
    border-left: 3px solid #c8a96e;
    padding: 0.5rem 1rem;
    margin-bottom: 1.5rem;
    border-radius: 0 4px 4px 0;
}

.tag {
    display: inline-block;
    background-color: #c8a96e22;
    color: #c8a96e;
    border: 1px solid #c8a96e44;
    border-radius: 3px;
    padding: 0.2rem 0.55rem;
    font-size: 0.8rem;
    margin: 0.2rem;
    font-family: 'DM Mono', monospace;
}

hr {
    border-color: #2a2a2a !important;
}

.stAlert {
    border-radius: 4px !important;
    font-family: 'DM Mono', monospace !important;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Building inverted index — please wait...")
def load_system():
    processor = InvertedIndex()
    processor.documentProcessing()
    return processor, Queries(processor)


processor, query_engine = load_system()

# Header
st.markdown("""
<div class="header-strip">
    <h1 style="margin:0; font-size:1.8rem; color:#e8e3d5;">
        VSM Speech Retrieval
    </h1>
    <p style="margin:0.2rem 0 0; font-size:0.82rem; color:#888;">
        Vector Space Model &nbsp;·&nbsp; TF-IDF &nbsp;·&nbsp; Cosine Similarity &nbsp;·&nbsp; 56 Trump Speeches
    </p>
</div>
""", unsafe_allow_html=True)

# Search row
col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input(
        label="query",
        label_visibility="collapsed",
        placeholder="e.g.  Hillary Clinton  |  pakistan afghanistan  |  energy revolution",
        key="query_input",
    )
with col_btn:
    search_clicked = st.button("Search", use_container_width=True)

st.markdown("---")

# Results
if search_clicked and query.strip():
    with st.spinner("Running VSM retrieval…"):
        result_set = query_engine.process_query(query.strip())

    results = sorted(result_set, key=lambda x: int(x)) if result_set else []

    m1, m2, m3 = st.columns(3)
    m1.metric("Documents Found", len(results))
    m2.metric("Documents Not Matched", 56 - len(results))
    m3.metric("Coverage", f"{round(len(results) / 56 * 100, 1)}%")

    st.markdown("")

    if results:
        st.markdown("#### Matching Document IDs")
        tags_html = "".join(
            f'<span class="tag">Speech_{doc_id}</span>' for doc_id in results
        )
        st.markdown(tags_html, unsafe_allow_html=True)

        st.markdown("")

        with st.expander("Detailed Results", expanded=False):
            for i, doc_id in enumerate(results, 1):
                st.markdown(
                    f'<div class="result-card"><span>#{i}</span> &nbsp; Document ID: <span>Speech_{doc_id}</span></div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("No documents matched your query. Try different or broader terms.")

elif search_clicked and not query.strip():
    st.warning("Please enter a query.")

else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0; color:#444;">
        <div style="font-size:3rem;">🔎</div>
        <p style="font-family:'Syne',sans-serif; font-size:1.1rem; margin-top:0.5rem;">
            Enter a query above to search the speech corpus.
        </p>
    </div>
    """, unsafe_allow_html=True)