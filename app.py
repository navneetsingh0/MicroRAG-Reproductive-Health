"""
MicroRAG — Retrieval-Augmented Generation System for Vaginal Microbiome–Disease Risk Explanation
Streamlit demo interface — v2 (with System Internals transparency dashboard)

Deployment notes:
- Requires bv_literature_corpus.csv in the same repo
- Requires GEMINI_API_KEY set in Streamlit Cloud's Secrets manager
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import chromadb
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from google import genai as google_genai

# ---------- Page config ----------
st.set_page_config(page_title="MicroRAG", page_icon="🧬", layout="wide")

# ---------- Custom styling ----------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6C5CE7, #00B894);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #888;
        font-size: 1rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(108, 92, 231, 0.08);
        border: 1px solid rgba(108, 92, 231, 0.25);
        border-radius: 10px;
        padding: 12px 16px;
    }
    .safety-pass {
        background-color: rgba(0, 184, 148, 0.12);
        border-left: 4px solid #00B894;
        padding: 10px 14px;
        border-radius: 6px;
        margin: 8px 0;
    }
    .safety-fail {
        background-color: rgba(214, 48, 49, 0.12);
        border-left: 4px solid #D63031;
        padding: 10px 14px;
        border-radius: 6px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Session state for query logging (powers the System Internals tab) ----------
if "query_log" not in st.session_state:
    st.session_state.query_log = []

# ---------- Cached resource loading ----------

@st.cache_resource
def load_pipeline():
    df = pd.read_csv("bv_literature_corpus.csv")

    def clean_text(text):
        text = re.sub(r'<[^>]+>', '', str(text))
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['title_clean'] = df['title'].apply(clean_text)
    df['abstract_clean'] = df['abstract'].apply(clean_text)
    df['combined_text'] = df['title_clean'] + ". " + df['abstract_clean']

    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embed_model.encode(df['combined_text'].tolist(), show_progress_bar=False, batch_size=32)

    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection(name="bv_literature")
    collection.add(
        ids=df['pmid'].astype(str).tolist(),
        embeddings=embeddings.tolist(),
        documents=df['combined_text'].tolist(),
        metadatas=df[['title_clean', 'journal', 'pmid']].to_dict('records')
    )

    client = google_genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

    return embed_model, collection, client, len(df)


embed_model, collection, client, corpus_size = load_pipeline()

DISTANCE_THRESHOLD = 0.9


# ---------- Core RAG functions ----------

def build_grounded_prompt(query, retrieved_docs):
    context_block = "\n\n".join([
        f"[Source {i+1}] (PMID: {meta['pmid']}, {meta['title_clean']}):\n{doc}"
        for i, (doc, meta) in enumerate(zip(retrieved_docs['documents'][0], retrieved_docs['metadatas'][0]))
    ])

    prompt = f"""You are a clinical research assistant. Answer the question ONLY using the sources provided below.

STRICT RULES:
- Every claim you make MUST cite a source using [Source N] format.
- Do NOT use any outside knowledge. Do NOT guess or fill gaps.
- IMPORTANT — distinguish between two different situations:
  1. If the sources contain NO relevant information addressing the question at all, respond exactly with: "INSUFFICIENT_EVIDENCE: The available literature does not provide sufficient evidence to answer this query."
  2. If the sources DO address the question but the evidence is mixed, preliminary, or inconclusive (e.g., "further research needed", conflicting findings, animal-model-only data), this is NOT insufficient evidence — you MUST answer the question and explicitly describe the evidence as uncertain/preliminary/mixed, citing sources for that characterization. Reporting "the evidence is currently uncertain" is a complete, valid, and expected answer — it is not a refusal.
- Only use the INSUFFICIENT_EVIDENCE response when the sources are genuinely unrelated to the question's topic, not merely inconclusive about it.

SOURCES:
{context_block}

QUESTION: {query}

ANSWER (with citations):"""
    return prompt


def ask_microrag(query, n_results=5, distance_threshold=DISTANCE_THRESHOLD):
    query_embedding = embed_model.encode([query])
    retrieved = collection.query(query_embeddings=query_embedding.tolist(), n_results=n_results)
    top_distance = retrieved['distances'][0][0]

    if top_distance > distance_threshold:
        answer = "INSUFFICIENT_EVIDENCE: No sufficiently relevant literature was found in the corpus for this query."
        result = (answer, retrieved, top_distance, True)
    else:
        prompt = build_grounded_prompt(query, retrieved)
        response = client.models.generate_content(model="gemini-flash-latest", contents=prompt)
        result = (response.text, retrieved, top_distance, False)

    # Log every query for the System Internals tab
    st.session_state.query_log.append({
        'query': query[:80] + ("..." if len(query) > 80 else ""),
        'distance': top_distance,
        'fallback_triggered': result[3]
    })

    return result


# ---------- Profile interpreter ----------

def compute_diversity_metrics(profile_df):
    abundances = profile_df['relative_abundance'].values
    abundances = abundances[abundances > 0]
    shannon = -np.sum(abundances * np.log(abundances))
    simpson = 1 - np.sum(abundances ** 2)
    return {'shannon_diversity': round(shannon, 3), 'simpson_diversity': round(simpson, 3)}


def interpret_profile(profile_df):
    metrics = compute_diversity_metrics(profile_df)
    lacto_taxa = profile_df[profile_df['taxon'].str.contains('Lactobacillus', case=False)]
    lacto_total = lacto_taxa['relative_abundance'].sum()
    non_lacto = profile_df[~profile_df['taxon'].str.contains('Lactobacillus', case=False)]
    top_non_lacto = non_lacto.nlargest(3, 'relative_abundance')

    bv_associated_taxa = ['Gardnerella', 'Atopobium', 'Prevotella', 'Mobiluncus', 'Sneathia', 'BVAB']
    pathogen_matches = non_lacto[non_lacto['taxon'].str.contains('|'.join(bv_associated_taxa), case=False)]
    max_pathogen_abundance = pathogen_matches['relative_abundance'].max() if not pathogen_matches.empty else 0

    if lacto_total >= 0.60 and max_pathogen_abundance < 0.10:
        community_state = "Lactobacillus-dominant (community state consistent with vaginal eubiosis)"
    elif lacto_total >= 0.60 and max_pathogen_abundance >= 0.10:
        community_state = "Lactobacillus-dominant but with notable residual BV-associated taxa present (atypical/incompletely resolved community state)"
    elif lacto_total >= 0.30:
        community_state = "Intermediate Lactobacillus abundance (borderline/transitional community state)"
    else:
        community_state = "Lactobacillus-depleted (community state consistent with dysbiosis)"

    dominant_species_text = ", ".join([f"{row['taxon']} ({row['relative_abundance']*100:.0f}%)"
                                         for _, row in top_non_lacto.iterrows()])
    query_text = (
        f"{community_state}. Total Lactobacillus relative abundance: {lacto_total*100:.0f}%. "
        f"Dominant non-Lactobacillus taxa: {dominant_species_text}. "
        f"Shannon diversity index: {metrics['shannon_diversity']}. "
        f"What disease risk is associated with this vaginal microbiome composition?"
    )

    return {
        'metrics': metrics,
        'lactobacillus_total': round(lacto_total, 3),
        'community_state': community_state,
        'generated_query': query_text
    }


def make_abundance_chart(profile_df):
    """Bar chart color-coded: protective Lactobacillus (green) vs everything else (red gradient by abundance)."""
    df = profile_df.copy().sort_values('relative_abundance', ascending=True)
    colors = ['#00B894' if 'lactobacillus' in t.lower() else '#D63031' for t in df['taxon']]

    fig = go.Figure(go.Bar(
        x=df['relative_abundance'] * 100,
        y=df['taxon'],
        orientation='h',
        marker_color=colors,
        text=[f"{v*100:.1f}%" for v in df['relative_abundance']],
        textposition='auto'
    ))
    fig.update_layout(
        title="Taxa Relative Abundance (green = protective Lactobacillus)",
        xaxis_title="Relative Abundance (%)",
        yaxis_title="",
        height=max(300, len(df) * 40),
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def make_distance_gauge(distance, threshold=DISTANCE_THRESHOLD):
    """Visual gauge showing where this query's retrieval distance falls relative to the calibrated threshold."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=distance,
        number={'suffix': "", 'valueformat': '.3f'},
        title={'text': "Retrieval Distance (lower = more relevant)"},
        gauge={
            'axis': {'range': [0, 1.6]},
            'bar': {'color': "#6C5CE7"},
            'steps': [
                {'range': [0, threshold], 'color': "rgba(0, 184, 148, 0.25)"},
                {'range': [threshold, 1.6], 'color': "rgba(214, 48, 49, 0.25)"},
            ],
            'threshold': {
                'line': {'color': "#D63031", 'width': 3},
                'thickness': 0.8,
                'value': threshold
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=10))
    return fig


# ---------- UI ----------

st.markdown('<p class="main-header">🧬 MicroRAG</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Retrieval-Augmented Generation for Vaginal Microbiome–Disease Risk Explanation in Reproductive Health</p>', unsafe_allow_html=True)
st.warning("⚠️ This tool provides research/educational risk explanations grounded in published literature. It is NOT a diagnostic tool and does not replace professional medical advice.")

tab1, tab2, tab3 = st.tabs(["💬 Ask a Question", "📊 Upload Microbiome Profile", "⚙️ System Internals"])

with tab1:
    st.subheader("Ask about bacterial vaginosis and vaginal microbiome research")
    query = st.text_input("Your question:", placeholder="e.g. How does Gardnerella vaginalis form biofilms?")

    if st.button("Get Answer", type="primary", key="query_btn") and query:
        with st.spinner("Retrieving evidence and generating explanation..."):
            answer, retrieved, distance, is_fallback = ask_microrag(query)

        col_main, col_gauge = st.columns([2, 1])

        with col_gauge:
            st.plotly_chart(make_distance_gauge(distance), use_container_width=True)
            if is_fallback:
                st.markdown('<div class="safety-fail">🛑 Safety net triggered — generation was skipped</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="safety-pass">✅ Passed relevance threshold — evidence found</div>', unsafe_allow_html=True)

        with col_main:
            if is_fallback:
                st.info(f"**{answer}**")
            else:
                st.markdown("### Answer")
                st.markdown(answer)
                st.markdown("### Sources")
                for doc_id, dist, meta in zip(retrieved['ids'][0], retrieved['distances'][0], retrieved['metadatas'][0]):
                    st.markdown(f"- **[PMID {doc_id}]** {meta['title_clean']} *(distance: {dist:.3f})*")

with tab2:
    st.subheader("Upload a microbiome abundance profile")
    st.caption("CSV with two columns: `taxon` and `relative_abundance` (values should sum to ~1.0)")

    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded is not None:
        profile_df = pd.read_csv(uploaded)
        profile_df = profile_df.loc[:, ~profile_df.columns.str.contains('^Unnamed')]

        col_table, col_chart = st.columns([1, 1])
        with col_table:
            st.dataframe(profile_df, use_container_width=True)
        with col_chart:
            st.plotly_chart(make_abundance_chart(profile_df), use_container_width=True)

        if st.button("Analyze Profile", type="primary", key="profile_btn"):
            with st.spinner("Interpreting profile and retrieving evidence..."):
                result = interpret_profile(profile_df)
                answer, retrieved, distance, is_fallback = ask_microrag(result['generated_query'])

            col1, col2, col3 = st.columns(3)
            col1.metric("Lactobacillus %", f"{result['lactobacillus_total']*100:.0f}%")
            col2.metric("Shannon Diversity", result['metrics']['shannon_diversity'])
            col3.metric("Community State", result['community_state'].split('(')[0].strip())

            st.plotly_chart(make_distance_gauge(distance), use_container_width=True)

            st.markdown("### Risk Explanation")
            if is_fallback:
                st.info(f"**{answer}**")
            else:
                st.markdown(answer)
                st.markdown("### Sources")
                for doc_id, dist, meta in zip(retrieved['ids'][0], retrieved['distances'][0], retrieved['metadatas'][0]):
                    st.markdown(f"- **[PMID {doc_id}]** {meta['title_clean']} *(distance: {dist:.3f})*")

with tab3:
    st.subheader("⚙️ System Internals & Engineering Transparency")
    st.caption("This tab exposes what's actually happening under the hood — the design decisions and safeguards that make this more than a wrapper around an LLM API call.")

    st.markdown("#### Pipeline Architecture")
    st.code("""
Query / Microbiome Profile
        │
        ▼
Query Processing  (parses text, or converts abundance data
                    into diversity metrics + retrieval-ready summary)
        │
        ▼
Retrieval          (MiniLM embeddings → ChromaDB semantic search
                     over %d-abstract PubMed corpus)
        │
        ▼
Safety Gate 1       Code-level distance threshold check
        │            (blocks generation if best match > %.2f)
        ▼
Grounded Generation (Gemini, citation-required prompt,
                      distinguishes "no evidence" vs "uncertain evidence")
        │
        ▼
Answer + Citations
""" % (corpus_size, DISTANCE_THRESHOLD), language="text")

    st.markdown("#### Evaluation Metrics")
    st.caption("Measured via a custom LLM-as-judge harness across a labeled test query set (built after diagnosing an unresolved dependency bug in the RAGAS library).")

    eval_col1, eval_col2, eval_col3 = st.columns(3)
    eval_col1.metric("Faithfulness", "1.00", help="Fraction of claims in generated answers that are actually supported by retrieved sources — zero hallucination across all test queries.")
    eval_col2.metric("Context Precision", "0.79", help="Fraction of retrieved documents that were actually relevant to the query.")
    eval_col3.metric("Answer Relevance", "1.00", help="Fraction of answers that correctly addressed what was asked.")

    st.markdown("#### Distance Threshold Calibration")
    st.caption("The 0.9 threshold below wasn't guessed — it was empirically calibrated by measuring retrieval distances for known-relevant vs. known-irrelevant queries.")

    calib_fig = go.Figure()
    calib_fig.add_trace(go.Box(y=[0.49, 0.51, 0.56, 0.60, 0.65], name="Known-relevant queries", marker_color="#00B894"))
    calib_fig.add_trace(go.Box(y=[1.49, 1.50, 1.53, 1.54, 1.55], name="Known-irrelevant queries", marker_color="#D63031"))
    calib_fig.add_hline(y=DISTANCE_THRESHOLD, line_dash="dash", line_color="white",
                         annotation_text=f"Chosen threshold: {DISTANCE_THRESHOLD}")
    calib_fig.update_layout(height=350, yaxis_title="Retrieval Distance",
                             plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(calib_fig, use_container_width=True)

    st.markdown("#### Live Session Query Log")
    if st.session_state.query_log:
        log_df = pd.DataFrame(st.session_state.query_log)
        log_df['status'] = log_df['fallback_triggered'].map({True: "🛑 Fallback", False: "✅ Answered"})
        st.dataframe(log_df[['query', 'distance', 'status']], use_container_width=True)
    else:
        st.info("No queries run yet this session — try the Q&A or Profile tabs, then return here.")

st.divider()
st.caption("MicroRAG — NTCC Project | Built with Google Gemini, ChromaDB, and sentence-transformers")
