"""
MicroRAG Labs — Hub + Module Platform
A growing set of bioinformatics AI research tools. MicroRAG is Module 1 (live).

Deployment notes (Hugging Face Spaces):
- Requires bv_literature_corpus.csv in the same repo
- Requires GEMINI_API_KEY set as a Space secret (Settings -> Variables and secrets)
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import chromadb
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from google import genai as google_genai

st.set_page_config(page_title="MicroRAG Labs", page_icon="🧬", layout="wide")

# ================= GLOBAL THEME =================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0A0912;
    --dish: #131120;
    --magenta: #FF3EC9;
    --yellow: #FFC145;
    --text-primary: #F4F2FA;
    --text-muted: #7A7690;
    --border: #262238;
  }
  .stApp { background-color: var(--bg); color: var(--text-primary); font-family: 'IBM Plex Sans', sans-serif; }
  h1, h2, h3, .eyebrow, code { font-family: 'IBM Plex Mono', monospace !important; }
  .eyebrow {
    font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--magenta); margin-bottom: 4px;
  }
  .grad-title {
    font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 2.1rem;
    background: linear-gradient(90deg, var(--magenta), var(--yellow));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
  }
  .subtitle-muted { color: var(--text-muted); font-size: 0.95rem; margin-top: 2px; }
  div[data-testid="stMetric"] {
    background-color: rgba(255,62,201,0.06);
    border: 1px solid rgba(255,62,201,0.25);
    border-radius: 10px; padding: 10px 14px;
  }
  div.stButton > button {
    background: linear-gradient(90deg, var(--magenta), var(--yellow));
    color: #150a12; font-family: 'IBM Plex Mono', monospace; font-weight: 600;
    border: none; border-radius: 8px;
  }
  .safety-pass {
    background-color: rgba(255,193,69,0.10); border-left: 4px solid var(--yellow);
    padding: 10px 14px; border-radius: 6px; margin: 8px 0;
  }
  .safety-fail {
    background-color: rgba(255,62,201,0.10); border-left: 4px solid var(--magenta);
    padding: 10px 14px; border-radius: 6px; margin: 8px 0;
  }
</style>
""", unsafe_allow_html=True)

DISTANCE_THRESHOLD = 0.9

# ================= NAVIGATION (query-param based router) =================
if "module" not in st.query_params:
    st.query_params["module"] = "hub"
current_module = st.query_params.get("module", "hub")


# ================= PIPELINE (cached, only loaded when MicroRAG module is opened) =================
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
  2. If the sources DO address the question but the evidence is mixed, preliminary, or inconclusive, this is NOT insufficient evidence — you MUST answer the question and explicitly describe the evidence as uncertain/preliminary/mixed, citing sources. Reporting "the evidence is currently uncertain" is a complete, valid answer — it is not a refusal.
- Only use INSUFFICIENT_EVIDENCE when the sources are genuinely unrelated to the question's topic.

SOURCES:
{context_block}

QUESTION: {query}

ANSWER (with citations):"""
    return prompt


def ask_microrag(query, embed_model, collection, client, n_results=5, distance_threshold=DISTANCE_THRESHOLD):
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

    if "query_log" not in st.session_state:
        st.session_state.query_log = []
    st.session_state.query_log.append({
        'query': query[:80] + ("..." if len(query) > 80 else ""),
        'distance': top_distance,
        'fallback_triggered': result[3]
    })
    return result


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
    return {'metrics': metrics, 'lactobacillus_total': round(lacto_total, 3),
            'community_state': community_state, 'generated_query': query_text}


def make_abundance_chart(profile_df):
    df = profile_df.copy().sort_values('relative_abundance', ascending=True)
    colors = ['#FFC145' if 'lactobacillus' in t.lower() else '#FF3EC9' for t in df['taxon']]
    fig = go.Figure(go.Bar(
        x=df['relative_abundance'] * 100, y=df['taxon'], orientation='h',
        marker_color=colors, text=[f"{v*100:.1f}%" for v in df['relative_abundance']], textposition='auto'
    ))
    fig.update_layout(
        title="Taxa Relative Abundance (yellow = protective Lactobacillus, magenta = other)",
        xaxis_title="Relative Abundance (%)", yaxis_title="",
        height=max(300, len(df) * 40), margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color="#F4F2FA"
    )
    return fig


def make_distance_gauge(distance, threshold=DISTANCE_THRESHOLD):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=distance, number={'valueformat': '.3f'},
        title={'text': "Retrieval Distance (lower = more relevant)"},
        gauge={
            'axis': {'range': [0, 1.6]}, 'bar': {'color': "#FF3EC9"},
            'steps': [
                {'range': [0, threshold], 'color': "rgba(255,193,69,0.25)"},
                {'range': [threshold, 1.6], 'color': "rgba(255,62,201,0.25)"},
            ],
            'threshold': {'line': {'color': "#FF3EC9", 'width': 3}, 'thickness': 0.8, 'value': threshold}
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=10),
                       paper_bgcolor='rgba(0,0,0,0)', font_color="#F4F2FA")
    return fig


# ================= HUB VIEW =================
def render_hub():
    st.markdown('<div class="eyebrow">// bioinformatics research platform</div>', unsafe_allow_html=True)
    st.markdown('<p class="grad-title">MicroRAG Labs</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-muted">A growing plate of evidence-grounded AI research tools — each module a culture, validated before going live.</p>', unsafe_allow_html=True)
    st.write("")

    st.markdown("""
    <style>
      .dish {
        position: relative; background: #131120; border: 1px solid #262238; border-radius: 28px;
        padding: 64px 40px; display: flex; justify-content: space-evenly; align-items: center;
        flex-wrap: wrap; gap: 40px; overflow: hidden;
      }
      .rim-label { position: absolute; top: 18px; left: 24px; font-family: 'IBM Plex Mono', monospace;
        font-size: 10px; color: #7A7690; letter-spacing: 0.08em; }
      .culture { display: flex; flex-direction: column; align-items: center; text-align: center; }
      .circle { border-radius: 50%; transition: transform 0.35s ease, box-shadow 0.35s ease; }
      .culture.active a { text-decoration: none; }
      .culture.active .circle {
        width: 150px; height: 150px;
        background: radial-gradient(circle at 32% 28%, #FFE0A3 0%, #FFC145 32%, #FF3EC9 78%, #7A1F63 100%);
        box-shadow: 0 0 30px rgba(255,62,201,0.45), 0 0 60px rgba(255,193,69,0.2);
      }
      .culture.active:hover .circle { transform: scale(1.08); box-shadow: 0 0 45px rgba(255,62,201,0.65), 0 0 90px rgba(255,193,69,0.35); }
      .culture.dormant .circle { width: 96px; height: 96px; background: transparent; border: 1.5px dashed #262238; }
      .label { margin-top: 16px; font-family: 'IBM Plex Mono', monospace; }
      .culture.active .label .name { font-size: 16px; font-weight: 600; color: #F4F2FA; }
      .culture.active .label .desc { font-size: 11px; color: #7A7690; max-width: 170px; margin: 5px auto 0; line-height: 1.5; }
      .culture.active .label .status { font-size: 9px; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 8px; display: inline-block;
        background: linear-gradient(90deg, #FF3EC9, #FFC145); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
      .culture.dormant .label .name { font-size: 13px; color: #7A7690; font-weight: 500; }
      .culture.dormant .label .status { font-size: 9px; color: #4E4A61; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 5px; display: block; }
    </style>

    <div class="dish">
      <div class="rim-label">specimen plate · 3 cultures</div>
      <div class="culture active">
        <a href="?module=microrag" target="_self">
          <div class="circle"></div>
        </a>
        <div class="label">
          <div class="name">MicroRAG</div>
          <div class="desc">Vaginal microbiome risk explanation, grounded in cited literature</div>
          <span class="status">● active culture — click to open</span>
        </div>
      </div>
      <div class="culture dormant">
        <div class="circle"></div>
        <div class="label"><div class="name">RepurposeAI</div><span class="status">not yet inoculated</span></div>
      </div>
      <div class="culture dormant">
        <div class="circle"></div>
        <div class="label"><div class="name">OncoPath</div><span class="status">not yet inoculated</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")


# ================= MICRORAG MODULE VIEW =================
def render_microrag():
    if st.button("← Back to Labs"):
        st.query_params["module"] = "hub"
        st.rerun()

    embed_model, collection, client, corpus_size = load_pipeline()

    st.markdown('<p class="grad-title" style="font-size:1.8rem;">🧬 MicroRAG</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-muted">Retrieval-Augmented Generation for Vaginal Microbiome–Disease Risk Explanation</p>', unsafe_allow_html=True)
    st.warning("⚠️ Research/educational tool only — not a diagnostic device. Does not replace professional medical advice.")

    tab1, tab2, tab3 = st.tabs(["💬 Ask a Question", "📊 Upload Microbiome Profile", "⚙️ System Internals"])

    with tab1:
        st.subheader("Ask about bacterial vaginosis and vaginal microbiome research")
        query = st.text_input("Your question:", placeholder="e.g. How does Gardnerella vaginalis form biofilms?")

        if st.button("Get Answer", type="primary", key="query_btn") and query:
            with st.spinner("Retrieving evidence and generating explanation..."):
                answer, retrieved, distance, is_fallback = ask_microrag(query, embed_model, collection, client)

            col_main, col_gauge = st.columns([2, 1])
            with col_gauge:
                st.plotly_chart(make_distance_gauge(distance), use_container_width=True)
                if is_fallback:
                    st.markdown('<div class="safety-fail">🛑 Safety net triggered — generation skipped</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="safety-pass">✅ Passed relevance threshold</div>', unsafe_allow_html=True)
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
        st.caption("CSV with two columns: `taxon` and `relative_abundance`")
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
                    answer, retrieved, distance, is_fallback = ask_microrag(result['generated_query'], embed_model, collection, client)

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
        st.caption("What's actually happening under the hood.")

        st.markdown("#### Pipeline Architecture")
        st.code("""
Query / Microbiome Profile
        │
        ▼
Query Processing
        │
        ▼
Retrieval  (MiniLM embeddings → ChromaDB over %d-abstract PubMed corpus)
        │
        ▼
Safety Gate 1  (blocks generation if best match distance > %.2f)
        │
        ▼
Grounded Generation  (Gemini, citation-required prompt)
        │
        ▼
Answer + Citations
""" % (corpus_size, DISTANCE_THRESHOLD), language="text")

        st.markdown("#### Evaluation Metrics")
        eval_col1, eval_col2, eval_col3 = st.columns(3)
        eval_col1.metric("Faithfulness", "1.00")
        eval_col2.metric("Context Precision", "0.79")
        eval_col3.metric("Answer Relevance", "1.00")

        st.markdown("#### Distance Threshold Calibration")
        calib_fig = go.Figure()
        calib_fig.add_trace(go.Box(y=[0.49, 0.51, 0.56, 0.60, 0.65], name="Known-relevant", marker_color="#FFC145"))
        calib_fig.add_trace(go.Box(y=[1.49, 1.50, 1.53, 1.54, 1.55], name="Known-irrelevant", marker_color="#FF3EC9"))
        calib_fig.add_hline(y=DISTANCE_THRESHOLD, line_dash="dash", line_color="white",
                             annotation_text=f"Threshold: {DISTANCE_THRESHOLD}")
        calib_fig.update_layout(height=350, yaxis_title="Retrieval Distance",
                                 plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="#F4F2FA")
        st.plotly_chart(calib_fig, use_container_width=True)

        st.markdown("#### Live Session Query Log")
        if st.session_state.get("query_log"):
            log_df = pd.DataFrame(st.session_state.query_log)
            log_df['status'] = log_df['fallback_triggered'].map({True: "🛑 Fallback", False: "✅ Answered"})
            st.dataframe(log_df[['query', 'distance', 'status']], use_container_width=True)
        else:
            st.info("No queries run yet this session.")


# ================= ROUTER =================
if current_module == "microrag":
    render_microrag()
else:
    render_hub()
