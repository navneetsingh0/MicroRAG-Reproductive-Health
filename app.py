"""
MicroRAG — Retrieval-Augmented Generation System for Vaginal Microbiome–Disease Risk Explanation
Streamlit demo interface.

Deployment notes:
- Requires bv_literature_corpus.csv in the same repo (exported from your Colab notebook)
- Requires GEMINI_API_KEY set in Streamlit Cloud's Secrets manager (Settings -> Secrets)
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai as google_genai

# ---------- Page config ----------
st.set_page_config(page_title="MicroRAG", page_icon="🧬", layout="wide")

# ---------- Cached resource loading (runs once, not on every interaction) ----------

@st.cache_resource
def load_pipeline():
    # Load corpus
    df = pd.read_csv("bv_literature_corpus.csv")

    def clean_text(text):
        text = re.sub(r'<[^>]+>', '', str(text))
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['title_clean'] = df['title'].apply(clean_text)
    df['abstract_clean'] = df['abstract'].apply(clean_text)
    df['combined_text'] = df['title_clean'] + ". " + df['abstract_clean']

    # Embedding model
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embed_model.encode(df['combined_text'].tolist(), show_progress_bar=False, batch_size=32)

    # In-memory ChromaDB (ephemeral client — rebuilt fresh each app start, fine for this corpus size)
    chroma_client = chromadb.EphemeralClient()
    collection = chroma_client.get_or_create_collection(name="bv_literature")
    collection.add(
        ids=df['pmid'].astype(str).tolist(),
        embeddings=embeddings.tolist(),
        documents=df['combined_text'].tolist(),
        metadatas=df[['title_clean', 'journal', 'pmid']].to_dict('records')
    )

    # Gemini client
    client = google_genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

    return embed_model, collection, client


embed_model, collection, client = load_pipeline()


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


def ask_microrag(query, n_results=5, distance_threshold=0.9):
    query_embedding = embed_model.encode([query])
    retrieved = collection.query(query_embeddings=query_embedding.tolist(), n_results=n_results)
    top_distance = retrieved['distances'][0][0]

    if top_distance > distance_threshold:
        answer = "INSUFFICIENT_EVIDENCE: No sufficiently relevant literature was found in the corpus for this query."
        return answer, retrieved, top_distance, True

    prompt = build_grounded_prompt(query, retrieved)
    response = client.models.generate_content(model="gemini-flash-latest", contents=prompt)
    return response.text, retrieved, top_distance, False


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


# ---------- UI ----------

st.title("🧬 MicroRAG")
st.caption("Retrieval-Augmented Generation for Vaginal Microbiome–Disease Risk Explanation in Reproductive Health")
st.warning("⚠️ This tool provides research/educational risk explanations grounded in published literature. It is NOT a diagnostic tool and does not replace professional medical advice.")

tab1, tab2 = st.tabs(["💬 Ask a Question", "📊 Upload Microbiome Profile"])

with tab1:
    st.subheader("Ask about bacterial vaginosis and vaginal microbiome research")
    query = st.text_input("Your question:", placeholder="e.g. How does Gardnerella vaginalis form biofilms?")

    if st.button("Get Answer", type="primary", key="query_btn") and query:
        with st.spinner("Retrieving evidence and generating explanation..."):
            answer, retrieved, distance, is_fallback = ask_microrag(query)

        if is_fallback:
            st.info(f"**{answer}**\n\n(Retrieval distance {distance:.3f} exceeded the confidence threshold — no generation was attempted.)")
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
        st.dataframe(profile_df)

        if st.button("Analyze Profile", type="primary", key="profile_btn"):
            with st.spinner("Interpreting profile and retrieving evidence..."):
                result = interpret_profile(profile_df)
                answer, retrieved, distance, is_fallback = ask_microrag(result['generated_query'])

            col1, col2, col3 = st.columns(3)
            col1.metric("Lactobacillus %", f"{result['lactobacillus_total']*100:.0f}%")
            col2.metric("Shannon Diversity", result['metrics']['shannon_diversity'])
            col3.metric("Community State", result['community_state'].split('(')[0].strip())

            st.markdown("### Risk Explanation")
            if is_fallback:
                st.info(f"**{answer}**")
            else:
                st.markdown(answer)
                st.markdown("### Sources")
                for doc_id, dist, meta in zip(retrieved['ids'][0], retrieved['distances'][0], retrieved['metadatas'][0]):
                    st.markdown(f"- **[PMID {doc_id}]** {meta['title_clean']} *(distance: {dist:.3f})*")

st.divider()
st.caption("MicroRAG — NTCC Project | Built with Google Gemini, ChromaDB, and sentence-transformers")
