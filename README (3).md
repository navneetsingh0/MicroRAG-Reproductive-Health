# 🧬 MicroRAG

**A Retrieval-Augmented Generation System for Vaginal Microbiome–Disease Risk Explanation in Reproductive Health**

MicroRAG connects vaginal/gut microbiome abundance data with biomedical literature to generate evidence-grounded, citation-backed explanations of reproductive health risks — starting with bacterial vaginosis (BV). Every claim the system makes is traceable to a real, retrievable research source. When the evidence doesn't support a confident answer, it says so, instead of guessing.

**🔗 Live Demo:** [microragforlifescience.streamlit.app](https://microragforlifescience.streamlit.app/)

---

## The Problem

Interpreting a microbiome profile against the reproductive-health literature today means manually searching, reading, and synthesizing across dozens of scattered papers. A general-purpose LLM query is unreliable here — it's prone to hallucinating microbiology facts with no real source behind them. MicroRAG solves this by grounding every generated explanation in literature that was actually retrieved and can be cited.

## Features

- **💬 Natural-language Q&A** — ask a research question, get a cited, evidence-grounded answer
- **📊 Microbiome profile analysis** — upload a taxon/abundance CSV, get diversity metrics, a community-state classification, and a citation-backed risk explanation
- **🛡️ Two-layer hallucination safeguard**:
  1. A code-level retrieval-distance threshold that blocks generation entirely when no sufficiently relevant literature exists
  2. Citation-required prompting that explicitly distinguishes "no evidence" from "evidence exists but is mixed/uncertain" — so the system reports genuine scientific uncertainty rather than either fabricating confidence or refusing unnecessarily

## How It Works

```
Query or Microbiome Profile
        │
        ▼
 Query Processing (parses text, or converts abundance
 data into diversity metrics + a retrieval-ready summary)
        │
        ▼
 Retrieval (sentence-transformer embeddings → ChromaDB
 semantic search over a 350-abstract PubMed corpus)
        │
        ▼
 Grounded Generation (Google Gemini, citation-required
 prompting, distance-gated fallback)
        │
        ▼
 Evidence-Backed Answer + Source Citations
```

## Tech Stack

| Component | Tool |
|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector database | ChromaDB |
| Literature source | PubMed / NCBI Entrez (via Biopython) |
| Generation | Google Gemini API |
| Bioinformatics | scikit-bio, pandas, numpy (Shannon/Simpson diversity metrics) |
| Interface | Streamlit |
| Evaluation | Custom LLM-as-judge harness (context precision, faithfulness, answer relevance) |

## Evaluation Results

Tested across a labeled query set spanning mechanism, risk-factor, treatment, and adjacent-topic questions:

| Metric | Score |
|---|---|
| Faithfulness | **1.00** — zero unsupported claims across every test query |
| Answer Relevance | **1.00** |
| Context Precision | **0.79** |

The system was also stress-tested against out-of-scope queries and unusual/edge-case microbiome profiles, correctly triggering its evidence-insufficiency fallback rather than fabricating an answer.

## Project Structure

```
├── app.py                      # Streamlit application
├── requirements.txt            # Python dependencies
├── bv_literature_corpus.csv    # PubMed literature corpus (350 abstracts)
└── README.md
```

## Running Locally

```bash
git clone https://github.com/navneetsingh0/MicroRAG-Reproductive-Health.git
cd MicroRAG-Reproductive-Health
pip install -r requirements.txt
```

Create a `.streamlit/secrets.toml` file with your own Gemini API key:
```toml
GEMINI_API_KEY = "your-key-here"
```

Then run:
```bash
streamlit run app.py
```

## ⚠️ Disclaimer

This tool provides research/educational risk explanations grounded in published literature. It is **not** a diagnostic tool and does not replace professional medical advice.

## Project Context

Developed as a final-year BTech NTCC (Non-Teaching Credit Course) project — Bioinformatics with an AI/ML minor, Amity University, Noida — under the guidance of **Prof. Abhishek Sengupta** (Systems Biology, Microbiomics, Reproductive Health).

**Author:** Navneet Singh
