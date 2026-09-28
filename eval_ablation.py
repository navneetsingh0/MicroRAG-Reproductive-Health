"""
eval_ablation.py - standalone evaluation + ablation for MicroRAG (does NOT import app.py, which runs Streamlit code).

Configs compared on the same query set:
  baseline : original system  (top-5 from ChromaDB, distance gate ON)
  rerank   : + cross-encoder  (top-15 -> re-ranked top-5, distance gate ON)
  no_gate  : ablation         (top-5, distance gate OFF - always calls Gemini; only the prompt-level guard remains)

Usage (run from the repo root, where bv_literature_corpus.csv lives):
  export GEMINI_API_KEY="..."            # Windows PowerShell:  $env:GEMINI_API_KEY="..."
  python eval_ablation.py --limit 3      # smoke test on 3 queries per config
  python eval_ablation.py                # full run; safe to stop and re-run - finished rows are skipped
  python eval_ablation.py --summary      # just re-print the tables from eval_results.csv

NOTE: build_pipeline(), retrieval and build_grounded_prompt() mirror app.py. If you change them there, change them here.
"""
import os, re, sys, json, time, argparse
import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai
from google.genai import types

from eval_queries import TEST_QUERIES

DISTANCE_THRESHOLD = 0.9
RETRIEVE_K, FINAL_K = 15, 5
MODEL = "gemini-flash-latest"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RESULTS_CSV = "eval_results.csv"

CONFIGS = {
    "baseline": dict(rerank=False, gate=True),
    "rerank":   dict(rerank=True,  gate=True),
    "no_gate":  dict(rerank=False, gate=False),
}


# ---------------- pipeline (mirrors app.py) ----------------
def build_pipeline():
    df = pd.read_csv("bv_literature_corpus.csv")
    clean = lambda t: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", str(t))).strip()
    df["title_clean"] = df["title"].apply(clean)
    df["abstract_clean"] = df["abstract"].apply(clean)
    df["combined_text"] = df["title_clean"] + ". " + df["abstract_clean"]

    embed = SentenceTransformer("all-MiniLM-L6-v2")
    emb = embed.encode(df["combined_text"].tolist(), show_progress_bar=False, batch_size=32)
    coll = chromadb.EphemeralClient().get_or_create_collection(name="bv_literature")
    coll.add(ids=df["pmid"].astype(str).tolist(), embeddings=emb.tolist(),
             documents=df["combined_text"].tolist(),
             metadatas=df[["title_clean", "journal", "pmid"]].to_dict("records"))
    return embed, coll, CrossEncoder(RERANKER_NAME)


def build_grounded_prompt(query, retrieved_docs):
    context_block = "\n\n".join([
        f"[Source {i+1}] (PMID: {meta['pmid']}, {meta['title_clean']}):\n{doc}"
        for i, (doc, meta) in enumerate(zip(retrieved_docs['documents'][0], retrieved_docs['metadatas'][0]))
    ])
    return f"""You are a clinical research assistant. Answer the question ONLY using the sources provided below.

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


def rerank(reranker, query, retrieved, top_n):
    scores = reranker.predict([(query, d) for d in retrieved["documents"][0]])
    order = np.argsort(-scores)[:top_n]
    return {k: [[retrieved[k][0][i] for i in order]] for k in ("ids", "documents", "metadatas", "distances")}


# ---------------- Gemini helpers ----------------
def call_gemini(client, prompt, json_mode=False, retries=6):
    for attempt in range(retries):
        try:
            cfg = types.GenerateContentConfig(temperature=0, response_mime_type="application/json") if json_mode else None
            return client.models.generate_content(model=MODEL, contents=prompt, config=cfg).text
        except Exception as e:  # rate limits / transient errors on the free tier
            wait = min(5 * 2 ** attempt, 120)
            print(f"    API error ({type(e).__name__}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("Gemini call kept failing")


def parse_json(text):
    return json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip()))


def judge_answer(client, question, passages, answer):
    ctx = "\n\n".join(f"[Source {i+1}] {p[:1500]}" for i, p in enumerate(passages))
    out = parse_json(call_gemini(client, f"""You are a strict evaluator of a retrieval-augmented answer.

QUESTION:
{question}

SOURCES:
{ctx}

ANSWER:
{answer}

Score two things, each from 0.0 to 1.0:
- faithfulness: fraction of the factual claims in the ANSWER that are directly supported by the SOURCES (1.0 = no unsupported claims).
- answer_relevance: how directly and completely the ANSWER addresses the QUESTION (1.0 = fully on point).
Return ONLY JSON: {{"faithfulness": <float>, "answer_relevance": <float>}}""", json_mode=True))
    return float(out["faithfulness"]), float(out["answer_relevance"])


def judge_context_precision(client, question, passages):
    ctx = "\n\n".join(f"[Passage {i+1}] {p[:1500]}" for i, p in enumerate(passages))
    out = parse_json(call_gemini(client, f"""For each passage, decide whether it contains information that helps answer the question.

QUESTION:
{question}

PASSAGES:
{ctx}

Return ONLY JSON: {{"relevant": [true or false for each passage, in order]}}""", json_mode=True))
    flags = [bool(x) for x in out["relevant"]][:len(passages)]
    return float(np.mean(flags)) if flags else np.nan


# ---------------- run ----------------
def run_one(cfg, item, embed, coll, reranker, client):
    q = item["query"]
    qe = embed.encode([q]).tolist()
    k = RETRIEVE_K if cfg["rerank"] else FINAL_K
    retrieved = coll.query(query_embeddings=qe, n_results=k)
    top_distance = retrieved["distances"][0][0]     # gate always uses the bi-encoder distance

    row = dict(top_distance=top_distance, abstained=False, faithfulness=np.nan,
               answer_relevance=np.nan, context_precision=np.nan, answer="")
    if cfg["gate"] and top_distance > DISTANCE_THRESHOLD:
        row.update(abstained=True, answer="INSUFFICIENT_EVIDENCE (distance gate)")
        return row

    if cfg["rerank"]:
        retrieved = rerank(reranker, q, retrieved, FINAL_K)
    answer = call_gemini(client, build_grounded_prompt(q, retrieved)).strip()
    row["answer"] = answer
    if answer.startswith("INSUFFICIENT_EVIDENCE"):
        row["abstained"] = True                      # caught by the prompt-level guard instead
        return row

    passages = retrieved["documents"][0]
    row["faithfulness"], row["answer_relevance"] = judge_answer(client, q, passages, answer)
    if item["expect"] == "answer":
        row["context_precision"] = judge_context_precision(client, q, passages)
    return row


def summarize(df):
    df = df.copy()
    df["group"] = np.where(df["expect"] == "answer", "on_topic", "off_topic")
    order = [c for c in CONFIGS if c in df["config"].unique()]

    on = df[df.group == "on_topic"].groupby("config")
    t1 = pd.DataFrame({
        "n": on.size(),
        "false_abstain_rate": on["abstained"].mean(),
        "faithfulness": on["faithfulness"].mean(),
        "answer_relevance": on["answer_relevance"].mean(),
        "context_precision": on["context_precision"].mean(),
    }).reindex(order)
    print("\n=== ON-TOPIC queries (means over answered queries) ===")
    print(t1.round(3).to_string())

    off = df[df.group == "off_topic"]
    t2 = pd.DataFrame({
        "n": off.groupby("config").size(),
        "correct_abstain_rate": off.groupby("config")["abstained"].mean(),
        "answered_anyway": off.groupby("config")["abstained"].apply(lambda s: int((~s).sum())),
        "faithfulness_when_answered": off.groupby("config")["faithfulness"].mean(),
    }).reindex(order)
    print("\n=== OFF-TOPIC queries (want: high abstain; if answered anyway, faithfulness shows hallucination) ===")
    print(t2.round(3).to_string())

    t3 = df.groupby(["category", "config"])["context_precision"].mean().unstack("config").reindex(columns=order)
    print("\n=== Context precision by category ===")
    print(t3.dropna(how="all").round(3).to_string())

    d = df[df.config == order[0]].groupby("category")["top_distance"].agg(["min", "median", "max"])
    print("\n=== Top-1 retrieval distance by category (use to re-check the 0.9 threshold) ===")
    print(d.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only run the first N queries per config (smoke test)")
    ap.add_argument("--sleep", type=float, default=1.5, help="seconds between queries (free-tier rate limits)")
    ap.add_argument("--summary", action="store_true", help="just print tables from the existing CSV")
    args = ap.parse_args()

    if args.summary:
        summarize(pd.read_csv(RESULTS_CSV)); return

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Set the GEMINI_API_KEY environment variable first.")
    client = genai.Client(api_key=api_key)

    print("Building pipeline (embedding corpus + loading models)...")
    embed, coll, reranker = build_pipeline()

    done = set()
    rows = []
    if os.path.exists(RESULTS_CSV):
        prev = pd.read_csv(RESULTS_CSV)
        rows = prev.to_dict("records")
        done = set(zip(prev["config"], prev["id"]))

    queries = TEST_QUERIES[:args.limit] if args.limit else TEST_QUERIES
    for cname, cfg in CONFIGS.items():
        for item in queries:
            if (cname, item["id"]) in done:
                continue
            print(f"[{cname}] #{item['id']} ({item['category']}): {item['query'][:70]}")
            res = run_one(cfg, item, embed, coll, reranker, client)
            rows.append({"config": cname, "id": item["id"], "category": item["category"],
                         "expect": item["expect"], "query": item["query"], **res})
            pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)   # save after every row so a crash loses nothing
            time.sleep(args.sleep)

    summarize(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
