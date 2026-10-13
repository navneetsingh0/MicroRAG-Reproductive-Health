"""
eval_ablation.py - standalone evaluation + ablation for MicroRAG (does NOT import app.py, which runs Streamlit code).

Configs compared on the same query set:
  baseline : original system  (top-5 from ChromaDB, distance gate ON)
  rerank   : + cross-encoder  (top-15 -> re-ranked top-5, distance gate ON)
  no_gate  : ablation         (top-5, distance gate OFF - always calls Gemini; only the prompt-level guard remains;
                               run on the should-refuse queries only: for answerable queries it is identical to baseline)
Judge: Groq Llama (independent of the Gemini generator) when GROQ_API_KEY is set, else Gemini. One judge call per answer.

Usage (run from the repo root, where bv_literature_corpus.csv lives):
  export GEMINI_API_KEY="..."            # Windows PowerShell:  $env:GEMINI_API_KEY="..."
  python eval_ablation.py --limit 3      # smoke test on 3 queries per config
  python eval_ablation.py                # full run; safe to stop and re-run - finished rows are skipped
  python eval_ablation.py --summary      # just re-print the tables from eval_results.csv

NOTE: build_pipeline(), retrieval and build_grounded_prompt() mirror app.py. If you change them there, change them here.
"""
import os, re, sys, json, time, requests, random, argparse
import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from google import genai
from google.genai import types

from eval_queries import TEST_QUERIES

DISTANCE_THRESHOLD = 0.9
RETRIEVE_K, FINAL_K = 15, 5
MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")      # generator (same model as the app)
JUDGE = "gemini"                                                   # set in main(): "groq" (independent judge) or "gemini"
GENERATOR = "gemini"                                     # set in main(): "gemini" (as in the app) or "groq"
GROQ_GEN_MODEL = ""
GROQ_MODEL = os.environ.get("GROQ_JUDGE_MODEL", "")      # empty = auto-pick from the models your key can use
GROQ_PREFERRED = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
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
def call_gemini(client, prompt, json_mode=False, retries=9):
    """retries=9 with a 90s cap gives ~9 minutes of total patience for free-tier 503 bursts,
    which are common on the -latest alias models and usually clear within a few minutes."""
    use_json_config = json_mode
    for attempt in range(retries):
        try:
            cfg = (types.GenerateContentConfig(temperature=0, response_mime_type="application/json")
                   if use_json_config else None)
            return client.models.generate_content(model=MODEL, contents=prompt, config=cfg).text
        except Exception as e:  # rate limits / transient errors on the free tier
            code = getattr(e, "code", "?")
            msg = str(e).replace("\n", " ")[:200]
            if code == 429 and ("PerDay" in str(e) or "per day" in str(e).lower()):
                raise RuntimeError(
                    f"Gemini DAILY free-tier quota for {MODEL} is used up (it resets in many hours; retrying cannot help). "
                    "Options: (a) set GEMINI_MODEL=gemini-flash-lite-latest (separate quota), "
                    "(b) use a Gemini key from a NEW Google AI Studio project, or (c) run with --generator groq. "
                    "Progress is saved; re-run the same command afterwards.")
            wait = min(5 * 2 ** attempt, 90) + random.uniform(0, 3)   # jitter avoids retrying in lockstep
            print(f"    API error {type(e).__name__} code={code}: {msg} | retrying in {wait:.0f}s (attempt {attempt+1}/{retries})")
            if use_json_config and attempt >= 1:
                use_json_config = False   # JSON-mode kept failing: ask for JSON in the prompt instead (parse_json handles it)
                print("    -> switching to plain-text mode for this call")
            time.sleep(wait)
    raise RuntimeError(f"Gemini ({MODEL}) kept failing after {retries} attempts (~{sum(min(5*2**i,90) for i in range(retries))//60} min). "
                        "Progress is saved - re-run the same command later to resume, or try a non-preview model, "
                        "e.g. GEMINI_MODEL=gemini-2.5-flash python eval_ablation.py")


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)     # tolerate code fences / extra prose around the JSON
    return json.loads(m.group(0))


def ping(client):
    """Diagnostic: which kinds of Gemini call work right now?"""
    tests = {
        "plain": None,
        "temperature=0": types.GenerateContentConfig(temperature=0),
        "json mode": types.GenerateContentConfig(temperature=0, response_mime_type="application/json"),
    }
    print(f"Model: {MODEL}")
    for name, cfg in tests.items():
        try:
            prompt = 'Reply with JSON {"ok": true}' if name == "json mode" else "Reply with the single word: ok"
            out = client.models.generate_content(model=MODEL, contents=prompt, config=cfg).text
            print(f"  {name:14s} OK   -> {out.strip()[:60]!r}")
        except Exception as e:
            print(f"  {name:14s} FAIL -> {type(e).__name__} code={getattr(e, 'code', '?')}: {str(e)[:160]}")


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


# ---------------- judge (one call per answered query) ----------------
def groq_available():
    """Ask Groq which models this key can actually use and take the best judge from a preference list."""
    r = requests.get("https://api.groq.com/openai/v1/models", timeout=30,
                     headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"})
    if r.status_code != 200:
        raise RuntimeError(f"Could not list Groq models ({r.status_code}): {r.text[:150]}. Check GROQ_API_KEY.")
    return {m["id"] for m in r.json().get("data", [])}


def pick_groq_model(available):
    for m in GROQ_PREFERRED:
        if m in available:
            return m
    raise RuntimeError("None of the preferred judge models are available to this Groq key. Available: "
                       + ", ".join(sorted(available)[:15]) + " - set GROQ_JUDGE_MODEL to one of them, or use --judge gemini.")


def call_groq(prompt, retries=8, model=None, json_mode=True):
    key = os.environ["GROQ_API_KEY"]
    use_json_format = json_mode
    for attempt in range(retries):
        try:
            body = {"model": model or GROQ_MODEL, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
            if use_json_format:
                body["response_format"] = {"type": "json_object"}
            r = requests.post(GROQ_URL, timeout=120, headers={"Authorization": f"Bearer {key}"}, json=body)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            if r.status_code == 400 and use_json_format:       # some models reject json_object: retry asking for JSON in the prompt
                use_json_format = False
                print("    Groq 400 with JSON mode; retrying without it")
                continue
            if r.status_code in (400, 401, 403, 404):          # not retryable: bad key / model name / request
                raise RuntimeError(f"Groq rejected the request ({r.status_code}): {r.text[:200]}. "
                                   "Check GROQ_API_KEY and GROQ_JUDGE_MODEL (list: console.groq.com/docs/models)")
            ra = r.headers.get("retry-after")
            raise requests.HTTPError(f"{r.status_code}: {r.text[:150]}", response=r) if not ra else \
                  requests.HTTPError(f"{r.status_code} (retry-after {ra}s)", response=r)
        except requests.RequestException as e:
            ra = getattr(getattr(e, "response", None), "headers", {}).get("retry-after") if getattr(e, "response", None) is not None else None
            wait = (float(ra) + 1 if ra else min(5 * 2 ** attempt, 60)) + random.uniform(0, 2)
            print(f"    Groq error: {str(e)[:160]} | retrying in {wait:.0f}s (attempt {attempt+1}/{retries})")
            time.sleep(wait)
    raise RuntimeError("Groq kept failing. Progress is saved - re-run the same command to resume.")


def judge_all(client, question, passages, answer, want_precision):
    """Single judge call returning faithfulness, answer relevance and (optionally) per-passage relevance."""
    ctx = "\n\n".join(f"[Source {i+1}] {p[:1500]}" for i, p in enumerate(passages))
    prompt = f"""You are a strict evaluator of a retrieval-augmented answer.

QUESTION:
{question}

SOURCES:
{ctx}

ANSWER:
{answer}

Score:
- faithfulness (0.0-1.0): fraction of the factual claims in the ANSWER directly supported by the SOURCES (1.0 = no unsupported claims).
- answer_relevance (0.0-1.0): how directly and completely the ANSWER addresses the QUESTION.
- relevant: for EACH source in order, true if it contains information that helps answer the QUESTION, else false.
Return ONLY JSON: {{"faithfulness": <float>, "answer_relevance": <float>, "relevant": [<true|false> per source]}}"""
    text = call_groq(prompt) if JUDGE == "groq" else call_gemini(client, prompt, json_mode=True)
    out = parse_json(text)
    cp = np.nan
    if want_precision:
        flags = [bool(x) for x in out.get("relevant", [])][:len(passages)]
        cp = float(np.mean(flags)) if flags else np.nan
    return float(out["faithfulness"]), float(out["answer_relevance"]), cp


def generate(client, prompt):
    if GENERATOR == "groq":
        return call_groq(prompt, model=GROQ_GEN_MODEL, json_mode=False)
    return call_gemini(client, prompt)


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
    answer = generate(client, build_grounded_prompt(q, retrieved)).strip()
    row["answer"] = answer
    if answer.startswith("INSUFFICIENT_EVIDENCE"):
        row["abstained"] = True                      # caught by the prompt-level guard instead
        return row

    passages = retrieved["documents"][0]
    row["faithfulness"], row["answer_relevance"], row["context_precision"] = judge_all(
        client, q, passages, answer, want_precision=(item["expect"] == "answer"))
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
    ap.add_argument("--ping", action="store_true", help="test which Gemini call types work, then exit")
    ap.add_argument("--smoke", action="store_true", help="tiny mixed test: 3 on-topic + 1 profile + 2 should-refuse queries")
    ap.add_argument("--generator", choices=["gemini", "groq"], default="gemini",
                    help="model that writes the answers (default gemini, as in the deployed app)")
    ap.add_argument("--judge", choices=["groq", "gemini"], help="judge model provider (default: groq if GROQ_API_KEY is set)")
    args = ap.parse_args()

    if args.summary:
        summarize(pd.read_csv(RESULTS_CSV)); return

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Set the GEMINI_API_KEY environment variable first.")
    client = genai.Client(api_key=api_key)
    if args.ping:
        ping(client); return

    global JUDGE
    JUDGE = args.judge or ("groq" if os.environ.get("GROQ_API_KEY") else "gemini")
    if JUDGE == "groq" and not os.environ.get("GROQ_API_KEY"):
        sys.exit("--judge groq needs the GROQ_API_KEY environment variable.")
    global GROQ_MODEL, GENERATOR, GROQ_GEN_MODEL
    GENERATOR = args.generator
    if GENERATOR == "groq" and JUDGE != "groq":
        sys.exit("--generator groq needs GROQ_API_KEY (Groq is then also the judge).")
    if JUDGE == "groq":
        avail = groq_available()
        if not GROQ_MODEL:
            GROQ_MODEL = pick_groq_model(avail)
        if GENERATOR == "groq":   # keep generator and judge different models when possible
            GROQ_GEN_MODEL = os.environ.get("GROQ_GEN_MODEL") or (
                "openai/gpt-oss-20b" if "openai/gpt-oss-20b" in avail and GROQ_MODEL != "openai/gpt-oss-20b" else GROQ_MODEL)
    gen_name = GROQ_GEN_MODEL if GENERATOR == "groq" else MODEL
    print(f"Generator: {GENERATOR} ({gen_name}) | Judge: {JUDGE}" + (f" ({GROQ_MODEL})" if JUDGE == "groq" else f" ({MODEL}, same model as generator)"))
    if GENERATOR == "groq":
        print("NOTE: answers are generated by a Groq model, not the Gemini model used in the deployed app - state this in the report.")

    print("Building pipeline (embedding corpus + loading models)...")
    embed, coll, reranker = build_pipeline()

    done = set()
    rows = []
    if os.path.exists(RESULTS_CSV):
        prev = pd.read_csv(RESULTS_CSV)
        rows = prev.to_dict("records")
        done = set(zip(prev["config"], prev["id"]))

    queries = TEST_QUERIES[:args.limit] if args.limit else TEST_QUERIES
    if args.smoke:
        pick = [t for t in TEST_QUERIES if t["category"] == "symptoms"][:2] \
             + [t for t in TEST_QUERIES if t["category"] == "uncertain_evidence"][:1] \
             + [t for t in TEST_QUERIES if t["category"] == "profile"][:1] \
             + [t for t in TEST_QUERIES if t["expect"] == "abstain"][:1] \
             + [t for t in TEST_QUERIES if t["category"] == "near_miss"][:1]
        queries = pick
    for cname, cfg in CONFIGS.items():
        for item in queries:
            if (cname, item["id"]) in done:
                continue
            # Ablation economy: with the gate OFF, an answerable query that passes the gate anyway is identical to baseline,
            # so only the queries the gate would have blocked are re-run here.
            if cname == "no_gate" and item["expect"] == "answer":
                continue
            print(f"[{cname}] #{item['id']} ({item['category']}): {item['query'][:70]}")
            try:
                res = run_one(cfg, item, embed, coll, reranker, client)
            except RuntimeError as e:
                print(f"\nSTOPPED: {e}")
                if rows: summarize(pd.DataFrame(rows))
                sys.exit(1)
            rows.append({"config": cname, "id": item["id"], "category": item["category"],
                         "expect": item["expect"], "query": item["query"], **res})
            pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)   # save after every row so a crash loses nothing
            time.sleep(args.sleep)

    summarize(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
