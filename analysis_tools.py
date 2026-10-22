"""
analysis_tools.py - small helpers that work on the CSVs written by eval_ablation.py (no API keys, no models needed).

  python analysis_tools.py compare eval_results.csv eval_results_t0.csv
        How much do the numbers move between two runs? (the noise floor for every comparison in the report)

  python analysis_tools.py inspect eval_results_t0.csv
        For every in-scope question that was declined: the retrieved titles, and what the model said.
        Needs a results file produced by the CURRENT eval_ablation.py (it stores the retrieved passages).

  python analysis_tools.py sheet eval_results_t0.csv --n 12
        Writes validation_sheet.csv (question, answer, the 5 sources, EMPTY rating columns) and validation_key.csv (judge scores).
        You fill in human_faithfulness and human_relevance in the sheet (0, 0.5 or 1), WITHOUT looking at the key.

  python analysis_tools.py score
        Compares your ratings with the judge's: mean absolute difference, correlation, agreement.
"""
import sys, argparse
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 120)


def inscope(df, config):
    return df[(df["config"] == config) & (df["expect"] == "answer") & (df["category"] != "profile_v2")].set_index("id")


def compare(a_path, b_path):
    a, b = pd.read_csv(a_path), pd.read_csv(b_path)
    print(f"Run A: {a_path}   Run B: {b_path}\n")
    for cfg in ("baseline", "rerank"):
        A, B = inscope(a, cfg), inscope(b, cfg)
        common = A.index.intersection(B.index)
        A, B = A.loc[common], B.loc[common]
        print(f"=== {cfg}: {len(common)} in-scope queries in both runs ===")
        rows = []
        for m in ("faithfulness", "answer_relevance", "context_precision"):
            rows.append([m, A[m].mean(), B[m].mean(), B[m].mean() - A[m].mean()])
        rows.append(["declined (rate)", A["abstained"].mean(), B["abstained"].mean(), B["abstained"].mean() - A["abstained"].mean()])
        print(pd.DataFrame(rows, columns=["metric", "run A", "run B", "B - A"]).round(3).to_string(index=False))
        da, db = set(A.index[A["abstained"]]), set(B.index[B["abstained"]])
        print(f"declined in A only: {sorted(da - db)} | in B only: {sorted(db - da)} | in both: {sorted(da & db)}\n")

    print("=== re-ranker effect inside each run (paired, queries answered in both configs) ===")
    rng = np.random.default_rng(0)
    for name, df in (("A", a), ("B", b)):
        base, rr = inscope(df, "baseline"), inscope(df, "rerank")
        both = base.index[(~base["abstained"]) & (~rr.loc[base.index, "abstained"])]
        for m in ("context_precision", "faithfulness"):
            diff = (rr.loc[both, m] - base.loc[both, m]).values
            boots = [rng.choice(diff, len(diff)).mean() for _ in range(4000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print(f"run {name}  {m:18s} n={len(both):2d}  mean diff {diff.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    print("\nIf the differences between runs (B - A) are as large as the re-ranker effect, the effect is within noise.")


def inspect(path):
    df = pd.read_csv(path)
    if "retrieved_titles" not in df.columns:
        sys.exit("This results file has no retrieved passages. Re-run with the current eval_ablation.py (it stores them).")
    d = df[(df["config"] == "baseline") & (df["expect"] == "answer") & (df["abstained"])]
    print(f"{len(d)} declined in-scope questions in the baseline run\n")
    for _, r in d.iterrows():
        how = "DISTANCE GATE" if r["top_distance"] > 0.9 else "MODEL said insufficient evidence"
        print("=" * 100)
        print(f"#{r['id']} [{r['category']}] best distance {r['top_distance']:.3f}  ->  {how}")
        print("Q:", str(r["query"])[:300])
        print("Retrieved:")
        for t in str(r["retrieved_titles"]).split(" || "):
            print("   -", t)
        print("Model said:", str(r["answer"])[:250])
    print("\nFor each one decide: (a) the corpus really does not cover it, (b) the right abstract was not retrieved, or (c) the sources were relevant and the model was too strict.")


def sheet(path, n, seed):
    df = pd.read_csv(path)
    if "retrieved_text" not in df.columns:
        sys.exit("This results file has no retrieved passages. Re-run with the current eval_ablation.py first.")
    d = df[(df["config"] == "baseline") & (df["expect"] == "answer") & (~df["abstained"]) & (df["category"] != "profile_v2")]
    cats = d["category"].unique()
    per = max(1, int(np.ceil(n / len(cats))))
    pick = pd.concat([g.sample(min(per, len(g)), random_state=seed) for _, g in d.groupby("category")])
    pick = pick.sample(min(n, len(pick)), random_state=seed).sort_values("id")
    sheet_rows, key_rows = [], []
    for _, r in pick.iterrows():
        sources = str(r["retrieved_text"]).split(" ### ")
        row = {"id": r["id"], "question": r["query"], "answer": r["answer"]}
        for i in range(5):
            row[f"source_{i+1}"] = sources[i] if i < len(sources) else ""
        row.update({"human_faithfulness": "", "human_relevance": "", "notes": ""})
        sheet_rows.append(row)
        key_rows.append({"id": r["id"], "judge_faithfulness": r["faithfulness"], "judge_relevance": r["answer_relevance"]})
    pd.DataFrame(sheet_rows).to_csv("validation_sheet.csv", index=False)
    pd.DataFrame(key_rows).to_csv("validation_key.csv", index=False)
    print(f"Wrote validation_sheet.csv ({len(pick)} answers) and validation_key.csv.")
    print("Rate each answer WITHOUT opening the key.  Scale for both columns: 1 = fully, 0.5 = partly, 0 = not at all.")
    print("  human_faithfulness: are the claims in the answer supported by the sources shown?")
    print("  human_relevance:    does the answer actually address the question?")


def score():
    s, k = pd.read_csv("validation_sheet.csv"), pd.read_csv("validation_key.csv")
    m = s.merge(k, on="id")
    m = m.dropna(subset=["human_faithfulness", "human_relevance"])
    if m.empty:
        sys.exit("Fill in human_faithfulness and human_relevance in validation_sheet.csv first.")
    print(f"{len(m)} answers rated by hand\n")
    for h, j, name in (("human_faithfulness", "judge_faithfulness", "Faithfulness"), ("human_relevance", "judge_relevance", "Answer relevance")):
        diff = (m[j] - m[h]).abs()
        corr = m[h].corr(m[j], method="spearman") if m[h].nunique() > 1 and m[j].nunique() > 1 else float("nan")
        print(f"{name}: mean human {m[h].mean():.2f} | mean judge {m[j].mean():.2f} | mean absolute difference {diff.mean():.2f} "
              f"| within 0.25: {(diff <= 0.25).mean():.0%} | Spearman {corr:.2f}")
    print("\nLargest disagreements:")
    m["gap"] = (m["judge_faithfulness"] - m["human_faithfulness"]).abs() + (m["judge_relevance"] - m["human_relevance"]).abs()
    print(m.sort_values("gap", ascending=False)[["id", "human_faithfulness", "judge_faithfulness", "human_relevance", "judge_relevance"]].head(5).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare"); c.add_argument("run_a"); c.add_argument("run_b")
    i = sub.add_parser("inspect"); i.add_argument("results")
    h = sub.add_parser("sheet"); h.add_argument("results"); h.add_argument("--n", type=int, default=12); h.add_argument("--seed", type=int, default=1)
    sub.add_parser("score")
    a = ap.parse_args()
    if a.cmd == "compare": compare(a.run_a, a.run_b)
    elif a.cmd == "inspect": inspect(a.results)
    elif a.cmd == "sheet": sheet(a.results, a.n, a.seed)
    else: score()
