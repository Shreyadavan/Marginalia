"""Research experiments.

  python evaluate.py leakage data/sample_lecture.md        # E1/E2: knowledge-leakage rate, plain vs critic
  python evaluate.py audit data/sample_lecture.md          # E5: fairness / stereotype / grounding audit
  python evaluate.py correlate data/sample_lecture.md data/real_confusion.csv   # E3: predicted vs real
"""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from marginalia import audit, llm, loader, pipeline


def leakage_experiment(text: str, out="results/leakage.csv"):
    rows = []
    for mode in ["plain", "critic"]:
        a = pipeline.run(text, mode=mode)
        rows.append({"mode": mode, "traces": len(a.df),
                     "first_attempt_leakage": round(a.first_attempt_leakage_rate, 3),
                     "final_leakage": round(a.leakage_rate, 3),
                     "mean_attempts": round(a.df["attempts"].mean(), 2)})
    res = pd.DataFrame(rows)
    Path(out).parent.mkdir(exist_ok=True)
    res.to_csv(out, index=False)
    print(res.to_string(index=False))
    return res


def correlate(text: str, signal_csv: str, column="forum_questions"):
    """Spearman correlation between predicted chunk confusion and a real signal per chunk."""
    a = pipeline.run(text, mode="critic")
    pred = a.df[a.df.persona != "control"].groupby("chunk")["confusion"].mean().rename("predicted")
    real = pd.read_csv(signal_csv).set_index("chunk_id")[column]
    joined = pd.concat([pred, real], axis=1).dropna()
    rho, p = spearmanr(joined["predicted"], joined[column])
    print(joined.round(2).to_string())
    print(f"\nSpearman rho = {rho:.3f} (p = {p:.3f}, n = {len(joined)})")
    return rho, p


def audit_experiment(text: str, out="results"):
    """E5: fairness/realism audit of personas."""
    a = pipeline.run(text, mode="critic")
    rep = audit.audit(a)
    Path(out).mkdir(exist_ok=True)
    print("PERSONA SUMMARY (grounding_rate = share of flagged terms explained by the persona's knowledge state)\n")
    print(rep["summary"].to_string())
    n = len(rep["stereotype_hits"])
    print(f"\nStereotype-lexicon hits: {n}")
    if n:
        print(rep["stereotype_hits"][["persona", "chunk", "terms"]].to_string(index=False))
    print(f"Ungrounded flags needing manual review: {len(rep['ungrounded_flags'])}")
    rep["summary"].to_csv(f"{out}/audit_summary.csv")
    rep["ungrounded_flags"].to_csv(f"{out}/audit_ungrounded.csv", index=False)
    audit.annotation_sample(a).to_csv(f"{out}/audit_annotation_sample.csv", index=False)
    print(f"\nSaved to {out}/ (annotate audit_annotation_sample.csv with human raters)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("leakage"); l.add_argument("file")
    c = sub.add_parser("correlate"); c.add_argument("file"); c.add_argument("signal")
    c.add_argument("--column", default="forum_questions")
    au = sub.add_parser("audit"); au.add_argument("file")
    args = ap.parse_args()
    if llm.MOCK:
        print("** MOCK MODE: numbers are illustrative. Add an API key for real experiments. **\n")
    text = loader.load_path(args.file)
    if args.cmd == "leakage":
        leakage_experiment(text)
    elif args.cmd == "audit":
        audit_experiment(text)
    else:
        correlate(text, args.signal, args.column)


if __name__ == "__main__":
    main()
