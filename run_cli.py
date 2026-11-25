"""Command-line runner:  python run_cli.py data/sample_lecture.md --mode critic --edits"""
import argparse
from pathlib import Path

from marginalia import editor, llm, loader, pipeline, scoring


def main():
    ap = argparse.ArgumentParser(description="Marginalia: simulate confused readers.")
    ap.add_argument("file", help="Markdown/text course material")
    ap.add_argument("--mode", choices=["plain", "critic"], default="critic")
    ap.add_argument("--edits", action="store_true", help="propose + test fixes for hot chunks")
    ap.add_argument("--out", default="results", help="output folder")
    args = ap.parse_args()

    if llm.MOCK:
        print("** MOCK MODE (no ANTHROPIC_API_KEY): heuristic reader, results are illustrative only **\n")

    text = loader.load_path(args.file)
    a = pipeline.run(text, mode=args.mode, progress=lambda p: print(f"\rsimulating {p:4.0%}", end=""))
    print()

    heat = scoring.heat_matrix(a.df)
    print("CONFUSION HEATMAP (rows=persona, cols=chunk, 0-10)\n")
    print(heat.round(1).to_string(), "\n")
    print(f"Leakage rate (after critic): {a.leakage_rate:.1%} | first attempt: {a.first_attempt_leakage_rate:.1%}\n")
    print("Mean confusion per chunk:")
    for cid, s in a.chunk_scores().items():
        title = a.chunks[cid].text.splitlines()[0][:50]
        print(f"  chunk {cid} [{'#' * int(round(s))}{'.' * (10 - int(round(s)))}] {s:4.1f}  {title}")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    heat.to_csv(out / "heatmap.csv")
    a.df.to_csv(out / "traces_summary.csv", index=False)

    if args.edits:
        print("\nSUGGESTED EDITS")
        for r in editor.improve(a):
            print(f"  chunk {r['chunk']}: {r['edit_type']} | confusion {r['before']} -> {r['after']} "
                  f"(gain {r['gain']})\n    {r['rationale']}")
    print(f"\nSaved CSVs to {out}/")


if __name__ == "__main__":
    main()
