# Marginalia: simulated confused readers for course material

LLMs play **students with limited knowledge** who read your lecture notes and think aloud.
The system turns their confusion into a **heatmap** (persona x paragraph), checks whether the
simulated students *cheated* by using knowledge they should not have (**knowledge leakage**),
proposes minimal edits, and re-simulates to see if confusion dropped.

## 1. Setup in VS Code (5 minutes)

1. Install Python 3.10+ and the VS Code **Python** extension.
2. Open this folder in VS Code (`File > Open Folder`).
3. Open a terminal (`Ctrl+` `) and run:

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        Mac/Linux:  source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
```
4. Paste your key into `.env` (`ANTHROPIC_API_KEY=...`). **No key? It still runs in MOCK mode**
   (keyword heuristics instead of an LLM), which is good for checking the app works, not for research results.
5. Press `Ctrl+Shift+P > Python: Select Interpreter` and choose `.venv`.

## 2. Run it

| What | Command (or press F5 and pick the config) |
|------|-------------------------------------------|
| Web app | `streamlit run app.py` |
| CLI heatmap + edit test | `python run_cli.py data/sample_lecture.md --mode critic --edits` |
| E1/E2 leakage experiment | `python evaluate.py leakage data/sample_lecture.md` |
| E5 fairness / stereotype audit | `python evaluate.py audit data/sample_lecture.md` |
| E3 predicted vs real confusion | `python evaluate.py correlate data/sample_lecture.md data/real_confusion.csv` |
| Unit tests | `python -m unittest discover -s tests -v` |

## 3. Project structure

```
marginalia/
  app.py                 Streamlit UI (upload, heatmap, inspector, edit tester)
  run_cli.py             command-line runner
  evaluate.py            research experiments (leakage, correlation)
  data/
    sample_lecture.md    calculus lecture with deliberate jargon/jumps
    glossary.json        offline concept keywords + definitions (mock mode)
    real_confusion.csv   PLACEHOLDER numbers, replace with your real data
  marginalia/
    config.py            weights, thresholds
    llm.py               Anthropic wrapper: JSON, retries, disk cache
    loader.py            read .md/.txt/.pdf/.docx (PDF needs a text layer, not scans)
    chunker.py           split material into chunks (splits long PDF paragraphs)
    audit.py             fairness audit: stereotype lexicon, grounding rate, disparity
    concepts.py          concept extraction + prerequisite graph (networkx)
    personas.py          knowledge-state personas built from graph depth
    schema.py            Trace dataclass (validated think-aloud JSON)
    reader.py            reader agent (+ offline mock reader)
    leakage.py           critic: detects use of forbidden knowledge
    scoring.py           confusion score, failure types, heatmap matrix
    pipeline.py          orchestration + critic retry loop
    editor.py            editor agent + before/after re-simulation
  tests/test_core.py
```

## 4. How it maps to the proposal

| Step | Where |
|------|-------|
| Concept extraction + prerequisite graph | `concepts.py` |
| Persona = knowledge state | `personas.py` (mastered / weak / unknown by graph depth) |
| Think-aloud reading | `reader.py` |
| Leakage detection + critic loop | `leakage.py`, `pipeline.simulate_with_critic` |
| Confusion heatmap, 3 failure types | `scoring.py` |
| Repair + re-simulate | `editor.py` |
| Calibration vs reality | `evaluate.py correlate` |

## 5. Doing the research part

1. **E1/E2 leakage:** run `evaluate.py leakage` on 2-3 courses with a real API key. Compare
   `first_attempt_leakage` (plain prompt) with `final_leakage` (critic loop). Try a different `MARGINALIA_MODEL`.
2. **E3 validation:** replace `data/real_confusion.csv` with real data per chunk (forum questions,
   exam item error rate). Chunk ids must match your chunking (`python run_cli.py` prints them).
   Report Spearman rho. Use `--column exam_error_rate` to switch signal.
3. **E4 edit effectiveness:** use "Propose and test edits" in the app, then show original vs revised
   text to real students or classmates.
4. **E5 fairness audit:** `python evaluate.py audit <file>` (or the audit panel in the app) reports
   stereotype-lexicon hits, the grounding rate per persona and the confusion gap vs the control. It writes
   `results/audit_annotation_sample.csv`; have 2+ people rate it (caricature / plausible gap, 1-5) and report agreement.
   The lexicon and grounding checks are screens, not proof; human ratings are the real evidence. Personas here are defined purely by knowledge state on purpose.

## 6. Known limitations (write these in your report)

- Models leak knowledge despite instructions; that is the finding, measure it honestly.
- Depth-based mastery is a simplification: a concept with no listed prerequisites is treated as foundational.
- Textual leakage detection is heuristic (keyword + hedge words); validate it manually on a sample.
- `data/real_confusion.csv` ships with made-up placeholder numbers so the command runs. Do not report them.
- PDFs: scanned PDFs have no text layer and need OCR first; equations and tables extract poorly.
- Mock-mode output is never a research result.
