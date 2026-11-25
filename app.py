"""Streamlit UI:  streamlit run app.py"""
from pathlib import Path

import plotly.express as px
import streamlit as st

from marginalia import audit, config, editor, llm, loader, pipeline, scoring
from marginalia.personas import TEMPLATES

st.set_page_config(page_title="Marginalia", layout="wide")
st.title("Marginalia")
st.caption("Simulated confused readers show you where real students will get lost, before the semester starts.")

with st.sidebar:
    st.header("Settings")
    has_key = bool(llm.os.getenv("ANTHROPIC_API_KEY"))
    mock = st.toggle("Mock mode (offline heuristics)", value=llm.MOCK or not has_key,
                     disabled=not has_key, help="Needs ANTHROPIC_API_KEY in .env to turn off.")
    llm.set_mock(mock or not has_key)
    mode = st.radio("Leakage control", ["critic", "plain"],
                    help="critic = re-run readers that use knowledge they should not have")
    ids = st.multiselect("Personas", [t["id"] for t in TEMPLATES], default=[t["id"] for t in TEMPLATES])

src = st.radio("Course material", ["Sample lecture", "Upload file", "Paste text"], horizontal=True)
text = ""
if src == "Sample lecture":
    text = Path("data/sample_lecture.md").read_text(encoding="utf-8")
elif src == "Upload file":
    up = st.file_uploader("PDF, Word, Markdown or text file", type=["pdf", "docx", "md", "txt"])
    if up:
        try:
            text = loader.load_bytes(up.read(), up.name)
            st.caption(f"Extracted {len(text.split())} words from {up.name}")
        except Exception as e:
            st.error(str(e))
else:
    text = st.text_area("Paste lecture notes", height=250)

if llm.MOCK:
    st.warning("Mock mode: a keyword heuristic stands in for the LLM. Results are illustrative only.")

if st.button("Analyse", type="primary", disabled=not text.strip()):
    bar = st.progress(0.0, text="Simulating readers...")
    st.session_state.analysis = pipeline.run(text, ids or None, mode, lambda p: bar.progress(p))
    st.session_state.edits = None
    bar.empty()

a = st.session_state.get("analysis")
if a:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chunks", len(a.chunks))
    c2.metric("Personas", len(a.personas))
    c3.metric("Leakage (final)", f"{a.leakage_rate:.0%}")
    c4.metric("Leakage (1st attempt)", f"{a.first_attempt_leakage_rate:.0%}")

    heat = scoring.heat_matrix(a.df)
    fig = px.imshow(heat, color_continuous_scale="YlOrRd", zmin=0, zmax=config.MAX_SCORE,
                    labels=dict(x="Chunk", y="Persona", color="Confusion"), aspect="auto", text_auto=".1f")
    st.subheader("Confusion heatmap")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Inspect a chunk")
    cid = st.selectbox("Chunk", [c.id for c in a.chunks],
                       format_func=lambda i: f"{i}: {a.chunks[i].text.splitlines()[0][:60]}")
    st.markdown(a.chunks[cid].text)
    for t in a.traces:
        if t.chunk_id == cid:
            with st.expander(f"{t.persona_id}: confusion {scoring.confusion_score(t)} "
                             f"{'(LEAKED: ' + ', '.join(t.leaked) + ')' if t.leaked else ''}"):
                st.write(t.think_aloud)
                st.json({k: v for k, v in t.to_dict().items()
                         if k in ("unknown_terms", "unstated_assumptions", "skipped_steps",
                                  "hidden_prereqs", "confidence", "would_stop_reading")})

    with st.expander("Prerequisite graph (concept, depth, requires)"):
        st.dataframe([{"concept": n, "depth": d.get("depth"), "requires": sorted(a.graph.predecessors(n))}
                      for n, d in a.graph.nodes(data=True)])

    with st.expander("Fairness and realism audit (E5)"):
        rep = audit.audit(a)
        st.caption("grounding_rate = share of flagged terms explained by the persona's knowledge state. "
                   "Low values or stereotype hits mean the persona may be caricatured.")
        st.dataframe(rep["summary"])
        st.write(f"Stereotype-lexicon hits: **{len(rep['stereotype_hits'])}** | "
                 f"ungrounded flags to review: **{len(rep['ungrounded_flags'])}**")
        if len(rep["stereotype_hits"]):
            st.dataframe(rep["stereotype_hits"])
        st.download_button("Download annotation sample (CSV)",
                           audit.annotation_sample(a).to_csv(index=False), "audit_annotation_sample.csv")

    st.subheader("Suggested fixes")
    if st.button("Propose and test edits for the hottest chunks"):
        with st.spinner("Editing and re-simulating..."):
            st.session_state.edits = editor.improve(a, top_n=3)
    for r in st.session_state.get("edits") or []:
        st.markdown(f"**Chunk {r['chunk']}**: {r['edit_type']} | confusion **{r['before']} -> {r['after']}** "
                    f"(gain {r['gain']})  \n{r['rationale']}")
        left, right = st.columns(2)
        left.text_area("Original", r["original_text"], height=200, key=f"o{r['chunk']}")
        right.text_area("Revised", r["revised_text"], height=200, key=f"r{r['chunk']}")
    if st.session_state.get("edits") == []:
        st.info("No chunk exceeded the confusion threshold, nothing to fix.")
