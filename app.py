import streamlit as st
from deploy import run_pipeline

st.set_page_config(page_title="Quantum Discovery", layout="centered")

st.title("Quantum Molecular Discovery")
st.caption("Simulator-based pipeline · VQE + iterative learning")

disease_input = st.text_input(
    "Disease / target (demo)",
    value="",
    placeholder="e.g. Hypertension",
    help="For demonstration only; does not affect backend logic yet.",
)

if st.button("Run Quantum Discovery", type="primary"):
    with st.spinner("Running pipeline (generation → VQE scoring → learning)…"):
        final_candidates, explanation, iteration_logs = run_pipeline(include_explanation=True)

    st.success("Pipeline finished.")

    st.subheader("Iteration progress")
    progress_text = "\n".join(iteration_logs)
    st.text(progress_text)

    st.subheader("Final Top 5 Molecules")
    table_data = [
        {"Molecule": mol, "Quantum Score": f"{score:.4f}"}
        for mol, score in final_candidates
    ]
    st.table(table_data)

    if explanation:
        st.subheader("Explanation")
        st.markdown(explanation)
