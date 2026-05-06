"""Streamlit UI for the Quantum-AI Drug Discovery pipeline.

Run with::

    streamlit run app.py

Workflow:
1.  Pretrain MolGAN once (``python scripts/train.py``) so a checkpoint
    exists at ``checkpoints/molgan.pt``.
2.  Launch this UI to:
      - generate candidate molecules from the trained generator,
      - score them with VQE on PennyLane Lightning,
      - inspect 2D structures, scaffolds, and functional groups,
      - cross-validate the best one on Qiskit Aer,
      - optionally have Gemini explain the result in plain English.

The heavy MolGAN load is cached via ``@st.cache_resource`` so re-running
inference does not reload the network on every interaction.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from src.config import LLM_CONFIG, QUANTUM_CONFIG, SIMULATION_CONFIG  # noqa: E402
from src.data_loader import discover_dataset, load_zinc_subset  # noqa: E402
from src.inference import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    InferenceResult,
    load_pretrained_molgan,
    run_inference,
    score_candidates,
    smiles_to_image_bytes,
    validate_on_hardware,
)
from src.pipeline.feature_extractor import (  # noqa: E402
    extract_functional_groups,
    extract_scaffolds,
)
from src.quantum.vqe import VQEResult, VQEScorer  # noqa: E402

st.set_page_config(
    page_title="Quantum-AI Drug Discovery",
    page_icon="\U0001f9ea",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading MolGAN checkpoint...")
def get_molgan(checkpoint_path: str):
    return load_pretrained_molgan(Path(checkpoint_path))


@st.cache_resource(show_spinner="Initializing VQE scorer...")
def get_scorer(
    active_electrons: int,
    active_orbitals: int,
    vqe_steps: int,
    vqe_stepsize: float,
    max_heavy_atoms: int,
    hamiltonian_timeout_sec: float,
    optimization_timeout_sec: float,
):
    return VQEScorer(
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        vqe_steps=vqe_steps,
        vqe_stepsize=vqe_stepsize,
        max_heavy_atoms=max_heavy_atoms,
        hamiltonian_timeout_sec=hamiltonian_timeout_sec,
        optimization_timeout_sec=optimization_timeout_sec,
    )


@st.cache_data(show_spinner=False)
def get_dataset_preview(n: int):
    return load_zinc_subset(n=n)


@st.cache_data(show_spinner=False)
def render_smiles(smiles: str, size: int = 280) -> Optional[bytes]:
    return smiles_to_image_bytes(smiles, size=size)


# ---------------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------------

QUICK_DEMO_PRESET = {
    "cfg_n_molecules": 8,
    "cfg_iterations": 1,
    "cfg_top_k": 5,
    "cfg_active_electrons": 2,
    "cfg_active_orbitals": 2,
    "cfg_vqe_steps": 10,
    "cfg_vqe_stepsize": 0.1,
    "cfg_max_heavy_atoms": 4,
    "cfg_hamiltonian_timeout": 15,
    "cfg_optimization_timeout": 15,
    "cfg_enable_hardware": False,
    "cfg_enable_explanation": False,
    "cfg_seed_from_pool": True,
}


def _apply_preset(preset: dict) -> None:
    for k, v in preset.items():
        st.session_state[k] = v


def sidebar_config() -> dict:
    st.sidebar.title("Configuration")

    if st.sidebar.button(
        "Quick demo (~2 min)",
        type="primary",
        use_container_width=True,
        help=(
            "Sets all knobs for a fast review run: 8 small candidates, "
            "15 VQE steps, 7 heavy-atom budget. Tops up from a curated "
            "fallback pool if MolGAN can't produce enough small molecules."
        ),
    ):
        _apply_preset(QUICK_DEMO_PRESET)
        st.rerun()

    checkpoint_path = st.sidebar.text_input(
        "MolGAN checkpoint",
        value=str(DEFAULT_CHECKPOINT),
        help="Path to a checkpoint produced by `scripts/train.py`.",
    )
    checkpoint_exists = Path(checkpoint_path).exists()
    if checkpoint_exists:
        st.sidebar.success("Checkpoint found.")
    else:
        st.sidebar.warning(
            "Checkpoint not found. Run `python scripts/train.py` first; "
            "the UI will use an untrained generator until then."
        )

    st.sidebar.subheader("Generation")
    n_molecules = st.sidebar.slider(
        "Molecules per iteration", 5, 50,
        SIMULATION_CONFIG.get("molecules_per_iter", 20),
        key="cfg_n_molecules",
    )
    iterations = st.sidebar.slider(
        "Feedback iterations", 1, 5, 1,
        help="1 = pure inference. >1 enables fine-tuning between iterations.",
        key="cfg_iterations",
    )
    top_k = st.sidebar.slider(
        "Top-K to keep", 1, 10,
        SIMULATION_CONFIG.get("final_candidates_count", 5),
        key="cfg_top_k",
    )

    st.sidebar.subheader("VQE")
    active_electrons = st.sidebar.number_input(
        "Active electrons", 2, 8, QUANTUM_CONFIG.get("active_electrons", 2), step=2,
        key="cfg_active_electrons",
    )
    active_orbitals = st.sidebar.number_input(
        "Active orbitals", 2, 8, QUANTUM_CONFIG.get("active_orbitals", 2), step=1,
        key="cfg_active_orbitals",
    )
    vqe_steps = st.sidebar.slider(
        "VQE steps", 5, 200, QUANTUM_CONFIG.get("vqe_steps", 50),
        key="cfg_vqe_steps",
    )
    vqe_stepsize = st.sidebar.number_input(
        "VQE step size", 0.001, 1.0,
        float(QUANTUM_CONFIG.get("vqe_stepsize", 0.1)),
        step=0.01, key="cfg_vqe_stepsize",
    )

    st.sidebar.subheader("Safety guards")
    st.sidebar.caption(
        "VQE runs full Hartree-Fock SCF on the entire molecule. Larger "
        "molecules can take many minutes per scoring; these guards keep "
        "the run bounded."
    )
    max_heavy_atoms = st.sidebar.slider(
        "Max heavy atoms (skip if larger)",
        min_value=3, max_value=12,
        value=int(QUANTUM_CONFIG.get("max_heavy_atoms_for_vqe", 6)),
        key="cfg_max_heavy_atoms",
    )
    hamiltonian_timeout = st.sidebar.slider(
        "Hamiltonian timeout (s)", 10, 300,
        int(QUANTUM_CONFIG.get("hamiltonian_timeout_sec", 60)),
        key="cfg_hamiltonian_timeout",
    )
    optimization_timeout = st.sidebar.slider(
        "Optimization timeout (s)", 10, 300,
        int(QUANTUM_CONFIG.get("optimization_timeout_sec", 60)),
        key="cfg_optimization_timeout",
    )

    st.sidebar.subheader("Validation & explanation")
    enable_hardware = st.sidebar.checkbox(
        "Validate best on Qiskit Aer", value=True,
        key="cfg_enable_hardware",
    )
    enable_explanation = st.sidebar.checkbox(
        "Generate Gemini explanation",
        value=False,
        help="Requires GOOGLE_API_KEY in .env",
        key="cfg_enable_explanation",
    )
    if enable_explanation and not os.getenv("GOOGLE_API_KEY"):
        st.sidebar.error("GOOGLE_API_KEY not set in .env; explanation will fail.")

    return dict(
        checkpoint_path=checkpoint_path,
        checkpoint_exists=checkpoint_exists,
        n_molecules=n_molecules,
        iterations=iterations,
        top_k=top_k,
        active_electrons=int(active_electrons),
        active_orbitals=int(active_orbitals),
        vqe_steps=int(vqe_steps),
        vqe_stepsize=float(vqe_stepsize),
        max_heavy_atoms=int(max_heavy_atoms),
        hamiltonian_timeout_sec=float(hamiltonian_timeout),
        optimization_timeout_sec=float(optimization_timeout),
        enable_hardware=enable_hardware,
        enable_explanation=enable_explanation,
    )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def render_molecule_card(result: VQEResult, idx: int) -> None:
    img_bytes = render_smiles(result.smiles)
    with st.container(border=True):
        cols = st.columns([1, 2])
        with cols[0]:
            if img_bytes is not None:
                st.image(img_bytes, caption=result.smiles, use_container_width=True)
            else:
                st.code(result.smiles)
        with cols[1]:
            st.markdown(f"**Rank #{idx}**")
            st.metric("VQE energy (Ha)", f"{result.energy:+.4f}")
            st.caption(
                f"qubits: {result.n_qubits} \u00b7 steps: {result.steps_run} "
                f"\u00b7 status: `{result.status.value}`"
            )


def render_top_candidates(top: List[VQEResult]) -> None:
    if not top:
        st.warning("No successful VQE results to display.")
        return
    st.subheader("Top candidates")
    cols = st.columns(min(len(top), 3))
    for i, result in enumerate(top, start=1):
        with cols[(i - 1) % len(cols)]:
            render_molecule_card(result, idx=i)


def render_chemistry_summary(scaffolds: List[str], groups: List[str]) -> None:
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Common scaffolds")
        if scaffolds:
            for scaf in scaffolds:
                img = render_smiles(scaf, size=220)
                with st.container(border=True):
                    if img is not None:
                        st.image(img, caption=scaf, use_container_width=True)
                    else:
                        st.code(scaf)
        else:
            st.caption("No scaffolds extracted.")
    with cols[1]:
        st.subheader("Common functional groups")
        if groups:
            for g in groups:
                st.markdown(f"- `{g}`")
        else:
            st.caption("No functional groups detected.")


def render_hardware_validation(result_obj: InferenceResult) -> None:
    hv = result_obj.hardware_validation
    if hv is None:
        return
    st.subheader("Qiskit Aer validation")
    if hv.succeeded:
        delta = None
        if result_obj.top_candidates:
            delta = hv.energy - result_obj.top_candidates[0].energy
        cols = st.columns(3)
        cols[0].metric("Molecule", hv.smiles)
        cols[1].metric("Aer energy (Ha)", f"{hv.energy:+.4f}",
                       delta=f"{delta:+.4f}" if delta is not None else None,
                       delta_color="inverse")
        cols[2].metric("Qubits", hv.n_qubits)
    else:
        st.error(f"Validation failed: {hv.error}")


def render_results_table(results: List[VQEResult]) -> None:
    if not results:
        return
    rows = [
        {
            "smiles": r.smiles,
            "energy_Ha": r.energy if r.succeeded else None,
            "qubits": r.n_qubits,
            "status": r.status.value,
            "error": r.error or "",
        }
        for r in results
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def tab_generate_and_score(cfg: dict) -> None:
    st.header("Generate candidates and score with VQE")
    st.caption(
        "Sample novel SMILES from the trained MolGAN, then evaluate each "
        "with a Variational Quantum Eigensolver to estimate its ground-state "
        "energy.  Lower energies are preferred."
    )

    seed_from_pool = st.checkbox(
        "Seed half the candidates from the curated drug-fragment pool",
        value=st.session_state.get("cfg_seed_from_pool", False),
        key="cfg_seed_from_pool",
        help=(
            "PySCF doesn't build cleanly on Windows + Python 3.11, so the "
            "live VQE path falls back to PennyLane's slow pure-Python HF. "
            "Seeding from the curated pool guarantees instant cache hits "
            "on chemistry-realistic molecules so the pipeline finishes "
            "well under 2 minutes."
        ),
    )

    if st.button("Run inference", type="primary", use_container_width=True):
        progress_bar = st.progress(0.0, text="Starting...")
        status_box = st.empty()

        def _progress(msg: str, frac: float) -> None:
            progress_bar.progress(min(1.0, frac), text=msg)
            status_box.caption(msg)

        scorer_kwargs = dict(
            active_electrons=cfg["active_electrons"],
            active_orbitals=cfg["active_orbitals"],
            vqe_steps=cfg["vqe_steps"],
            vqe_stepsize=cfg["vqe_stepsize"],
            max_heavy_atoms=cfg["max_heavy_atoms"],
            hamiltonian_timeout_sec=cfg["hamiltonian_timeout_sec"],
            optimization_timeout_sec=cfg["optimization_timeout_sec"],
        )

        fallback_first = max(1, cfg["n_molecules"] // 2) if seed_from_pool else 0

        result = run_inference(
            checkpoint_path=cfg["checkpoint_path"],
            n_molecules=cfg["n_molecules"],
            iterations=cfg["iterations"],
            top_k=cfg["top_k"],
            enable_hardware_validation=cfg["enable_hardware"],
            enable_explanation=cfg["enable_explanation"],
            scorer_kwargs=scorer_kwargs,
            progress=_progress,
            fallback_pool_first=fallback_first,
        )

        st.session_state["last_result"] = result
        progress_bar.progress(1.0, text=f"Done in {result.elapsed_sec:.1f}s")

    result: Optional[InferenceResult] = st.session_state.get("last_result")
    if result is None:
        st.info("Click **Run inference** to generate and score molecules.")
        return

    cols = st.columns(4)
    cols[0].metric("Total scored", len(result.candidates))
    cols[1].metric(
        "Successful",
        sum(1 for r in result.candidates if r.succeeded),
    )
    cols[2].metric("Iterations", result.iterations_run)
    cols[3].metric("Elapsed (s)", f"{result.elapsed_sec:.1f}")

    render_top_candidates(result.top_candidates)
    render_chemistry_summary(result.scaffolds, result.functional_groups)
    render_hardware_validation(result)

    if result.explanation:
        st.subheader("Gemini explanation")
        st.markdown(result.explanation)

    with st.expander("All scored candidates"):
        render_results_table(result.candidates)


def tab_score_custom(cfg: dict) -> None:
    st.header("Score your own SMILES")
    st.caption(
        "Paste one SMILES per line.  Each one is encoded to a 3D geometry, "
        "lifted to a molecular Hamiltonian (STO-3G), and scored with VQE."
    )

    default = "CCO\nc1ccccc1\nCC(=O)O"
    text = st.text_area("SMILES (one per line)", value=default, height=160)

    if st.button("Score", type="primary"):
        smiles = [s.strip() for s in text.splitlines() if s.strip()]
        if not smiles:
            st.warning("Please enter at least one SMILES.")
            return

        scorer = get_scorer(
            cfg["active_electrons"],
            cfg["active_orbitals"],
            cfg["vqe_steps"],
            cfg["vqe_stepsize"],
            cfg["max_heavy_atoms"],
            cfg["hamiltonian_timeout_sec"],
            cfg["optimization_timeout_sec"],
        )
        progress_bar = st.progress(0.0, text="Scoring...")
        results: List[VQEResult] = []
        for i, smi in enumerate(smiles, start=1):
            progress_bar.progress((i - 1) / len(smiles), text=f"Scoring {smi}...")
            results.append(scorer.score(smi))
        progress_bar.progress(1.0, text="Done.")

        successful = sorted(
            (r for r in results if r.succeeded), key=lambda r: r.energy
        )
        if successful:
            st.subheader("Ranked results")
            render_top_candidates(successful)
            scaffolds = extract_scaffolds(successful)
            groups = extract_functional_groups(successful)
            render_chemistry_summary(scaffolds, groups)
        else:
            st.warning("No SMILES could be scored successfully.")

        with st.expander("Raw VQE results"):
            render_results_table(results)


def tab_validate_one(cfg: dict) -> None:
    st.header("Cross-validate one SMILES on Qiskit Aer")
    st.caption(
        "Runs a short VQE on the Qiskit Aer simulator (falls back to "
        "`default.qubit` if Aer is unavailable).  Used as a backend "
        "consistency check on a candidate of interest."
    )
    smiles = st.text_input("SMILES", value="CCO")
    if st.button("Validate"):
        with st.spinner("Running VQE on Qiskit Aer..."):
            t0 = time.time()
            hv = validate_on_hardware(
                smiles,
                active_electrons=cfg["active_electrons"],
                active_orbitals=cfg["active_orbitals"],
                basis=QUANTUM_CONFIG.get("basis", "sto-3g"),
                vqe_steps=min(40, cfg["vqe_steps"]),
                stepsize=cfg["vqe_stepsize"],
            )
            elapsed = time.time() - t0
        if hv.succeeded:
            st.success(f"Energy: {hv.energy:+.4f} Ha (qubits={hv.n_qubits}, {elapsed:.1f}s)")
            img = render_smiles(smiles, size=320)
            if img is not None:
                st.image(img, caption=smiles)
        else:
            st.error(f"Validation failed: {hv.error}")


def tab_dataset() -> None:
    st.header("Dataset preview")
    dataset_path = discover_dataset()
    if dataset_path is None:
        st.warning(
            "No dataset detected in `data/`.  Drop `Zinc_250K.csv` (or any CSV "
            "with a `smiles` column) in there to enable training on real data. "
            "The pipeline will currently fall back to a small built-in list."
        )
    else:
        st.success(f"Active dataset: `{dataset_path.name}`")

    n = st.slider("Preview size", 10, 200, 50)
    smiles = get_dataset_preview(n)
    st.write(f"Showing {len(smiles)} filtered SMILES (\u2264 12 heavy atoms).")

    cols = st.columns(5)
    for i, smi in enumerate(smiles):
        with cols[i % 5]:
            img = render_smiles(smi, size=180)
            if img is not None:
                st.image(img, caption=smi, use_container_width=True)
            else:
                st.code(smi)


def tab_about() -> None:
    st.header("About")
    st.markdown(
        """
        **Quantum-AI Drug Discovery Pipeline**

        This research tool combines three ideas:

        - **MolGAN (WGAN-GP)** generates candidate small-molecule SMILES from
          one-hot graph tensors.
        - **VQE on PennyLane Lightning** estimates each candidate's
          ground-state energy via a 3-layer hardware-efficient ansatz on a
          PySCF Hamiltonian (STO-3G, active-space approximation).
        - **Gemini** writes a plain-English explanation of which molecules
          were selected and why.

        **Limitations**

        - 2 electron / 2 orbital active space \u2192 limited chemical accuracy.
        - Simulator-only; no real quantum hardware.
        - Heavy-atom ceiling of 12; vocabulary limited to C, N, O, F, S, Cl.
        - Outputs are a methodological demonstration, not real drug candidates.
        """
    )

    st.subheader("Active configuration")
    st.json({
        "llm": LLM_CONFIG,
        "simulation": SIMULATION_CONFIG,
        "quantum": QUANTUM_CONFIG,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("Quantum-AI Drug Discovery")
    st.caption(
        "Generate \u2192 quantum-score \u2192 refine \u2192 explain.  "
        "MolGAN + VQE + Gemini, end-to-end."
    )

    cfg = sidebar_config()

    tabs = st.tabs([
        "Generate & score",
        "Score custom SMILES",
        "Validate on Aer",
        "Dataset",
        "About",
    ])
    with tabs[0]:
        tab_generate_and_score(cfg)
    with tabs[1]:
        tab_score_custom(cfg)
    with tabs[2]:
        tab_validate_one(cfg)
    with tabs[3]:
        tab_dataset()
    with tabs[4]:
        tab_about()


if __name__ == "__main__":
    main()
