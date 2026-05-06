# Quantum-AI Drug Discovery Pipeline

A research pipeline that combines **quantum chemistry simulation** (VQE-based
molecular scoring), **graph generative adversarial networks** (MolGAN-style
WGAN-GP on real graph encodings of SMILES), and **large language models**
(Google Gemini) into a closed feedback loop for exploring small-molecule
drug candidates.

## Pipeline overview

```
                    +---------------------+
                    |  ZINC SMILES (CSV)  |
                    +----------+----------+
                               |
                               v
+------------------------------+--------------------------------+
| src.smiles_graph: SMILES <-> one-hot graph tensors            |
+------------------------------+--------------------------------+
                               |
                               v
+------------------------------+--------------------------------+
| src.models.molgan:                                            |
|   - WGAN-GP trained on real graph tensors                     |
|   - .sample_smiles() decodes generator output -> valid SMILES |
|   - .fine_tune() biases toward top scorers each iteration     |
+------------------------------+--------------------------------+
                               |
                               v   (per iteration, x N)
+------------------------------+--------------------------------+
| src.quantum.vqe.VQEScorer:                                    |
|   SMILES -> 3D geometry -> molecular Hamiltonian -> VQE       |
|   (PennyLane Lightning, STO-3G, configurable active space,    |
|    cached per SMILES, structured VQEResult return)            |
+------------------------------+--------------------------------+
                               |
                               v
+------------------------------+--------------------------------+
| src.pipeline:                                                 |
|   classifier (percentile) -> feature_extractor (Murcko        |
|   scaffolds + functional groups) -> feedback (build           |
|   fine-tune dataset of boosted top + base sample)             |
+------------------------------+--------------------------------+
                               |
                               v   (after final iteration)
+------------------------------+--------------------------------+
| src.quantum.hardware.HardwareValidator:                       |
|   re-runs VQE on Qiskit Aer for the best candidate            |
+------------------------------+--------------------------------+
                               |
                               v
+------------------------------+--------------------------------+
| src.llm.explain: Gemini -> natural-language summary           |
+----------------------------------------------------------------+
```

Every box above is wired into `src/pipeline/orchestrator.py`. Each module
also has a clean public API so it can be used standalone in a notebook.

## Project layout

```
Quantum AI Drug Discovery/
├── run.py                       # CLI entry point (full train+inference loop)
├── app.py                       # Streamlit UI for inference
├── config.json                  # Tunable hyperparameters
├── requirements.txt
├── .env                         # GOOGLE_API_KEY (do NOT commit)
├── README.md
├── data/
│   └── Zinc_250K.csv            # User-provided dataset (auto-detected)
├── checkpoints/                 # Saved MolGAN weights (created by training)
├── scripts/
│   ├── train.py                 # Pretrain MolGAN -> checkpoints/molgan.pt
│   ├── check_gemini.py          # List available Gemini models
│   └── vqe_smoke_test.py        # LiH VQE sanity check
└── src/
    ├── __init__.py
    ├── config.py                # .env + config.json loader
    ├── logging_utils.py         # Project-wide logger
    ├── data_loader.py           # ZINC loader (auto-detects any CSV)
    ├── smiles_graph.py          # SMILES <-> one-hot graph tensors
    ├── inference.py             # High-level inference helpers
    ├── models/
    │   └── molgan.py            # Generator, Discriminator, save/load
    ├── quantum/
    │   ├── geometry.py          # SMILES -> 3D Bohr coordinates
    │   ├── vqe.py               # VQEScorer (cached, structured results)
    │   └── hardware.py          # Qiskit Aer second-opinion validator
    ├── pipeline/
    │   ├── classifier.py        # Percentile good/bad split
    │   ├── feature_extractor.py # Murcko scaffolds + functional groups
    │   ├── feedback.py          # Build fine-tuning dataset
    │   └── orchestrator.py      # End-to-end run_pipeline()
    └── llm/
        └── explain.py           # Gemini natural-language explanation
```

## Tech stack

| Layer | Library |
|---|---|
| Quantum simulation | PennyLane (`lightning.qubit`), Qiskit Aer |
| Quantum chemistry | PySCF (Hamiltonian, STO-3G, active-space approximation) |
| Deep learning | PyTorch (WGAN-GP) |
| Cheminformatics | RDKit (parsing, ETKDG embedding, UFF, Murcko scaffolds) |
| LLM | `google-generativeai` (Gemini) |
| Data | pandas, NumPy |

## Installation

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

If RDKit fails via pip, prefer conda: `conda install -c conda-forge rdkit`.

## Environment

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_api_key_here
```

You only need this for the explanation step; the rest of the pipeline works
without it.  Pass `--no-explanation` to skip the LLM stage entirely.

## Data setup

Drop a SMILES dataset into `data/`.  The loader auto-detects any CSV
containing a `smiles`-like column (case-insensitive: `smiles`, `SMILES`,
`canonical_smiles`, `smi`); the documented default is `Zinc_250K.csv`.
If no dataset is present the loader falls back to a small built-in list so
you can still exercise the pipeline end-to-end.

## Usage

There are three ways to run the project, in increasing order of friendliness:

### 1. Streamlit UI (recommended for inference)

```bash
# One-time pretraining (run whenever the dataset or architecture changes)
python scripts/train.py

# Launch the interactive UI
streamlit run app.py
```

The UI exposes five tabs:

- **Generate & score** \u2014 sample novel candidates from MolGAN, score with VQE,
  view 2D structures, scaffolds, functional groups, hardware validation, and
  optional Gemini explanation.
- **Score custom SMILES** \u2014 paste your own molecules to be VQE-scored.
- **Validate on Aer** \u2014 run a single SMILES through Qiskit Aer.
- **Dataset** \u2014 preview the active ZINC subset with rendered structures.
- **About** \u2014 active configuration and limitations.

### 2. Pretraining only

```bash
python scripts/train.py --epochs 200 --dataset-size 20000
python scripts/train.py --output checkpoints/molgan_big.pt
```

### 3. CLI full pipeline (train + iterate + score in one shot)

```bash
python run.py                                 # end-to-end with config defaults
python run.py --epochs 30 --iterations 1      # quick smoke run
python run.py --no-explanation                # skip Gemini
python run.py --per-iter 10 --top-k 3         # tighter loop
```

### Standalone smoke tests

```bash
python scripts/vqe_smoke_test.py     # LiH VQE convergence (~ -7.86 Ha)
python scripts/check_gemini.py       # Lists Gemini models you can call
```

## Configuration

All hyperparameters are centralized in `config.json` under four sections:

| Section | What it controls |
|---|---|
| `llm` | Gemini model name, explanation rounds |
| `simulation` | Iterations, candidates per iteration, percentile cut, feedback boost |
| `molgan` | Latent dim, hidden sizes, learning rates, n_critic, gradient penalty |
| `quantum` | Basis set, active space, VQE step count and step size |

## Module-by-module reference

### `src/smiles_graph.py`

Real SMILES <-> graph tensor conversion using a 7-class atom space
(C, N, O, F, S, Cl, PAD) and a 5-class bond space (NoBond, Single, Double,
Triple, Aromatic).  Provides:

- `smiles_to_graph(s)` -> `(nodes, edges)` one-hot tensors or `None`.
- `encode_smiles_dataset(list)` -> stacked tensors + canonical SMILES that
  survived encoding.
- `graph_to_mol(nodes, edges)` -> RDKit `Mol` with sanitization and a
  fallback that keeps the largest sanitizable connected fragment.
- `batch_decode_to_smiles(...)` -> list of canonical SMILES (or `None`).

### `src/models/molgan.py`

`MolGAN` class wrapping a generator/discriminator pair, their optimizers and
training/sampling logic.  Notable behaviour:

- Trains on **real** graph encodings produced by `encode_smiles_dataset`.
- Generator outputs **soft one-hot** edge/node tensors via softmax,
  symmetrized along the edge axis.
- `sample_smiles(n)` retries decoding until `n` unique sanitizable SMILES
  are produced (or a try-budget is exceeded).
- `fine_tune(top_smiles, epochs)` re-trains briefly on a feedback dataset.

### `src/quantum/vqe.py`

`VQEScorer` builds a `qml.qchem.molecular_hamiltonian` from RDKit-derived
geometry, uses a 3-layer hardware-efficient ansatz (RY + CNOT + RY layers),
and runs Adam optimization for `vqe_steps`.  Returns a structured
`VQEResult(smiles, energy, n_qubits, steps_run, status, error)`.  Per-SMILES
results are cached for the lifetime of the scorer.

### `src/quantum/hardware.py`

`HardwareValidator` runs an actual short VQE optimization on the
`qiskit.aer` device for the best candidate (falls back to `default.qubit`
if Aer is unavailable).  This replaces the original "single forward pass at
theta=0" placeholder.

### `src/pipeline/feature_extractor.py`

Uses RDKit `MurckoScaffold` to extract recurring core skeletons from top
candidates, plus a small SMARTS library for common functional groups
(hydroxyl, amide, carbonyl, carboxyl, ether, nitrile, halide, ...).

### `src/pipeline/feedback.py`

Builds the dataset that `MolGAN.fine_tune` consumes between iterations:
the top molecules are oversampled by `feedback_boost_factor` and mixed with
a random sample of the base corpus to prevent mode collapse.

### `src/pipeline/orchestrator.py`

`run_pipeline(...)` executes every stage in order and returns a
`PipelineResult` containing per-iteration reports, final candidates, the
hardware validation result, and the LLM explanation.

## Limitations

- **Active-space approximation**: 2 electrons / 2 orbitals on STO-3G keeps
  VQE tractable but limits chemical accuracy of the energy ranking.
- **Simulator-only**: `lightning.qubit` and `qiskit.aer` are classical
  simulators; this is not a substitute for real quantum hardware results.
- **Heavy-atom ceiling**: SMILES are filtered to <=12 heavy atoms so PySCF
  Hamiltonian construction stays under typical laptop memory.
- **Atom vocabulary**: only C, N, O, F, S, Cl are supported in graph
  encoding/decoding.  Other elements are silently dropped during dataset
  encoding.
- **Proof of concept**: scaffolds learned by the GAN are conditioned on a
  small corpus; do not interpret outputs as real drug candidates.

## Troubleshooting

| Issue | Fix |
|---|---|
| `ValueError: GOOGLE_API_KEY is not set` | Add it to `.env` or run with `--no-explanation`. |
| `ModuleNotFoundError: rdkit` | `pip install rdkit` or `conda install -c conda-forge rdkit`. |
| `ModuleNotFoundError: pyscf` | `pip install pyscf` (Linux/macOS preferred; Windows: WSL). |
| Many `geometry_failed` statuses | Reduce `molecules_per_iter` or lower `--epochs`; the GAN may need more pretraining. |
| Slow VQE | Lower `quantum.vqe_steps` or `simulation.iterations` in `config.json`. |
| Out of memory in MolGAN | Lower `molgan.batch_size`. |

## License

Provided as-is for research and educational purposes.
