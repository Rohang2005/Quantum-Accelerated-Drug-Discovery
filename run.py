"""Top-level entry point for the quantum-AI drug discovery pipeline.

Usage::

    python run.py
    python run.py --no-explanation        # skip the Gemini step
    python run.py --epochs 30             # short MolGAN pretrain
    python run.py --iterations 1          # one feedback loop only
"""

from __future__ import annotations

import argparse

from src.pipeline import run_pipeline


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the quantum-AI drug discovery pipeline.")
    p.add_argument("--dataset-size", type=int, default=10_000,
                   help="Max ZINC SMILES to load (default: 10000).")
    p.add_argument("--epochs", type=int, default=None,
                   help="MolGAN pretraining epochs (default: from config.json).")
    p.add_argument("--fine-tune-epochs", type=int, default=None,
                   help="Per-iteration fine-tuning epochs (default: from config.json).")
    p.add_argument("--iterations", type=int, default=None,
                   help="Number of generate-score-feedback iterations.")
    p.add_argument("--per-iter", type=int, default=None,
                   help="Candidates sampled per iteration.")
    p.add_argument("--top-k", type=int, default=None,
                   help="Number of final candidates to keep.")
    p.add_argument("--no-explanation", action="store_true",
                   help="Skip the Gemini explanation stage.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_pipeline(
        dataset_size=args.dataset_size,
        molgan_epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        iterations=args.iterations,
        molecules_per_iter=args.per_iter,
        final_candidates_count=args.top_k,
        enable_explanation=not args.no_explanation,
    )

    print("\n=== TOP CANDIDATES ===")
    for r in result.final_candidates:
        print(f"  {r.smiles:<40s} energy={r.energy:+.4f} Ha  qubits={r.n_qubits}")

    if result.hardware_validation and result.hardware_validation.succeeded:
        hv = result.hardware_validation
        print(f"\nHardware validation: {hv.smiles} -> {hv.energy:+.4f} Ha")

    if result.explanation:
        print("\n=== EXPLANATION ===")
        print(result.explanation)

    print(f"\nTotal time: {result.elapsed_sec:.1f}s")


if __name__ == "__main__":
    main()
