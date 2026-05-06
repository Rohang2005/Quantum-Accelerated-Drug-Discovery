"""Pretrain the MolGAN generator on the ZINC subset and save a checkpoint.

Usage::

    python scripts/train.py
    python scripts/train.py --epochs 200 --dataset-size 20000
    python scripts/train.py --output checkpoints/molgan_big.pt

The resulting checkpoint is what the Streamlit UI (``app.py``) and the
inference helpers in :mod:`src.inference` consume.  Re-running training is
only necessary when you change the architecture, the dataset, or the
encoding vocabulary.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SIMULATION_CONFIG  # noqa: E402
from src.data_loader import load_zinc_subset  # noqa: E402
from src.logging_utils import get_logger  # noqa: E402
from src.models.molgan import MolGAN  # noqa: E402

logger = get_logger("train")

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "molgan.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain MolGAN on the ZINC subset.")
    p.add_argument("--dataset-size", type=int, default=10_000,
                   help="Max SMILES to pull from the ZINC CSV.")
    p.add_argument("--max-heavy-atoms", type=int, default=12,
                   help="Drop molecules with more heavy atoms than this.")
    p.add_argument("--epochs", type=int,
                   default=SIMULATION_CONFIG.get("molgan_epochs", 100),
                   help="Number of WGAN-GP training epochs.")
    p.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT,
                   help="Path to write the checkpoint to.")
    p.add_argument("--log-every", type=int, default=10,
                   help="Log training stats every N epochs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Loading dataset (n=%d, max_heavy_atoms=%d)",
                args.dataset_size, args.max_heavy_atoms)
    corpus = load_zinc_subset(n=args.dataset_size, max_heavy_atoms=args.max_heavy_atoms)
    logger.info("Training corpus size: %d", len(corpus))

    if len(corpus) < 32:
        logger.warning(
            "Only %d molecules available; training will be unstable. "
            "Drop a real ZINC CSV into the data/ folder for meaningful results.",
            len(corpus),
        )

    started = time.time()
    molgan = MolGAN()
    molgan.train(corpus, epochs=args.epochs, log_every=args.log_every)
    elapsed = time.time() - started

    args.output.parent.mkdir(parents=True, exist_ok=True)
    molgan.save_checkpoint(args.output)
    logger.info("Training complete in %.1fs. Checkpoint -> %s", elapsed, args.output)


if __name__ == "__main__":
    main()
