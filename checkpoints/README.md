# Checkpoints

This folder stores trained MolGAN weights produced by `scripts/train.py`.

The default checkpoint path is `checkpoints/molgan.pt`. The Streamlit UI
(`app.py`) and the inference helpers in `src/inference.py` load from here
unless given an explicit path.

Run pretraining once:

```bash
python scripts/train.py
```

Re-train when:
- the dataset under `data/` changes,
- you change the MolGAN architecture or graph encoding vocabulary in
  `src/smiles_graph.py` or `src/models/molgan.py`,
- you want a fresh model.
