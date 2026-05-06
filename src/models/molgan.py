"""MolGAN-style WGAN-GP that actually learns from real molecular graphs.

The generator emits a pair of tensors ``(edges, nodes)`` matching the layout in
:mod:`src.smiles_graph`.  Training is the standard WGAN-GP recipe with
``n_critic=5`` and a gradient penalty of ``lambda=10``.  Two hooks make this
generator usable as part of a feedback loop:

* :meth:`MolGAN.sample_smiles` decodes generator output back into SMILES via
  :mod:`src.smiles_graph` and retries until ``n`` valid molecules are produced.
* :meth:`MolGAN.fine_tune` lets the orchestrator nudge the generator toward
  the top VQE-scored candidates from a previous iteration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union

SampleProgress = Callable[[int, int, int, int], None]
"""``progress(collected, target, attempt, max_tries)``."""

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from ..config import MOLGAN_CONFIG
from ..logging_utils import get_logger
from ..smiles_graph import (
    ATOM_TYPES,
    BOND_TYPES,
    MAX_ATOMS,
    batch_decode_to_smiles,
    encode_smiles_dataset,
)

logger = get_logger("molgan")


@dataclass
class MolGANConfig:
    z_dim: int = 32
    g_hidden: int = 256
    d_hidden: int = 256
    lr_g: float = 1e-4
    lr_d: float = 1e-4
    betas: tuple = (0.5, 0.9)
    n_critic: int = 5
    gp_lambda: float = 10.0
    batch_size: int = 64

    @classmethod
    def from_config(cls) -> "MolGANConfig":
        return cls(
            z_dim=MOLGAN_CONFIG.get("z_dim", 32),
            g_hidden=MOLGAN_CONFIG.get("g_hidden", 256),
            d_hidden=MOLGAN_CONFIG.get("d_hidden", 256),
            lr_g=MOLGAN_CONFIG.get("lr_g", 1e-4),
            lr_d=MOLGAN_CONFIG.get("lr_d", 1e-4),
            n_critic=MOLGAN_CONFIG.get("n_critic", 5),
            gp_lambda=MOLGAN_CONFIG.get("gp_lambda", 10.0),
            batch_size=MOLGAN_CONFIG.get("batch_size", 64),
        )


class _Generator(nn.Module):
    def __init__(self, cfg: MolGANConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.g_hidden
        self.trunk = nn.Sequential(
            nn.Linear(cfg.z_dim, h),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h, h * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h * 2, h * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.edge_head = nn.Linear(h * 4, MAX_ATOMS * MAX_ATOMS * BOND_TYPES)
        self.node_head = nn.Linear(h * 4, MAX_ATOMS * ATOM_TYPES)

    def forward(self, z: torch.Tensor):
        h = self.trunk(z)
        e_logits = self.edge_head(h).view(-1, MAX_ATOMS, MAX_ATOMS, BOND_TYPES)
        n_logits = self.node_head(h).view(-1, MAX_ATOMS, ATOM_TYPES)
        # Symmetrize edge logits so the decoder sees an undirected graph.
        e_logits = 0.5 * (e_logits + e_logits.transpose(1, 2))
        # Soft one-hot via Gumbel-softmax for differentiable sampling.
        edges = torch.softmax(e_logits, dim=-1)
        nodes = torch.softmax(n_logits, dim=-1)
        return edges, nodes


class _Discriminator(nn.Module):
    def __init__(self, cfg: MolGANConfig):
        super().__init__()
        in_dim = (MAX_ATOMS * MAX_ATOMS * BOND_TYPES) + (MAX_ATOMS * ATOM_TYPES)
        h = cfg.d_hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(h * 2, h),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h, 1),  # WGAN critic - linear output, no sigmoid.
        )

    def forward(self, edges: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        x = torch.cat([edges.flatten(1), nodes.flatten(1)], dim=1)
        return self.net(x)


class MolGAN:
    """Owner of the generator/discriminator pair plus their optimizers."""

    def __init__(self, cfg: Optional[MolGANConfig] = None, device: Optional[str] = None):
        self.cfg = cfg or MolGANConfig.from_config()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.generator = _Generator(self.cfg).to(self.device)
        self.discriminator = _Discriminator(self.cfg).to(self.device)
        self.opt_g = optim.Adam(self.generator.parameters(), lr=self.cfg.lr_g, betas=self.cfg.betas)
        self.opt_d = optim.Adam(self.discriminator.parameters(), lr=self.cfg.lr_d, betas=self.cfg.betas)

    def train(
        self,
        smiles_list: List[str],
        epochs: int = 100,
        log_every: int = 10,
    ) -> None:
        """Train on a real SMILES dataset.

        Encoding is run once up-front; if no SMILES survive we exit early
        rather than waste epochs on noise.
        """
        nodes_real, edges_real, kept = encode_smiles_dataset(smiles_list)
        if len(kept) == 0:
            logger.warning("MolGAN: no SMILES to train on; aborting.")
            return

        nodes_real = nodes_real.to(self.device)
        edges_real = edges_real.to(self.device)
        n_samples = nodes_real.shape[0]
        bs = min(self.cfg.batch_size, n_samples)
        n_batches = max(1, n_samples // bs)

        logger.info(
            "Training MolGAN on %d molecules | device=%s | epochs=%d | batches/epoch=%d",
            n_samples, self.device.type, epochs, n_batches,
        )

        for epoch in range(1, epochs + 1):
            perm = torch.randperm(n_samples, device=self.device)
            d_total, g_total = 0.0, 0.0

            for b in range(n_batches):
                idx = perm[b * bs : (b + 1) * bs]
                real_edges = edges_real[idx]
                real_nodes = nodes_real[idx]
                cur_bs = real_edges.size(0)

                for _ in range(self.cfg.n_critic):
                    self.opt_d.zero_grad(set_to_none=True)
                    d_real = self.discriminator(real_edges, real_nodes).mean()
                    z = torch.randn(cur_bs, self.cfg.z_dim, device=self.device)
                    fake_edges, fake_nodes = self.generator(z)
                    d_fake = self.discriminator(fake_edges.detach(), fake_nodes.detach()).mean()
                    gp = self._gradient_penalty(
                        real_edges, real_nodes, fake_edges.detach(), fake_nodes.detach()
                    )
                    d_loss = d_fake - d_real + self.cfg.gp_lambda * gp
                    d_loss.backward()
                    self.opt_d.step()
                    d_total += d_loss.item()

                self.opt_g.zero_grad(set_to_none=True)
                z = torch.randn(cur_bs, self.cfg.z_dim, device=self.device)
                fake_edges, fake_nodes = self.generator(z)
                g_loss = -self.discriminator(fake_edges, fake_nodes).mean()
                g_loss.backward()
                self.opt_g.step()
                g_total += g_loss.item()

            if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
                d_avg = d_total / (n_batches * self.cfg.n_critic)
                g_avg = g_total / n_batches
                validity = self._spot_check_validity(64)
                logger.info(
                    "Epoch %3d/%d | D %+0.4f | G %+0.4f | valid %.1f%%",
                    epoch, epochs, d_avg, g_avg, validity,
                )

    def fine_tune(self, smiles_list: List[str], epochs: int = 20) -> None:
        """Short fine-tuning run biased toward an external SMILES set.

        Used by the feedback loop to nudge the generator toward the lowest
        VQE-energy molecules found in a prior iteration.
        """
        if not smiles_list:
            logger.info("Fine-tune skipped: no top candidates.")
            return
        logger.info("Fine-tuning MolGAN on %d top molecules for %d epochs", len(smiles_list), epochs)
        self.train(smiles_list, epochs=epochs, log_every=max(1, epochs // 5))

    @torch.no_grad()
    def sample_smiles(
        self,
        n: int,
        max_tries: int = 10,
        progress: Optional["SampleProgress"] = None,
        max_heavy_atoms: Optional[int] = None,
    ) -> List[str]:
        """Generate ``n`` *valid* unique SMILES via repeated decoding.

        Each loop draws a batch of latents, decodes argmax outputs back into
        SMILES, and accumulates the unique sanitizable ones.  We bail out
        after ``max_tries`` so we never spin forever on a degenerate model.

        Parameters
        ----------
        max_heavy_atoms
            If set, skip decoded SMILES whose heavy-atom count exceeds this
            limit.  Use this to align sampling with your VQE budget so the
            scorer doesn't reject everything downstream.
        progress
            Optional callable ``progress(collected, target, attempt, max_tries)``
            so a UI can show progress while the sampler retries.
        """
        try:
            from rdkit import Chem
            _has_rdkit = True
        except ImportError:
            _has_rdkit = False

        def _heavy_atoms_ok(smi: str) -> bool:
            if not max_heavy_atoms or not _has_rdkit:
                return True
            mol = Chem.MolFromSmiles(smi)
            return mol is not None and mol.GetNumHeavyAtoms() <= max_heavy_atoms

        self.generator.eval()
        out: List[str] = []
        seen: set = set()
        # When filtering by size we need to oversample more aggressively.
        batch = max(n * 8, 64) if max_heavy_atoms else max(n * 2, 32)
        n_filtered = 0
        for attempt in range(1, max_tries + 1):
            if len(out) >= n:
                break
            z = torch.randn(batch, self.cfg.z_dim, device=self.device)
            edges, nodes = self.generator(z)
            decoded = batch_decode_to_smiles(nodes, edges)
            for smi in decoded:
                if not smi or smi in seen:
                    continue
                if smi.strip() == "":
                    continue
                if not _heavy_atoms_ok(smi):
                    n_filtered += 1
                    continue
                seen.add(smi)
                out.append(smi)
                if len(out) >= n:
                    break
            logger.info(
                "Sampling attempt %d/%d: %d unique valid SMILES so far"
                " (%d filtered as too large)",
                attempt, max_tries, len(out), n_filtered,
            )
            if progress is not None:
                progress(len(out), n, attempt, max_tries)
        if len(out) < n:
            logger.warning(
                "Only sampled %d/%d valid SMILES (max_heavy_atoms=%s); "
                "model may be biased toward larger structures.",
                len(out), n, max_heavy_atoms,
            )
        return out[:n]

    @torch.no_grad()
    def _spot_check_validity(self, batch: int = 64) -> float:
        self.generator.eval()
        z = torch.randn(batch, self.cfg.z_dim, device=self.device)
        edges, nodes = self.generator(z)
        decoded = batch_decode_to_smiles(nodes, edges)
        self.generator.train()
        valid = sum(1 for s in decoded if s)
        return 100.0 * valid / batch

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Persist generator/discriminator weights, optimizer state, and config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.cfg),
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
        }
        torch.save(payload, path)
        logger.info("MolGAN checkpoint saved to %s", path)

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        device: Optional[str] = None,
    ) -> "MolGAN":
        """Reconstruct a MolGAN from a checkpoint produced by :meth:`save_checkpoint`."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"MolGAN checkpoint not found: {path}")
        device_obj = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        payload = torch.load(path, map_location=device_obj, weights_only=False)
        cfg = MolGANConfig(**payload["config"])
        instance = cls(cfg=cfg, device=str(device_obj))
        instance.generator.load_state_dict(payload["generator"])
        instance.discriminator.load_state_dict(payload["discriminator"])
        try:
            instance.opt_g.load_state_dict(payload["opt_g"])
            instance.opt_d.load_state_dict(payload["opt_d"])
        except Exception as exc:
            logger.warning("Optimizer state not restored: %s", exc)
        logger.info("MolGAN checkpoint loaded from %s", path)
        return instance

    def _gradient_penalty(
        self,
        real_edges: torch.Tensor,
        real_nodes: torch.Tensor,
        fake_edges: torch.Tensor,
        fake_nodes: torch.Tensor,
    ) -> torch.Tensor:
        bs = real_edges.size(0)
        alpha = torch.rand(bs, 1, 1, 1, device=self.device)
        interp_edges = alpha * real_edges + (1 - alpha) * fake_edges
        interp_nodes = alpha.squeeze(-1) * real_nodes + (1 - alpha.squeeze(-1)) * fake_nodes
        interp_edges.requires_grad_(True)
        interp_nodes.requires_grad_(True)

        d_interp = self.discriminator(interp_edges, interp_nodes)
        grads = autograd.grad(
            outputs=d_interp,
            inputs=(interp_edges, interp_nodes),
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        flat = torch.cat([grads[0].flatten(1), grads[1].flatten(1)], dim=1)
        norm = flat.norm(2, dim=1)
        return ((norm - 1) ** 2).mean()
