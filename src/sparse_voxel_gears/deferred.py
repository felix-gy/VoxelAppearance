"""Deferred appearance decoder and versioned checkpoint I/O."""
import hashlib
import os
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

MODE = "deferred_mlp"
VERSION = 1


class DeferredAppearance(nn.Module):
    def __init__(self, voxels, device):
        super().__init__()
        self.features = nn.Parameter(torch.randn(voxels, 8, device=device) * 0.01)
        self.decoder = nn.Sequential(
            nn.Linear(29, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 3),
        ).to(device)
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, features, base, opacity, direction):
        direction = F.normalize(direction, dim=0)
        fourier = [fn(scale * direction) for scale in (1, 2, 4) for fn in (torch.sin, torch.cos)]
        inputs = torch.cat([features, base, *fourier]).permute(1, 2, 0)
        return base + opacity * self.decoder(inputs).permute(2, 0, 1)


def layout_digest(model):
    digest = hashlib.sha256()
    for value in (model.octpath, model.octlevel, model.grid_pts_key, model.vox_key):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def save_deferred(model, root, step, config, training):
    directory = Path(root) / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"iter{step:06d}_model.pt"
    state = {
        "appearance_mode": MODE, "version": VERSION, "step": step,
        "layout": layout_digest(model), "config": config,
        "model": {k: v.cpu() if torch.is_tensor(v) else v
                  for k, v in model.model_state().items()},
        "appearance": {k: v.cpu() for k, v in model.deferred.state_dict().items()},
        "training": training,
    }
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as tmp:
        temp = Path(tmp.name)
    try:
        torch.save(state, temp)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return path


def restore_deferred(model, state):
    if state.get("appearance_mode") != MODE or state.get("version") != VERSION:
        raise ValueError("Expected a compatible deferred_mlp checkpoint")
    model.load_model_state(state["model"])
    if layout_digest(model) != state["layout"]:
        raise ValueError("Deferred checkpoint layout mismatch")
    model.init_deferred()
    expected = model.deferred.state_dict()
    actual = state["appearance"]
    if expected.keys() != actual.keys() or any(
        actual[k].shape != value.shape or actual[k].dtype != value.dtype
        for k, value in expected.items()
    ):
        raise ValueError("Deferred checkpoint appearance layout mismatch")
    model.deferred.load_state_dict(actual)
