"""Independent appearance checkpoints tied to exact geometry and vertex order."""
import hashlib
import os
import tempfile
from pathlib import Path
import torch


def file_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def layout_fingerprint(model):
    h = hashlib.sha256()
    for tensor in (model.octpath, model.octlevel, model.grid_pts_key,
                   model.scene_center, model.scene_extent):
        value = tensor.detach().cpu().contiguous()
        h.update(str((tuple(value.shape), value.dtype)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def save_checkpoint(path, module, optimizer, step, config, identity, generator, frozen_sh0=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(schema_version=2, mode=config['mode'], config=config, identity=identity, step=step,
                 vertex_module=module.state_dict(), optimizer=optimizer.state_dict(),
                 torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                 camera_rng=generator.get_state())
    if frozen_sh0 is not None:
        state["frozen_sh0"] = frozen_sh0.detach().cpu()
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, module, identity, optimizer=None, generator=None):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['schema_version'] != 2 or state['identity'] != identity:
        raise ValueError('Appearance checkpoint geometry/layout mismatch')
    modes = ("vertex_two_bases", "vertex_two_bases_bounded")
    saved_mode = state["mode"] if state["mode"] in modes else None
    if saved_mode != getattr(module, "appearance_mode", None):
        raise ValueError("Appearance checkpoint mode mismatch")
    module.load_state_dict(state['vertex_module'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        generator.set_state(state['camera_rng'])
    return state
