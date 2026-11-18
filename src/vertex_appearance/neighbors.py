"""Same-level octree face neighbors and coefficient regularization."""
import torch

import svraster_cuda


def face_pairs(model, *, seed=3721, num_samples=8192, max_faces_per_axis=1024):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    sample = torch.randperm(model.num_voxels, generator=generator,
                            device="cuda")[:num_samples]
    levels = model.octlevel[sample]
    coords = svraster_cuda.utils.octpath_2_ijk(model.octpath[sample], levels)
    paths, order = model.octpath.flatten().sort()
    for axis in range(3):
        query = coords.clone()
        query[:, axis] += 1
        valid = query[:, axis] < (1 << levels.flatten().long())
        neighbor_paths = svraster_cuda.utils.ijk_2_octpath(query[valid], levels[valid]).flatten()
        index = torch.searchsorted(paths, neighbor_paths).clamp_max(len(paths)-1)
        left, right = sample[valid], order[index]
        present = ((paths[index] == neighbor_paths) &
                   (model.octlevel[left, 0] == model.octlevel[right, 0]))
        yield axis, left[present][:max_faces_per_axis], right[present][:max_faces_per_axis]


def coefficient_penalty(model, *, strength, seed=7919, batch_size=4096):
    pairs = [torch.stack((left, right), dim=1)
             for _, left, right in face_pairs(model, seed=seed, num_samples=65536,
                                              max_faces_per_axis=65536)]
    edges = torch.cat(pairs)
    if not len(edges):
        raise ValueError("Coefficient regularization requires shared same-level faces")

    def penalty(module, step):
        generator = torch.Generator(device=edges.device).manual_seed(seed + step)
        indices = torch.randint(len(edges), (batch_size,), device=edges.device,
                                generator=generator)
        left, right = edges[indices].unbind(dim=1)
        return strength * (module.mix[left] - module.mix[right]).square().mean()

    return penalty, len(edges)
