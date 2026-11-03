"""Optimizer construction shared by training and integration checks."""

import torch


def create_voxel_optimizer(model, config, optimizer_class=None):
    if optimizer_class is None:
        from svraster_cuda.sparse_adam import SparseAdam
        optimizer_class = SparseAdam
    groups = [
        ('geometry', model._geo_grid_pts, config.geo_lr),
        ('albedo', model._albedo, config.albedo_lr),
        ('roughness', model._roughness, config.roughness_lr),
        ('tint', model._specular_tint, config.tint_lr),
        ('bottleneck', model._bottleneck, config.bottleneck_lr),
    ]
    if any(value is None for _, value, _ in groups):
        raise ValueError("Initialize geometry and appearance before creating the optimizer; "
                         "unfreeze unshared corner geometry before training.")
    return optimizer_class(
        [dict(name=name, params=[value], lr=lr) for name, value, lr in groups],
        betas=(config.optim_beta1, config.optim_beta2), eps=config.optim_eps)


def create_specular_optimizer(model, config):
    # Keep this optimizer alive when adaptation replaces the voxel tensors.
    return torch.optim.Adam(
        model._specular_mlp.parameters(), lr=config.specular_lr,
        betas=(config.optim_beta1, config.optim_beta2),
        eps=max(config.optim_eps, 1e-8))


def create_scheduler(optimizer, config, completed_iterations=0):
    """Resume the LR schedule even when optimizer moments are intentionally reset."""
    milestones = config.lr_decay_ckpt
    previous_decays = sum(m < completed_iterations for m in milestones)
    for group in optimizer.param_groups:
        base_lr = group.setdefault('initial_lr', group['lr'])
        group['lr'] = base_lr * config.lr_decay_mult ** previous_decays
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=config.lr_decay_mult,
        last_epoch=completed_iterations - 1)
