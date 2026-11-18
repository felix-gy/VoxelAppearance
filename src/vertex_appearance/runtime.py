"""Fixed-layout experiment lifecycle shared by training and rendering."""
import argparse
import random
import json
from pathlib import Path
import torch
import yaml
from PIL import Image
from .decoder import VertexAppearance, VertexRGB, VertexTwoBases
from .checkpoint import file_digest, layout_fingerprint, save_checkpoint, load_checkpoint


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='cfg/vertex_fourier3_trilinear.yaml')
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int)
    p.add_argument('--benchmark', action='store_true')
    p.add_argument('--resume')
    p.add_argument('--appearance-checkpoint')
    p.add_argument('--geometry-checkpoint')
    p.add_argument('--mode', choices=['vertex_trilinear_fourier3', 'vertex_rgb_trilinear', 'vertex_two_bases'])
    return p


def setup(args):
    import svraster_cuda
    from src.sparse_voxel_model import SparseVoxelModel
    from src.dataloader.data_pack import DataPack
    root = Path(__file__).resolve().parents[2]
    if not Path(svraster_cuda.__file__).resolve().is_relative_to(root):
        raise RuntimeError('svraster_cuda must be installed from this checkout')
    config = yaml.safe_load(Path(args.config).read_text())
    if args.appearance_checkpoint or args.resume:
        saved = torch.load(args.appearance_checkpoint or args.resume, map_location='cpu', weights_only=False)
        config = saved['config']
    for key in ('steps', 'mode', 'geometry_checkpoint'):
        if getattr(args, key, None) is not None:
            config[key] = getattr(args, key)
    if config['mode'] not in ('vertex_trilinear_fourier3', 'vertex_rgb_trilinear', 'vertex_two_bases'):
        raise ValueError('Unsupported appearance mode')
    if config['steps'] < 1 or min(config['validation_interval'], config['checkpoint_interval']) < 1:
        raise ValueError('Steps and intervals must be positive')
    expected_appearance = dict(feature_dim=8, frequencies_pi=[1.0,2.0,4.0],
                               fourier_order='frequency_sin_xyz_cos_xyz',
                               include_raw_direction=False, direction='camera_to_vertex_world',
                               layers=[26,64,6 if config['mode'] == 'vertex_two_bases' else 3], dtype='float32')
    if config.get('appearance', expected_appearance) != expected_appearance:
        raise ValueError('This version requires the exact Fourier-3 architecture')
    config['appearance'] = expected_appearance
    base = config.get('resolved_base') or yaml.safe_load(Path(config['base_config']).read_text())
    if base['model']['n_samp_per_vox'] != 1 or base['auto_exposure']['enable']:
        raise ValueError('Requires one sample and disabled auto exposure')
    if base['regularizer']['use_l1'] or base['regularizer']['use_huber']:
        raise ValueError('This materials protocol requires MSE')
    config['resolved_base'] = base
    torch.manual_seed(config['seed'])
    model = SparseVoxelModel(**base['model'])
    model.load(config['geometry_checkpoint'])
    from .layout import validate_layout
    validate_layout(model)
    for key in ('_geo_grid_pts', '_sh0', '_shs', '_subdiv_p'):
        getattr(model, key).requires_grad_(False)
    model.freeze_vox_geo()
    if config['mode'] == 'vertex_two_bases':
        module = VertexTwoBases(model.num_grid_pts, model.num_voxels).cuda()
    else:
        kind = VertexAppearance if config['mode'] == 'vertex_trilinear_fourier3' else VertexRGB
        module = kind(model.num_grid_pts).cuda()
    if args.appearance_checkpoint or args.resume:
        if "frozen_sh0" in saved:
            model._sh0.copy_(saved["frozen_sh0"])
    data = dict(base['data'])
    data['use_test'] = data.pop('eval')
    data['alpha_is_white'] = base['model']['white_background']
    pack = DataPack(**data)
    identity = dict(checkpoint_sha256=file_digest(config['geometry_checkpoint']),
                    layout=layout_fingerprint(model), P=model.num_grid_pts, V=model.num_voxels)
    return config, base, model, module, pack, identity


def evaluate(model, module, cameras, directory, baseline=None, dense=False):
    from src.utils.loss_utils import psnr_score, ssim_score, lpips_loss, correct_lpips_loss
    directory = Path(directory)
    (directory / 'renders').mkdir(parents=True, exist_ok=True)
    (directory / 'gt').mkdir(exist_ok=True)
    rows = []
    module.eval()
    with torch.no_grad():
        for i, cam in enumerate(cameras):
            image = (model.render(cam, color_mode=baseline) if baseline else model.render_vertex(cam, module, decode_all_vertices=dense))['color']
            gt = cam.image.cuda()
            rows.append(dict(view=i, PSNR=psnr_score(image, gt).item(),
                             SSIM=ssim_score(image, gt).item(),
                             LPIPS=lpips_loss(image, gt).item(),
                             LPIPS_corrected=correct_lpips_loss(image, gt).item()))
            for sub, tensor in [('renders', image), ('gt', gt)]:
                Image.fromarray((tensor.clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype('uint8')).save(directory / sub / f'{i:04d}.png')
    result = dict(per_view=rows, mean={k: sum(r[k] for r in rows)/len(rows) for k in rows[0] if k != 'view'}) if rows else dict(per_view=[])
    (directory / 'metrics.json').write_text(json.dumps(result, indent=2))
    return result


def fit(model, module, cameras, camera_order, optimizer, start, stop, *, ssim_weight=0., dense=False, penalty=None):
    from src.utils.loss_utils import l2_loss, fast_ssim_loss

    for step in range(start + 1, stop + 1):
        module.train()
        camera = cameras[camera_order[step-1]]
        optimizer.zero_grad(set_to_none=True)
        image = model.render_vertex(camera, module, decode_all_vertices=dense)["color"]
        target = camera.image.cuda()
        loss = l2_loss(image, target)
        if ssim_weight:
            loss = loss + ssim_weight * fast_ssim_loss(image, target)
        if penalty is not None:
            loss = loss + penalty(module, step)
        if not loss.isfinite():
            raise RuntimeError("Nonfinite photometric loss")
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters()):
            raise RuntimeError("Nonfinite appearance gradient")
        optimizer.step()
        yield step, loss.item()


def train(args):
    config, base, model, module, pack, identity = setup(args)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'config.yaml').write_text(yaml.safe_dump(config))
    groups = ([dict(params=[module.vertex_features], lr=config['lr_vertex_features']),
               dict(params=module.decoder.parameters(), lr=config['lr_mlp'])]
              if isinstance(module, VertexAppearance) else
              [dict(params=module.parameters(), lr=config['lr_vertex_features'])])
    if isinstance(module, VertexTwoBases):
        groups.append(dict(params=[module.mix], lr=config.get("lr_mix", config["lr_vertex_features"])))
    comparison = config.get("comparison")
    betas = tuple(comparison["betas"]) if comparison else (0.9, 0.99)
    eps = comparison["eps"] if comparison else 1e-8
    optimizer = torch.optim.Adam(module.parameters() if comparison else groups,
                                 lr=config["lr_vertex_features"], betas=betas, eps=eps)
    generator = torch.Generator().manual_seed(config['seed'])
    start = 0
    if args.resume:
        start = load_checkpoint(args.resume, module, identity, optimizer, generator)['step']
    cameras = pack.get_train_cameras()
    test_cameras = pack.get_test_cameras()
    if comparison:
        train_by_name = {cam.image_name: cam for cam in cameras}
        test_by_name = {cam.image_name: cam for cam in test_cameras}
        cameras = [train_by_name[name] for name in comparison["train_views"]]
        test_cameras = [test_by_name[name] for name in comparison["test_views"]]
    # Match the baseline: shuffled complete epochs, never sample with replacement.
    camera_order = []
    camera_random = random.Random(config['seed'])
    while len(camera_order) < config['steps']:
        epoch = list(range(len(cameras)))
        if not comparison:
            camera_random.shuffle(epoch)
        camera_order.extend(epoch)
    torch.cuda.reset_peak_memory_stats()
    penalty = None
    if comparison and comparison.get("coefficient_penalty"):
        from .neighbors import coefficient_penalty
        if not isinstance(module, VertexTwoBases):
            raise ValueError("Coefficient regularization requires two bases")
        penalty, _ = coefficient_penalty(model, **comparison["coefficient_penalty"])
    for step, loss in fit(model, module, cameras, camera_order, optimizer, start, config["steps"],
                          ssim_weight=0. if comparison else base["regularizer"]["lambda_ssim"],
                          dense=bool(comparison and comparison["dense"]), penalty=penalty):
        if step == 1 or step % 100 == 0:
            print(f"step={step} loss={loss:.8f}", flush=True)
        if step % config['checkpoint_interval'] == 0 or step == config['steps']:
            save_checkpoint(directory/'checkpoints'/f'iter{step:06d}_appearance.pt',
                            module, optimizer, step, config, identity, generator,
                            frozen_sh0=model._sh0 if comparison else None)
        if step % config['validation_interval'] == 0 or step == config['steps']:
            evaluate(model, module, test_cameras, directory/'test'/f'iter{step:06d}',
                     dense=bool(comparison and comparison["dense"]))
    (directory/'memory.json').write_text(json.dumps(dict(peak_allocated_bytes=torch.cuda.max_memory_allocated())))


def render(args):
    config, base, model, module, pack, identity = setup(args)
    if args.appearance_checkpoint:
        load_checkpoint(args.appearance_checkpoint, module, identity)
    cameras = pack.get_test_cameras()
    comparison = config.get("comparison")
    if comparison:
        by_name = {cam.image_name: cam for cam in cameras}
        cameras = [by_name[name] for name in comparison["test_views"]]
    dense = bool(comparison and comparison["dense"])
    if args.benchmark:
        benchmark(model, module, cameras, args.output, dense=dense)
        return
    for baseline in ('sh0', 'sh3', None):
        evaluate(model, module, cameras, Path(args.output)/(baseline or config['mode']), baseline,
                 dense=dense)


def benchmark(model, module, cameras, output, *, dense=False):
    from .decoder import fourier_direction_3bands
    from torch.nn import functional as F
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for mode in ('sh0', 'sh3', 'vertex'):
        names = ('start','end','appearance_start','appearance_end','gather_end',
                 'raster_start','raster_end','post_end', 'preprocess_start', 'preprocess_end',
                 'selection_start')
        events = {name: torch.cuda.Event(enable_timing=True) for name in names}
        times = []
        per_camera = []
        module.to('cuda' if mode == 'vertex' else 'cpu')
        module.eval()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for i in range(60):
                cam = cameras[i % len(cameras)]
                events['start'].record()
                if mode == 'vertex':
                    vertex_stats = {}
                    model.render_vertex(cam, module, profile_events=events, vertex_stats=vertex_stats,
                                        decode_all_vertices=dense)
                else:
                    model.render(cam, color_mode=mode, profile_events=events)
                events['end'].record()
                events['end'].synchronize()
                if i >= 10:
                    row = dict(total_ms=events['start'].elapsed_time(events['end']),
                               preprocess_ms=events['preprocess_start'].elapsed_time(events['preprocess_end']),
                               raster_ms=events['raster_start'].elapsed_time(events['raster_end']),
                               postprocess_ms=events['raster_end'].elapsed_time(events['post_end']))
                    if mode == 'vertex':
                        row['selection_ms'] = events['selection_start'].elapsed_time(events['appearance_start'])
                        row['appearance_ms'] = events['appearance_start'].elapsed_time(events['appearance_end'])
                        row['gather_ms'] = events['appearance_end'].elapsed_time(events['gather_end'])
                        per_camera.append(dict(camera=i % len(cameras), **vertex_stats))
                    times.append(row)
        avg = {key: sum(row[key] for row in times)/len(times) for key in times[0]}
        avg['fps_total'] = 1000/avg['total_ms']
        avg['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        if per_camera:
            avg["per_camera"] = list({row["camera"]: row for row in per_camera}.values())
        results[mode] = avg
    # Keep a full-layout decoder microbenchmark as a separate reference.
    if isinstance(module, VertexAppearance):
        totals = {'fourier_ms': [], 'mlp_ms': []}
        with torch.no_grad():
            for i in range(60):
                cam = cameras[i % len(cameras)]
                e = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                e[0].record()
                d = F.normalize(model.grid_pts_xyz-cam.c2w[:3,3],dim=-1,eps=1e-8)
                enc = fourier_direction_3bands(d)
                e[1].record()
                result = module.decoder(torch.cat((module.vertex_features,enc),-1))
                e[2].record()
                e[2].synchronize()
                if i >= 10:
                    totals['fourier_ms'].append(e[0].elapsed_time(e[1]))
                    totals['mlp_ms'].append(e[1].elapsed_time(e[2]))
                del result, enc, d
        results['isolated_decoder'] = {k: sum(v)/len(v) for k,v in totals.items()}
    results['layout'] = dict(P=model.num_grid_pts,V=model.num_voxels,
                            vertices_per_voxel=model.num_grid_pts/model.num_voxels,
                            trainable=sum(p.numel() for p in module.parameters()),
                            parameters_bytes=sum(p.numel()*p.element_size() for p in module.parameters()),
                            vertex_rgb_bytes=12*model.num_grid_pts,
                            gather_bytes=96*model.num_voxels)
    (output/'benchmark.json').write_text(json.dumps(results,indent=2))
