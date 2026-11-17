"""Fixed-layout experiment lifecycle shared by training and rendering."""
import argparse
import random
import json
from pathlib import Path
import torch
import yaml
from PIL import Image
from .decoder import VertexAppearance, VertexRGB
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
    p.add_argument('--mode', choices=['vertex_trilinear_fourier3', 'vertex_rgb_trilinear'])
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
    if config['mode'] not in ('vertex_trilinear_fourier3', 'vertex_rgb_trilinear'):
        raise ValueError('Unsupported appearance mode')
    if config['steps'] < 1 or min(config['validation_interval'], config['checkpoint_interval']) < 1:
        raise ValueError('Steps and intervals must be positive')
    expected_appearance = dict(feature_dim=8, frequencies_pi=[1.0,2.0,4.0],
                               fourier_order='frequency_sin_xyz_cos_xyz',
                               include_raw_direction=False, direction='camera_to_vertex_world',
                               layers=[26,64,3], dtype='float32')
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
    module = (VertexAppearance if config['mode'] == 'vertex_trilinear_fourier3' else VertexRGB)(model.num_grid_pts).cuda()
    data = dict(base['data'])
    data['use_test'] = data.pop('eval')
    data['alpha_is_white'] = base['model']['white_background']
    pack = DataPack(**data)
    identity = dict(checkpoint_sha256=file_digest(config['geometry_checkpoint']),
                    layout=layout_fingerprint(model), P=model.num_grid_pts, V=model.num_voxels)
    return config, base, model, module, pack, identity


def evaluate(model, module, cameras, directory, baseline=None):
    from src.utils.loss_utils import psnr_score, ssim_score, lpips_loss, correct_lpips_loss
    directory = Path(directory)
    (directory / 'renders').mkdir(parents=True, exist_ok=True)
    (directory / 'gt').mkdir(exist_ok=True)
    rows = []
    module.eval()
    with torch.no_grad():
        for i, cam in enumerate(cameras):
            image = (model.render(cam, color_mode=baseline) if baseline else model.render_vertex(cam, module))['color']
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


def train(args):
    from src.utils.loss_utils import l2_loss, fast_ssim_loss
    config, base, model, module, pack, identity = setup(args)
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'config.yaml').write_text(yaml.safe_dump(config))
    groups = ([dict(params=[module.vertex_features], lr=config['lr_vertex_features']),
               dict(params=module.decoder.parameters(), lr=config['lr_mlp'])]
              if isinstance(module, VertexAppearance) else
              [dict(params=module.parameters(), lr=config['lr_vertex_features'])])
    optimizer = torch.optim.Adam(groups, betas=(0.9,0.99), eps=1e-8)
    generator = torch.Generator().manual_seed(config['seed'])
    start = 0
    if args.resume:
        start = load_checkpoint(args.resume, module, identity, optimizer, generator)['step']
    cameras = pack.get_train_cameras()
    # Match the baseline: shuffled complete epochs, never sample with replacement.
    camera_order = []
    camera_random = random.Random(config['seed'])
    while len(camera_order) < config['steps']:
        epoch = list(range(len(cameras)))
        camera_random.shuffle(epoch)
        camera_order.extend(epoch)
    torch.cuda.reset_peak_memory_stats()
    for step in range(start+1, config['steps']+1):
        module.train()
        cam = cameras[camera_order[step-1]]
        optimizer.zero_grad(set_to_none=True)
        image = model.render_vertex(cam, module)['color']
        gt = cam.image.cuda()
        loss = l2_loss(image, gt) + base['regularizer']['lambda_ssim'] * fast_ssim_loss(image, gt)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite photometric loss")
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in module.parameters()):
            raise RuntimeError('Nonfinite appearance gradient')
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(f'step={step} loss={loss.item():.8f}', flush=True)
        if step % config['checkpoint_interval'] == 0 or step == config['steps']:
            save_checkpoint(directory/'checkpoints'/f'iter{step:06d}_appearance.pt', module, optimizer, step, config, identity, generator)
        if step % config['validation_interval'] == 0 or step == config['steps']:
            evaluate(model, module, pack.get_test_cameras(), directory/'test'/f'iter{step:06d}')
    (directory/'memory.json').write_text(json.dumps(dict(peak_allocated_bytes=torch.cuda.max_memory_allocated())))


def render(args):
    config, base, model, module, pack, identity = setup(args)
    if args.appearance_checkpoint:
        load_checkpoint(args.appearance_checkpoint, module, identity)
    if args.benchmark:
        benchmark(model, module, pack.get_test_cameras(), args.output)
        return
    for baseline in ('sh0', 'sh3', None):
        evaluate(model, module, pack.get_test_cameras(), Path(args.output)/(baseline or config['mode']), baseline)


def benchmark(model, module, cameras, output):
    from .decoder import fourier_direction_3bands
    from torch.nn import functional as F
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for mode in ('sh0', 'sh3', 'vertex'):
        names = ('start','end','appearance_start','appearance_end','gather_end',
                 'raster_start','raster_end','post_end')
        events = {name: torch.cuda.Event(enable_timing=True) for name in names}
        times = []
        module.to('cuda' if mode == 'vertex' else 'cpu')
        module.eval()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for i in range(60):
                cam = cameras[i % len(cameras)]
                events['start'].record()
                if mode == 'vertex':
                    model.render_vertex(cam, module, profile_events=events)
                else:
                    model.render(cam, color_mode=mode, profile_events=events)
                events['end'].record()
                events['end'].synchronize()
                if i >= 10:
                    row = dict(total_ms=events['start'].elapsed_time(events['end']),
                               raster_ms=events['raster_start'].elapsed_time(events['raster_end']),
                               postprocess_ms=events['raster_end'].elapsed_time(events['post_end']))
                    if mode == 'vertex':
                        row['appearance_ms'] = events['appearance_start'].elapsed_time(events['appearance_end'])
                        row['gather_ms'] = events['appearance_end'].elapsed_time(events['gather_end'])
                    times.append(row)
        avg = {key: sum(row[key] for row in times)/len(times) for key in times[0]}
        avg['fps_total'] = 1000/avg['total_ms']
        avg['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        results[mode] = avg
    # Isolated decoder stages exclude neither stage from total render latency.
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
