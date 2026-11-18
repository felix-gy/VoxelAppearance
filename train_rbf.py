"""Frozen-geometry RBF stages: scales_only or appearance. No baseline execution."""
import argparse
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from yacs.config import CfgNode
from src.config import cfg as defaults
from src.dataloader.data_pack import DataPack
from src.sparse_voxel_gears.io import SVInOut
from src.vertex_appearance.rbf import RBFJointModel
from src.vertex_appearance.joint import training_state, restore_training
from src.vertex_appearance.checkpoint import file_digest, layout_fingerprint
from src.utils.loss_utils import l2_loss,l1_loss,huber_loss,fast_ssim_loss
from svraster_cuda.sparse_adam import SparseAdam


def load_source(path):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    config=defaults.clone()
    old=payload.get('vertex_joint')
    if old is not None:
        if old.get('schema_version') != 1:
            raise ValueError('Unsupported joint schema')
        if old.get('kernel','vertex_trilinear') not in ('vertex_trilinear','vertex_rbf_iq'):
            raise ValueError('Unsupported source kernel')
        if old.get('kernel') == 'vertex_rbf_iq' and old.get('appearance_spec') != RBFJointModel.appearance_spec:
            raise ValueError('RBF source specification mismatch')
        config.merge_from_other_cfg(CfgNode.load_cfg(old['config']))
        if not old['active']:
            raise ValueError('Import a checkpoint with learned appearance, after activation')
        module_state=old['module']
        geometry=path
    elif payload.get('schema_version')==2:
        experiment=payload['config']
        base=experiment.get('resolved_base') or yaml.safe_load(Path(experiment['base_config']).read_text())
        config.merge_from_other_cfg(CfgNode(base))
        module_state=payload['vertex_module']
        geometry=experiment['geometry_checkpoint']
        expected_identity=payload['identity']
    else:
        raise ValueError('Expected joint or frozen appearance checkpoint')
    if config.model.n_samp_per_vox!=1:
        raise ValueError('RBF requires one sample')
    model=RBFJointModel(**dict(config.model))
    SVInOut.load(model,geometry)
    if old is None:
        actual=dict(checkpoint_sha256=file_digest(geometry),layout=layout_fingerprint(model),P=model.num_grid_pts,V=model.num_voxels)
        if actual!=expected_identity:
            raise ValueError('Source checkpoint identity mismatch')
    elif old['layout']!=layout_fingerprint(model):
        raise ValueError('Source layout mismatch')
    model.activate_vertex()
    missing,unexpected=model.vertex_module.load_state_dict(module_state,strict=False)
    if unexpected or missing not in ([],['rbf_log_scales']):
        raise ValueError(f'Incompatible appearance state: {missing}, {unexpected}')
    model.vertex_module.project_scales()
    model.joint_config=config.dump()
    return model,config,old


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--scope',choices=['scales_only','appearance'],default='scales_only')
    parser.add_argument('--output',required=True)
    parser.add_argument('--steps',type=int,default=5000,help='Total additional-stage iterations')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.steps<1:parser.error('--steps must be positive')
    model,config,source=load_source(args.checkpoint)
    for name in ('_geo_grid_pts','_sh0','_shs','_subdiv_p'):
        getattr(model,name).requires_grad_(False)
    model.freeze_vox_geo()
    for parameter in model.vertex_module.parameters():
        parameter.requires_grad_(args.scope=='appearance')
    model.vertex_module.rbf_log_scales.requires_grad_(True)
    model.optimization_scope=args.scope
    parameters=[p for p in model.vertex_module.parameters() if p.requires_grad]
    optimizer=SparseAdam(parameters,lr=config.optimizer.shs_lr,
                         betas=(config.optimizer.optim_beta1,config.optimizer.optim_beta2),eps=config.optimizer.optim_eps)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,config.optimizer.lr_decay_ckpt,config.optimizer.lr_decay_mult)
    data=dict(config.data);data['use_test']=data.pop('eval');data['alpha_is_white']=config.model.white_background
    pack=DataPack(**data);cameras=pack.get_train_cameras()
    if not cameras:raise ValueError('No training cameras')
    random.seed(config.procedure.seed);np.random.seed(config.procedure.seed);torch.manual_seed(config.procedure.seed)
    order=[];rng=random.Random(config.procedure.seed)
    while len(order)<args.steps:
        epoch=list(range(len(cameras)));rng.shuffle(epoch);order.extend(epoch)
    start=0
    if args.resume:
        if source is None or source.get('kernel')!='vertex_rbf_iq' or source.get('optimization_scope')!=args.scope:
            raise ValueError('Resume requires matching RBF scope')
        state=source['training'];restore_training(state,optimizer,scheduler);start=state['iteration']
        if len(state['camera_order'])<args.steps:raise ValueError('Resume preserves original stage duration')
        order=state['camera_order']
    del source
    for step in range(start+1,args.steps+1):
        camera=cameras[order[step-1]];target=camera.image.cuda()
        optimizer.zero_grad(set_to_none=True)
        options=dict(lambda_R_concen=config.regularizer.lambda_R_concen,gt_color=target)
        image=model.render(camera,**options)['color']
        photo = (l1_loss(image,target) if config.regularizer.use_l1 else
                 huber_loss(image,target,config.regularizer.huber_thres) if config.regularizer.use_huber else
                 l2_loss(image,target))
        loss=config.regularizer.lambda_photo*photo+config.regularizer.lambda_ssim*fast_ssim_loss(image,target)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite RBF loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):raise RuntimeError('Nonfinite RBF gradient')
        optimizer.step();model.vertex_module.project_scales();scheduler.step()
        if step==1 or step%100==0:print(f'{args.scope} step={step} loss={loss.item():.8f}',flush=True)
        if step%1000==0 or step==args.steps:
            model.joint_training_state=training_state(optimizer,scheduler,step,order,{})
            model.save(str(Path(args.output)/'checkpoints'/f'iter{step:06d}_model.pt'))
    Path(args.output).mkdir(parents=True,exist_ok=True)
    (Path(args.output)/'config.yaml').write_text(config.dump())


if __name__=='__main__':main()
