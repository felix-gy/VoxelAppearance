import random
import numpy as np
import pytest
import torch
from test_vertex_cuda import scene
from src.vertex_appearance.joint import JointVoxelModel,training_state,restore_training

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def joint_scene():
    base,camera=scene()
    model=JointVoxelModel(ss=1,white_background=True)
    model.__dict__.update(base.__dict__)
    model.unfreeze_vox_geo()
    model._sh0.requires_grad_(True)
    model._subdiv_p.requires_grad_(True)
    return model,camera


@pytest.mark.parametrize("mode", ["vertex_mlp", "vertex_two_bases"])
def test_activation_preserves_rng_and_zero_render(mode):
    m,c=joint_scene()
    m.appearance_mode = mode
    ref=m.render(c)['color'].detach()
    rng=torch.get_rng_state().clone()
    assert m.activate_vertex()
    assert torch.equal(rng,torch.get_rng_state())
    assert not m.activate_vertex()
    torch.testing.assert_close(m.render(c)['color'],ref,atol=1e-5,rtol=0)
    m.render(c)['color'].mean().backward()
    assert m._geo_grid_pts.grad.abs().sum()>0
    assert m._sh0.grad.abs().sum()>0
    assert m._subdiv_p.grad is not None
    if mode == "vertex_two_bases":
        module = m.vertex_module
        assert module.mix.grad[:, 1].abs().sum() > 0
        grad = module.decoder[-1].weight.grad
        assert grad[:3].abs().sum() > 0
        assert grad[3:].count_nonzero() == 0
        with torch.no_grad():
            module.mix.add_(module.mix.grad, alpha=-0.1)
        module.zero_grad()
        m.render(c)['color'].mean().backward()
        assert module.decoder[-1].weight.grad[3:].abs().sum() > 0


def test_joint_regularizers_match_sh0_at_zero_residual():
    original,c=joint_scene()
    joint,_=joint_scene()
    joint.activate_vertex()
    options=dict(lambda_R_concen=0.01,lambda_dist=0.1,gt_color=torch.full((3,16,16),0.3,device='cuda'))
    for m in (original,joint):
        m.render(c,**options)['color'].square().mean().backward()
    torch.testing.assert_close(joint._geo_grid_pts.grad,original._geo_grid_pts.grad,atol=1e-4,rtol=1e-3)
    torch.testing.assert_close(joint._sh0.grad,original._sh0.grad,atol=1e-4,rtol=1e-3)


def test_joint_sh0_mask_and_finite_difference():
    m,c=joint_scene()
    m.activate_vertex()
    m.vertex_module.decoder[-1].bias.data.fill_(-1)
    m.render(c)['color'].mean().backward()
    assert m._sh0.grad.count_nonzero()==0
    m.vertex_module.decoder[-1].bias.data.zero_()
    for param in (m._sh0,m._geo_grid_pts):
        param.grad=None
    m.render(c)['color'].mean().backward()
    for param,index in ((m._sh0,(0,0)),(m._geo_grid_pts,(5,0))):
        analytical=param.grad[index].clone()
        eps=1e-3
        with torch.no_grad():
            param[index]+=eps
            plus=m.render(c)['color'].mean()
            param[index]-=2*eps
            minus=m.render(c)['color'].mean()
            param[index]+=eps
        torch.testing.assert_close(analytical,(plus-minus)/(2*eps),atol=1e-4,rtol=1e-2)


@pytest.mark.parametrize("mode", ["vertex_mlp", "vertex_two_bases"])
def test_adaptation_preserves_keys_and_decoder(mode):
    m,c=joint_scene()
    m.appearance_mode = mode
    m.activate_vertex()
    keys=m.grid_pts_key.clone()
    values=m.vertex_module.vertex_features.detach().clone()
    decoder={k:v.clone() for k,v in m.vertex_module.decoder.state_dict().items()}
    mask=torch.zeros(m.num_voxels,dtype=torch.bool,device='cuda');mask[0]=True
    if mode == "vertex_two_bases":
        mix = m.vertex_module.mix
        with torch.no_grad():
            mix.copy_(torch.arange(mix.numel(), device=mix.device).view_as(mix))
        expected = torch.cat((mix[~mask], mix[mask].repeat_interleave(8, dim=0)))
    m.subdividing(mask)
    if mode == "vertex_two_bases":
        torch.testing.assert_close(m.vertex_module.mix, expected)
    for key,value in zip(keys,values):
        found=(m.grid_pts_key==key).all(1)
        if found.any():
            torch.testing.assert_close(m.vertex_module.vertex_features[found][0],value,atol=0,rtol=0)
    for k,v in decoder.items():
        torch.testing.assert_close(m.vertex_module.decoder.state_dict()[k],v,atol=0,rtol=0)
    mask=torch.zeros(m.num_voxels,dtype=torch.bool,device='cuda');mask[-1]=True
    m.pruning(mask)
    if mode == "vertex_two_bases":
        torch.testing.assert_close(m.vertex_module.mix, expected[~mask])
    assert m.vertex_module.vertex_features.shape==(m.num_grid_pts,8)
    assert torch.isfinite(m.render(c)['color']).all()


def test_joint_checkpoint_and_rng_resume(tmp_path):
    m,c=joint_scene();m.activate_vertex();m.joint_config='model: {}'
    opt=torch.optim.Adam(m.vertex_module.parameters())
    sched=torch.optim.lr_scheduler.MultiStepLR(opt,[10])
    m.render(c)['color'].mean().backward();opt.step();sched.step()
    m.joint_training_state=training_state(opt,sched,1000,[0,1],{'ss':1})
    expected_random=(random.random(),np.random.rand(),torch.rand(1))
    expected=m.render(c)['color'].detach()
    p=tmp_path/'model.pt';m.save(str(p))
    n=JointVoxelModel(ss=1,white_background=True);n.load(str(p))
    other=torch.optim.Adam(n.vertex_module.parameters());other_sched=torch.optim.lr_scheduler.MultiStepLR(other,[10])
    restore_training(n.joint_training_state,other,other_sched)
    assert random.random()==expected_random[0]
    assert np.random.rand()==expected_random[1]
    torch.testing.assert_close(torch.rand(1),expected_random[2],atol=0,rtol=0)
    torch.testing.assert_close(n.render(c)['color'],expected,atol=1e-5,rtol=0)
    assert n.joint_training_state['iteration']==1000


@pytest.mark.parametrize("mode", ["vertex_mlp", "vertex_two_bases"])
def test_real_trainer_boundary_adaptation_and_resume(mode, tmp_path, monkeypatch):
    import train
    from argparse import Namespace
    from src.config import cfg as defaults
    from src.utils.system_utils import seed_everything
    from yacs.config import CfgNode
    from src.cameras import MiniCam
    import math
    config=defaults.clone()
    config.merge_from_file('cfg/synthetic_nerf.yaml')
    # Tiny synthetic fixture only: production defaults are not changed.
    config.procedure.n_iter=1501
    config.init.init_n_level=1
    config.init.geo_init=-1.0
    config.model.ss=1.0
    c2w=torch.eye(4);c2w[2,3]=-3
    camera=MiniCam(c2w,math.pi/3,math.pi/3,16,16)
    camera.image=torch.full((3,16,16),0.7)
    class Pack:
        point_cloud=None
        suggested_bounding=None
        def __init__(self,**kwargs):pass
        def get_train_cameras(self):return [camera]
        def get_test_cameras(self):return []
    monkeypatch.setattr(train,'cfg',config)
    monkeypatch.setattr(train,'DataPack',Pack)
    monkeypatch.setattr(train,'decide_main_bounding',lambda **kwargs: torch.tensor([[-1.,-1.,-1.],[1.,1.,1.]],device='cuda'))
    monkeypatch.setattr(train,'training_report',lambda **kwargs:None)
    args=Namespace(appearance_mode=mode,load_iteration=None,model_path=str(tmp_path),
                   load_optimizer=False,save_optimizer=False,save_quantized=False,
                   checkpoint_iterations=[1,999,1000,1001],test_iterations=[],pg_view_every=0)
    seed_everything(3721)
    train.training(args)
    initial=torch.load(tmp_path/'checkpoints/iter000001_model.pt',map_location='cpu',weights_only=False)
    before=torch.load(tmp_path/'checkpoints/iter000999_model.pt',map_location='cpu',weights_only=False)
    after=torch.load(tmp_path/'checkpoints/iter001000_model.pt',map_location='cpu',weights_only=False)
    expected=torch.load(tmp_path/'checkpoints/iter001501_model.pt',map_location='cpu',weights_only=False)
    assert not before['vertex_joint']['active'] and after['vertex_joint']['active']
    assert len(after['octpath'])>len(before['octpath'])
    groups=after['vertex_joint']['training']['optimizer']['param_groups']
    assert [g['lr'] for g in groups]==[0.025,0.01,0.00025]
    assert all(g['betas']==(0.1,0.99) and g['eps']==1e-15 for g in groups)
    assert after['vertex_joint']['training']['render_options']['ss']==1.0
    expected_first=torch.load(tmp_path/'checkpoints/iter001001_model.pt',map_location='cpu',weights_only=False)
    assert 1.0 < expected_first['vertex_joint']['training']['render_options']['ss'] <= 1.5
    args.load_iteration=1000
    train.training(args)
    resumed=torch.load(tmp_path/'checkpoints/iter001501_model.pt',map_location='cpu',weights_only=False)
    resumed_first=torch.load(tmp_path/'checkpoints/iter001001_model.pt',map_location='cpu',weights_only=False)
    for key in ('_geo_grid_pts','_sh0'):
        torch.testing.assert_close(resumed_first[key],expected_first[key],atol=1e-5,rtol=1e-4)
        assert torch.isfinite(resumed[key]).all()
    for key in expected['vertex_joint']['module']:
        torch.testing.assert_close(resumed_first['vertex_joint']['module'][key],expected_first['vertex_joint']['module'][key],atol=1e-5,rtol=1e-4)

    args.load_iteration=None
    args.appearance_mode='sh'
    args.model_path=str(tmp_path/'original')
    seed_everything(3721)
    train.training(args)
    baseline=torch.load(tmp_path/'original/checkpoints/iter000999_model.pt',map_location='cpu',weights_only=False)
    baseline_initial=torch.load(tmp_path/'original/checkpoints/iter000001_model.pt',map_location='cpu',weights_only=False)
    for key in ('octpath','octlevel','_geo_grid_pts','_sh0','_shs'):
        torch.testing.assert_close(initial[key],baseline_initial[key],atol=1e-5,rtol=1e-5)
    # CUDA atomic accumulation is not bitwise deterministic over 999 updates.
    for key in ('octpath','octlevel','_shs'):
        torch.testing.assert_close(before[key],baseline[key],atol=0,rtol=0)


def test_angular_replacement_preserves_optimizer_clock():
    from svraster_cuda.sparse_adam import SparseAdam
    m,c=joint_scene()
    opt=SparseAdam([{'params':[m._geo_grid_pts],'lr':0.025},
                    {'params':[m._sh0],'lr':0.01},
                    {'params':[m._shs],'lr':0.00025}],betas=(0.1,0.99),eps=1e-15)
    opt.state[m._shs]=dict(step=999,exp_avg=torch.zeros_like(m._shs),exp_avg_sq=torch.zeros_like(m._shs))
    assert m.activate_for_optimizer(opt)
    assert m._shs not in opt.state
    assert opt.param_groups[2]['lr']==0.00025
    for parameter in m.vertex_module.parameters():
        assert opt.state[parameter]['step']==999
        assert opt.state[parameter]['exp_avg'].count_nonzero()==0
        assert opt.state[parameter]['exp_avg_sq'].count_nonzero()==0
    assert not m.activate_for_optimizer(opt)
