"""Only the new RBF route; no calls to the prior appearance renderer."""
import math
import pytest
import torch
from torch import nn
from src.vertex_appearance.rbf import RBFJointModel,reference_weights


def make_scene():
    from src.utils.octree_utils import gen_octpath_dense
    from src.cameras import MiniCam
    model=RBFJointModel(ss=1,white_background=True)
    model.scene_center=torch.zeros(3,device='cuda')
    model.scene_extent=torch.tensor(2.,device='cuda');model.inside_extent=model.scene_extent
    model.octpath,model.octlevel=gen_octpath_dense(0,1)
    model._geo_grid_pts=torch.full((model.num_grid_pts,1),-1.,device='cuda',requires_grad=True)
    model._sh0=torch.zeros((model.num_voxels,3),device='cuda',requires_grad=True)
    model._shs=torch.zeros((model.num_voxels,15,3),device='cuda')
    model._subdiv_p=torch.ones((model.num_voxels,1),device='cuda',requires_grad=True)
    model.activate_vertex()
    transform=torch.eye(4);transform[2,3]=-3
    camera=MiniCam(transform,math.pi/3,math.pi/3,16,16)
    return model,camera


def test_reference_identity_and_support():
    t=torch.tensor([[0.,0.,0.],[0.5,0.5,0.5],[0.1,0.8,0.3]],dtype=torch.float64)
    bits=t.new_tensor([[k>>2,(k>>1)&1,k&1] for k in range(8)])
    N=torch.where(bits.bool(),t[:,None,:],1-t[:,None,:]).prod(-1)
    eta=torch.zeros((3,8,3),dtype=t.dtype)
    torch.testing.assert_close(reference_weights(t,eta),N,atol=1e-12,rtol=0)
    w=reference_weights(t,torch.randn_like(eta))
    assert (w>=0).all() and (w[N==0]==0).all()
    torch.testing.assert_close(w.sum(-1),torch.ones(3,dtype=t.dtype))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_rbf_backward_and_geometry():
    model,camera=make_scene()
    class Colors(nn.Module):
        def __init__(self,p):
            super().__init__()
            self.rgb=nn.Parameter(torch.linspace(-0.2,0.2,p*3,device='cuda').reshape(p,3))
            self.rbf_log_scales=nn.Parameter(torch.zeros(p,3,device='cuda'))
        def forward(self,xyz,origin):return self.rgb
    model.vertex_module=Colors(model.num_grid_pts)
    opts=dict(output_depth=True,output_T=True,output_normal=True)
    result=model.render(camera,**opts)
    result['color'].square().mean().backward()
    assert model._geo_grid_pts.grad is not None and torch.isfinite(model._geo_grid_pts.grad).all()
    assert model._sh0.grad is not None and torch.isfinite(model._sh0.grad).all()
    for parameter in (model.vertex_module.rbf_log_scales,model.vertex_module.rgb):
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().max()>0
        index=divmod(parameter.grad.abs().argmax().item(),3)
        analytical=parameter.grad[index].clone();eps=1e-2
        with torch.no_grad():
            parameter[index]+=eps;plus=model.render(camera)['color'].square().mean()
            parameter[index]-=2*eps;minus=model.render(camera)['color'].square().mean()
            parameter[index]+=eps
        torch.testing.assert_close(analytical,(plus-minus)/(2*eps),atol=1e-5,rtol=0.03)
    with torch.no_grad():
        model.vertex_module.rbf_log_scales.fill_(0.7)
        changed=model.render(camera,**opts)
    for name in ('T','depth','normal'):
        torch.testing.assert_close(changed[name],result[name],atol=1e-6,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_rbf_checkpoint_and_frozen_scopes(tmp_path,monkeypatch):
    model,camera=make_scene();model.joint_config='model: {}'
    model.vertex_module.decoder[-1].weight.data.normal_(0,0.1)
    for name in ('_geo_grid_pts','_sh0','_shs','_subdiv_p'):
        getattr(model,name).requires_grad_(False)
    for scope in ('scales_only','appearance'):
        for p in model.vertex_module.parameters():p.requires_grad_(scope=='appearance')
        model.vertex_module.rbf_log_scales.requires_grad_(True)
        params=[p for p in model.vertex_module.parameters() if p.requires_grad]
        optimizer=torch.optim.Adam(params,lr=0.001)
        optimizer.zero_grad()
        model.render(camera)['color'].square().mean().backward()
        assert model.vertex_module.rbf_log_scales.grad.abs().sum()>0
        optimizer.step();model.vertex_module.project_scales()
    with torch.no_grad():expected=model.render(camera)['color']
    path=tmp_path/'rbf.pt';model.save(str(path))
    other=RBFJointModel(ss=1,white_background=True);other.load(str(path))
    with torch.no_grad():actual=other.render(camera)['color']
    torch.testing.assert_close(actual,expected,atol=1e-5,rtol=0)
    # One real CLI step per frozen mode, using only this tiny RBF fixture.
    import sys
    import train_rbf
    camera.image=torch.full((3,16,16),0.7)
    class Pack:
        def __init__(self,**kwargs):pass
        def get_train_cameras(self):return [camera]
    monkeypatch.setattr(train_rbf,'DataPack',Pack)
    for scope in ('scales_only','appearance'):
        output=tmp_path/scope
        monkeypatch.setattr(sys,'argv',['train_rbf.py','--checkpoint',str(path),'--scope',scope,'--steps','1','--output',str(output)])
        train_rbf.main()
        saved=torch.load(output/'checkpoints/iter000001_model.pt',map_location='cpu',weights_only=False)
        assert saved['vertex_joint']['optimization_scope']==scope



@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_rbf_adaptation():
    model,camera=make_scene()
    model.vertex_module.rbf_log_scales.data.uniform_(-0.5,0.5)
    old_keys=model.grid_pts_key.clone();old_eta=model.vertex_module.rbf_log_scales.detach().clone()
    mask=torch.zeros(model.num_voxels,device='cuda',dtype=torch.bool);mask[0]=True
    model.subdividing(mask)
    for key,eta in zip(model.grid_pts_key,model.vertex_module.rbf_log_scales):
        match=(old_keys==key).all(-1)
        expected=old_eta[match][0] if match.any() else torch.zeros_like(eta)
        torch.testing.assert_close(eta,expected,atol=0,rtol=0)
    mask=torch.zeros(model.num_voxels,device='cuda',dtype=torch.bool);mask[-1]=True
    model.pruning(mask)
    model.render(camera)['color'].mean().backward()
    assert model.vertex_module.rbf_log_scales.grad is not None
    assert torch.isfinite(model.vertex_module.rbf_log_scales.grad).all()

    # A tiny post-adaptation joint optimization step, still only the RBF route.
    from svraster_cuda.sparse_adam import SparseAdam
    optimizer=SparseAdam([{'params':[model._geo_grid_pts],'lr':0.025},
                         {'params':[model._sh0],'lr':0.01},
                         {'params':model.angular_parameters(),'lr':0.00025}],
                        betas=(0.1,0.99),eps=1e-15)
    optimizer.zero_grad()
    model.render(camera)['color'].square().mean().backward()
    optimizer.step();model.vertex_module.project_scales()
    assert torch.isfinite(model.vertex_module.rbf_log_scales).all()
