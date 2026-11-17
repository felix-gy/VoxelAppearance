import torch


def weights(u):
    bits=torch.tensor([[k>>2,(k>>1)&1,k&1] for k in range(8)],dtype=u.dtype)
    return torch.where(bits.bool(),u[...,None,:],1-u[...,None,:]).prod(-1)


def test_partition_corners_center_and_gradients():
    bits=torch.tensor([[k>>2,(k>>1)&1,k&1] for k in range(8)],dtype=torch.float64)
    torch.testing.assert_close(weights(bits),torch.eye(8,dtype=torch.float64))
    torch.testing.assert_close(weights(torch.full((1,3),0.5)),torch.full((1,8),1/8))
    u=torch.rand(10,3,dtype=torch.float64)
    w=weights(u)
    torch.testing.assert_close(w.sum(-1),torch.ones(10,dtype=torch.float64))
    rgb=torch.randn(8,3,dtype=torch.float64,requires_grad=True)
    out=w@rgb
    out.sum().backward()
    torch.testing.assert_close(rgb.grad,w.sum(0)[:,None].expand(8,3))


def test_shared_vertex_accumulation():
    values=torch.randn(5,3,requires_grad=True)
    keys=torch.tensor([[0,1,2],[0,3,4]])
    values[keys].sum().backward()
    torch.testing.assert_close(values.grad[0],torch.full((3,),2.))
