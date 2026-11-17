import pytest
import torch
from src.vertex_appearance import VertexAppearance
from src.vertex_appearance.checkpoint import save_checkpoint, load_checkpoint


def test_checkpoint_roundtrip_and_rejection(tmp_path):
    module=VertexAppearance(9)
    opt=torch.optim.Adam(module.parameters())
    generator=torch.Generator().manual_seed(42)
    identity={'P':9,'V':2,'layout':'example'}
    path=tmp_path/'appearance.pt'
    save_checkpoint(path,module,opt,3,{'mode':'vertex_trilinear_fourier3'},identity,generator)
    other=VertexAppearance(9)
    state=load_checkpoint(path,other,identity)
    assert state['step']==3
    for key,value in module.state_dict().items():
        torch.testing.assert_close(value,other.state_dict()[key])
    with pytest.raises(ValueError,match='mismatch'):
        load_checkpoint(path,other,{'P':10})
