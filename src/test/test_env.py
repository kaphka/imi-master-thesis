import torch

def test_torch_gpu():
    assert torch.cuda.is_available()
    torch.eye(1).cuda()