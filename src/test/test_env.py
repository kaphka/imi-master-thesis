def test_torch_gpu():
    import torch
    torch.eye(1).cuda()