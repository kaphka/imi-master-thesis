from models.chennet import ChenNet

def test_ChenNet():
    net = ChenNet()
    assert net.log_name is not None
