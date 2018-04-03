from models.chen2017.transforms import *
import numpy as np

def test_tiles():
    seg = np.concatenate((np.zeros((20,20)), np.ones((20,20))))
    img = np.ones(seg.shape +(3,))*3
    assert img.shape == (40, 20, 3)

    tiler = SegmentTiling(patch_width=10)
    patches, patch_meta = tiler(img, seg)

    print(seg)
    assert len(patches) == 2

    seg = np.concatenate((np.zeros((20, 20)), np.ones((20, 20))))
    img = np.ones(seg.shape) * 3
    
    assert img.shape == (40, 20)
    tiler = SegmentTiling(patch_width=10)
    patches, patch_meta = tiler(img, seg)

    print(seg)
    assert len(patches) == 2