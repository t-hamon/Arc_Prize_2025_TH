import numpy as np
from analysis.convolution_utils import *

def test_checkerboard_true():
    grid = [[0, 1], [1, 0] * 3]
    grid = np.array(grid[:6]) # (6x2) damier
    assert detect_checkerboard_pattern(grid) is True

def test_vertical_stripes():
    grid = np.tile([1, 2], (5, 5))
    assert detect_vertical_stripes(grid) is True

def test_horizontal_stripes():
    grid = np.tile([[1], [2]], (5, 5))
    assert detect_horizontal_stripes(grid) is True

def test_blocks_uniform():
    grid = np.array([[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]])
    assert detect_large_uniform_blocks(grid) is True