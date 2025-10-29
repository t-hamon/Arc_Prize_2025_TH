import numpy as np
from scipy.signal import convolve2d

def detect_checkerboard_pattern(grid):
    """Détecte un motif en damier 2x2"""
    grid = np.array(grid)
    checker = np.indices(grid.shape).sum(axis=0) % 2
    for offset in range(10): # Décalage de couleur possible
        pattern = (grid % 2) == checker
        if np.mean(pattern) > 0.9:
            return True
    return False

def detect_vertical_stripes(grid):
    """Détecte des bandes verticales régulières"""
    grid = np.array(grid)
    diff = np.diff(grid, axis=1)
    return np.all(diff[:, ::2] == 0) or np.all(diff[:, 1::2] == 0)

def detect_horizontal_stripes(grid):
    """Détecte des bandes horizontales régulières"""
    grid = np.array(grid)
    diff = np.diff(grid, axis=0)
    return np.all(diff[::2, :] == 0) or np.all(diff[1::2, :] == 0)

def detect_large_uniform_blocks(grid, block_size=(2, 2)):
    """Détecte des blocs uniformes répétés dans la grille"""
    grid = np.array(grid)
    h, w = grid.shape
    bh, bw = block_size
    if h % bh != 0 or w % bw != 0:
        return False
    
    blocks = grid.reshape(h//bh, bh, w//bw, bw).swapaxes(1, 2)
    return all((b == b[0, 0]).all() for b in blocks.reshape(-1, bh, bw))