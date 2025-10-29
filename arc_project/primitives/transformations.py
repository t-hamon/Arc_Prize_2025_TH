import numpy as np

def rotate_grid(grid, angle=90):
    """Rotate a grid (list of lists) by 90, 180, or 270 degrees clockwise."""
    assert angle in [90, 180, 270], "Angle must be 90, 180, or 270 degrees"
    arr = np.array(grid)
    k = angle // 90
    return np.rot90(arr, k).tolist()

def flip_grid(grid, axis='horizontal'):
    """Flip a grid horizontally or vertically."""
    arr = np.array(grid)
    if axis == 'horizontal':
        return np.fliplr(arr).tolist()
    elif axis == 'vertical':
        return np.flipud(arr).tolist()
    else:
        raise ValueError("Axis must be 'horizontal' or 'vertical'")

def transpose_grid(grid):
    """Transpose a square or rectangular grid."""
    arr = np.array(grid)
    return arr.T.tolist()

def recolor_grid(grid: list[list[int]], color_map: dict[int, int]) -> list[list[int]]:
    """Recolor grid values according to a mapping dictionnary.
    e.g. {0: 1, 2: 5} maps color 0 -> 1 and color 2 -> 5
    """
    arr = np.array(grid)
    recolored = np.vectorize(lambda x: color_map.get(x, x))(arr)
    return recolored.tolist()

# primitives/transformations.py (extension)
def apply_transformations(grid, transformations):
    """
    Applique une série de transformations à une grille
    transformations: liste de tuples (nom_transformation, params)
    """
    for trans_name, params in transformations:
        if trans_name == 'rotation':
            grid = rotate_grid(grid, params['angle'])
        elif trans_name == 'recoloration':
            grid = recolor_grid(grid, params['color_map'])
        elif trans_name == 'reflection':
            grid = flip_grid(grid, params['axis'])
        elif trans_name == 'scaling':
            grid = tile_grid(grid, params['factor'])
        # Ajouter d'autres transformations
    return grid

# Fonction de tuilage
def tile_grid(grid, factor):
    """Répète la grille selon un facteur d'échelle"""
    arr = np.array(grid)
    return np.tile(arr, factor).tolist()