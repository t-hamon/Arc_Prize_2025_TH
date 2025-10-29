import numpy as np
from primitives.transformations import (
    rotate_grid, transpose_grid, flip_grid, recolor_grid
)

def detect_padding(input_arr, output_arr):
    for top in range(output_arr.shape[0] - input_arr.shape[0] + 1):
        for left in range(output_arr.shape[1] - input_arr.shape[1] + 1):
            if np.array_equal(output_arr[top:top+input_arr.shape[0], left:left+input_arr.shape[1]], input_arr):
                return True
    return False

def detect_translation(input_arr, output_arr):
    return detect_padding(input_arr, output_arr) # Approche équivalente

def detect_shape_based_recoloration(input_grid, output_grid):
    import numpy as np
    from collections import defaultdict

    input_grid = np.array(input_grid)
    output_grid = np.array(output_grid)

    if input_grid.shape != output_grid.shape:
        return False

    def extract_binary_shapes(grid):
        shapes = []
        visited = np.zeros_like(grid, dtype=bool)
        h, w = grid.shape

        def dfs(i, j, color, coords):
            if i < 0 or j < 0 or i >= h or j >= w:
                return
            if visited[i, j] or grid[i, j] != color:
                return
            visited[i, j] = True
            coords.append((i, j))
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                dfs(i+di, j+dj, color, coords)

        for i in range(h):
            for j in range(w):
                if grid[i, j] != 0 and not visited[i, j]:
                    coords = []
                    dfs(i, j, grid[i, j], coords)
                    if coords:
                        rows, cols = zip(*coords)
                        min_r, max_r = min(rows), max(rows)
                        min_c, max_c = min(cols), max(cols)
                        shape = grid[min_r:max_r+1, min_c:max_c+1]
                        binary = (shape == grid[i, j]).astype(int)
                        shapes.append(binary)
        return shapes

    input_shapes = extract_binary_shapes(input_grid)
    output_shapes = extract_binary_shapes(output_grid)

    # Compare chaque forme input avec toutes les formes output
    for in_shape in input_shapes:
        for out_shape in output_shapes:
            if in_shape.shape == out_shape.shape and np.array_equal(in_shape, out_shape):
                return True  # Une forme identique a changé de couleur

    return False



def detect_transformations(input_grid, output_grid):
    input_arr = np.array(input_grid)
    output_arr = np.array(output_grid)
    info = {}

    # Shape
    info["input_shape"] = input_arr.shape
    info["output_shape"] = output_arr.shape

    # Colors
    info["input_colors"] = np.unique(input_arr).tolist()
    info["output_colors"] = np.unique(output_arr).tolist()

    # Recoloration
    info["recoloration"] = sorted(info["input_colors"]) != sorted(info["output_colors"])

    # Rotation
    info["rotation_90"] = np.array_equal(np.rot90(input_arr, 1), output_arr)
    info["rotation_180"] = np.array_equal(np.rot90(input_arr, 2), output_arr)
    info["rotation_270"] = np.array_equal(np.rot90(input_arr, 3), output_arr)

    # Flip
    info["flip_horizontal"] = np.array_equal(np.fliplr(input_arr), output_arr)
    info["flip_vertical"] = np.array_equal(np.flipud(input_arr), output_arr)

    # Transpose
    info["transpose"] = np.array_equal(input_arr.T, output_arr)

    # Répétition
    rep_x = output_arr.shape[0] // input_arr.shape[0]
    rep_y = output_arr.shape[1] // input_arr.shape[1]
    if (output_arr.shape[0] % input_arr.shape[0] == 0 and output_arr.shape[1] % input_arr.shape[1] == 0):
        repeated = np.tile(input_arr, (rep_x, rep_y))
        info["repetition"] = np.array_equal(repeated, output_arr)
    else:
        info["repetition"] = False

    # Padding et translation
    info["padding"] = detect_padding(input_arr, output_arr)
    info["translation"] = detect_translation(input_arr, output_arr)

    def match_tiled_subgrid(input_arr, output_arr):
        """Vérifie si le output est formé par la répétition de blocs de l'input"""
        ih, iw = input_arr.shape
        oh, ow = output_arr.shape
        if oh % ih != 0 or ow % iw != 0:
            return False
        tile = np.tile(input_arr, (oh // ih, ow // iw))
        return np.array_equal(tile, output_arr)
    
    info["tiled_subgrid"] = match_tiled_subgrid(input_arr, output_arr)

    def has_partial_symetry(arr):
        """Détecte si l'Input est symétrique par quart ou par bande"""
        h, w = arr.shape
        if h % 2 == 0:
            top, bottom = arr[:h//2], arr[h//2:]
            if np.array_equal(top, np.flipud(bottom)):
                return "horizontal_half"
        if w % 2 == 0:
            left, right = arr[:, :w//2], arr[:, w//2:]
            if np.array_equal(left, np.fliplr(right)):
                return "vertical_half"
        return None
    
    info["partial_symetry"] = has_partial_symetry(input_arr)

    # Enrichissement par convolution
    from analysis.convolution_utils import (
        detect_checkerboard_pattern,
        detect_horizontal_stripes,
        detect_vertical_stripes,
        detect_large_uniform_blocks
    )

    info["pattern_checkerboard"] = detect_checkerboard_pattern(input_arr)
    info["pattern_horizontal_stripes"] = detect_horizontal_stripes(input_arr)
    info["pattern_vertical_stripes"] = detect_vertical_stripes(input_arr)
    info["pattern_uniform_blocks"] = detect_large_uniform_blocks(input_arr)
    info["shape_based_recoloration"] = detect_shape_based_recoloration(input_grid, output_grid)

    return info

    def detect_compositional_transformations(input_grid, output_grid):
        """Détecte les transformations composées"""
        results = {}

        # 1. Vérifie les transformations par blocs
        if output_grid.shape[0] % input_grid.shape[0] == 0 and output_grid.shape[1] % input_grid.shape[1] == 0:
            block_h = output_grid.shape[0] // input_grid.shape[0]
            block_w = output_grid.shape[1] // input_grid.shape[1]

            # Vérification de chaque bloc
            valid = True
            for i in range(block_h):
                for j in range(block_w):
                    block = output_grid[i*input_grid.shape[0]:(i+1)*input_grid.shape[0],
                                        j*input_grid.shape[1]:(j+1)*input_grid.shape[1]]
                    if not np.array_equal(block, input_grid):
                        valid = False
                        break
                if not valid:
                    break
            results['block_repetition'] = valid

        # 2. Vérifie les transformations par quadrant
        # implémentation à compléter

        return results
    
    # Mise à jour de la fonction principale
    def detect_transformations(input_grid, output_grid):
        # ... Détections existantes

        # Ajouter les détections composites
        composite = detect_compositional_transformations(np.array(input_grid), np.array(output_grid))
        info.update(composite)

        return info