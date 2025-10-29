from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes
import numpy as np

def detect_spatial_changes(input_grid, output_grid):
    input_bin = np.array(input_grid) != 0
    output_bin = np.array(output_grid) != 0

    # On s'assure que les deux grilles ont la même dimension par padding
    max_shape = (
        max(input_bin.shape[0], output_bin.shape[0]),
        max(input_bin.shape[1], output_bin.shape[1])
    )

    input_padded = np.zeros(max_shape, dtype=bool)
    output_padded = np.zeros(max_shape, dtype=bool)

    input_padded[:input_bin.shape[0], :input_bin.shape[1]] = input_bin
    output_padded[:output_bin.shape[0], :output_bin.shape[1]] = output_bin

    dilatation_diff = binary_dilation(input_padded) ^ output_padded
    erosion_diff = binary_erosion(input_padded) ^ output_padded
    fill_diff = binary_fill_holes(input_padded) ^ output_padded

    results = {
        "dilatation_diff_sum": int(dilatation_diff.sum()),
        "erosion_diff_sum": int(erosion_diff.sum()),
        "fill_diff_sum": int(fill_diff.sum()),
    }

    return results
