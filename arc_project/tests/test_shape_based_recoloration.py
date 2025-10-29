import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analysis.detectors import detect_transformations

def test_shape_based_recoloration_true():
    # Même forme, couleurs différentes
    input_grid = [
        [0, 0, 1],
        [0, 1, 1],
        [0, 0, 0]
    ]
    output_grid = [
        [0, 0, 2],
        [0, 2, 2],
        [0, 0, 0]
    ]
    result = detect_transformations(input_grid, output_grid)
    assert result["shape_based_recoloration"] == True
    assert result["recoloration"] == True

def test_shape_based_recoloration_false_different_shape():
    # La forme change
    input_grid = [
        [0, 0, 1],
        [0, 1, 1],
        [0, 0, 0]
    ]
    output_grid = [
        [0, 0, 2],
        [0, 0, 2],
        [2, 2, 2]
    ]
    result = detect_transformations(input_grid, output_grid)
    assert result["shape_based_recoloration"] == False

def test_shape_based_recoloration_false_no_recoloration():
    # Même forme, même couleur
    input_grid = [
        [0, 0, 1],
        [0, 1, 1],
        [0, 0, 0]
    ]
    output_grid = [
        [0, 0, 1],
        [0, 1, 1],
        [0, 0, 0]
    ]
    result = detect_transformations(input_grid, output_grid)
    assert result["shape_based_recoloration"] == False