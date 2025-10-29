import numpy as np
from scipy.ndimage import label
from torch import is_same_size

def detect_object_transformations(input_grid, output_grid):
    """Détecte les transformations appliquées à des objets spécifiques"""
    input_arr = np.array(input_grid)
    output_arr = np.array(output_grid)
    
    # Identification des objets
    input_objects = find_objects(input_arr)
    output_objects = find_objects(output_arr)
    
    transformations = []
    
    # Recherche de correspondances
    for inp_obj in input_objects:
        for out_obj in output_objects:
            if is_same_size(inp_obj['shape'], out_obj['shape']):
                # Calculer le déplacement
                dx = out_obj['position'][0] - inp_obj['position'][0]
                dy = out_obj['position'][1] - inp_obj['position'][1]
                
                # Calculer le changement de couleur
                if inp_obj['color'] != out_obj['color']:
                    transformations.append(('recolor', {
                        'from': inp_obj['color'],
                        'to': out_obj['color']
                    }))
                
                if dx != 0 or dy != 0:
                    transformations.append(('translate', {'dx': dx, 'dy': dy}))
    
    return transformations

def find_objects(grid):
    """Identifie les objets dans la grille"""
    labeled, num_features = label(grid > 0)
    objects = []
    
    for i in range(1, num_features + 1):
        positions = np.argwhere(labeled == i)
        min_y, min_x = np.min(positions, axis=0)
        max_y, max_x = np.max(positions, axis=0)
        
        # Extraire la forme et la couleur
        shape = grid[min_y:max_y+1, min_x:max_x+1]
        color = grid[positions[0][0], positions[0][1]]
        
        objects.append({
            'shape': shape,
            'color': color,
            'position': (min_y, min_x)
        })
    
    return objects