import numpy as np
from primitives.transformations import rotate_grid, flip_grid, recolor_grid

class ARCPredictor:
    def __init__(self, transformations_db):
        self.transformations_db = transformations_db
        self.common_transformations = self.analyze_common_patterns()

    def analyze_common_patterns(self):
        """Analyse les transformations les plus fréquentes"""
        # Implémente une analyse statistique des transformations
        return {
            'rotation': 0.35,
            'repetition': 0.28,
            'recoloration': 0.22,
            'reflection': 0.15
        }
    
    def predict(self, task_id, input_grid):
        """Prédit la sortie pour une tâche donnée"""
        # 1. Vérification si la tâche existe dans la base de données
        if task_id in self.transformations_db:
            return self.apply_known_transformations(task_id, input_grid)
        
        # 2. Application des transformations génériques
        return self.apply_generic_transformations(input_grid)
    
    def apply_known_transformations(self, task_id, input_grid):
        """Application des transformations spécifiques à la tâche"""
        # Implémentation basée sur detected_transformations.json
        pass

    def apply_generic_transformations(self, input_grid):
        """Essaye des transformations"""
        transformations = [
            ('rotation_90', lambda g: rotate_grid(g, 90)),
            ('repetition_2x2', lambda g: np.tile(g, (2, 2)).tolist()),
            ('rotation_180', lambda g: rotate_grid(g, 180)),
            ('recoloration', lambda g: recolor_grid(g, 'red'))
            ('flip', lambda g: flip_grid(g))
            # Ajouter d'autres transformations génériques ici
        ]

        for name, transform in transformations:
            try:
                result = transform(input_grid)
                # Valide si le résultat a du sens
                if self.is_valid_result(input_grid, result):
                    return result
            except:
                continue

        return None
    
    def validate_result(self, input_grid, output_grid):
        """Validation heuristique des résultats"""
        # 1. Taille cohérente
        if len(output_grid) < len(input_grid) or len(output_grid[0]) < len(input_grid[0]):
            return False
        
        # 2. Conservation des couleurs
        input_colors = set(np.unique(input_grid))
        output_colors = set(np.unique(output_grid))
        if not output_colors.issubset(input_colors | {0}):
            return False
        

        return True
