import json
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.detectors import detect_transformations
from primitives.transformations import rotate_grid, flip_grid
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

class ARCPredictor:
    def __init__(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        # Charge les classifications
        classification_path = os.path.join(base_dir, 'data', 'task_classification.json')
        with open(classification_path) as f:
            self.task_classification = json.load(f)
        
        # Charge les solutions d'entraînement
        solutions_path = os.path.join(base_dir, 'data', 'arc-agi_training_solutions.json')
        with open(solutions_path) as f:
            self.train_solutions = json.load(f)
        
        # Prépare les données d'entraînement
        self.feature_vectors = []
        self.task_ids = []
        
        for task_id, data in self.task_classification.items():
            # Vérifie et complète les caractéristiques manquantes
            features = []
            for i, trans in enumerate(self.transformation_names):
                features.append(data["transformations"].get(trans, 0))
            self.feature_vectors.append(features)
            self.task_ids.append(task_id)
        
        # Entraîne un modèle KNN
        self.knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
        self.knn.fit(self.feature_vectors)

    @property
    def transformation_names(self):
        """Retourne la liste ordonnées des noms de transformations"""
        return [
            'shape_based_recoloration',
            'recoloration',
            'padding',
            'translation',
            'pattern_vertical_stripes',
            'pattern_horizontal_stripes',
            'rotation_180',
            'repetition',
            'tiled_subgrid',
            'pattern_uniform_blocks',
            'rotation_90',
            'rotation_270',
            'flip_horizontal',
            'flip_vertical'
        ]
    
    def analyze_input(self, input_grid):
        """Crée un vecteur de caractéristiques pour l'input"""
        # Convertir en array numpy
        arr = np.array(input_grid)
        
        # Caractéristiques de base
        features = [
            arr.shape[0], 
            arr.shape[1],
            len(np.unique(arr)),
            np.mean(arr != 0)  # ratio de remplissage
        ]
        
        # Ajoute des caractéristiques basées sur la détection de motifs
        from analysis.convolution_utils import (
            detect_checkerboard_pattern,
            detect_vertical_stripes,
            detect_horizontal_stripes,
            detect_large_uniform_blocks
        )
        
        # Détecte différents motifs
        features.append(int(detect_checkerboard_pattern(arr)))
        features.append(int(detect_vertical_stripes(arr)))
        features.append(int(detect_horizontal_stripes(arr)))
        features.append(int(detect_large_uniform_blocks(arr)))
        
        # Ajoute des caractéristiques de transformation (initialisées à 0)
        features += [0] * (len(self.transformation_names) - 8)
        
        return features
    
    def predict(self, test_task_id, test_input):
        # Analyse le test input
        test_features = self.analyze_input(test_input)
        
        # Trouve les tâches similaires
        similar_tasks = self.find_similar_tasks(test_features)
        
        # Essaye les transformations des tâches similaires
        for task_id in similar_tasks:
            solution = self.apply_similar_transformation(
                task_id, 
                test_input
            )
            if solution and self.validate_solution(test_input, solution):
                return solution
        
        # Fallback: transformations génériques
        return self.apply_generic_transformations(test_input)
    
    def find_similar_tasks(self, test_features):
        """Trouve les 5 tâches d'entraînement les plus similaires"""
        if len(test_features) != len(self.feature_vectors[0]):
            test_features = test_features[:len(self.feature_vectors[0])]
            test_features += [0] * (len(self.feature_vectors[0]) - len(test_features))
        
        # Trouve les plus proches voisins
        distances, indices = self.knn.kneighbors([test_features])
        
        # Récupère les IDs des tâches similaires
        return [self.task_ids[i] for i in indices[0]]
    
    def apply_similar_transformation(self, similar_task_id, test_input):
        """Applique les transformations d'une tâche similaire"""
        # Récupérer la solution d'entraînement
        if similar_task_id in self.train_solutions:
            solution = self.train_solutions[similar_task_id][0]  # Première solution
            return solution
        
        return None
    
    def validate_solution(self, input_grid, output_grid):
        """Validation de base d'une solution"""
        if not output_grid or not all(isinstance(row, list) for row in output_grid):
            return False
        
        if len(output_grid) == 0 or len(output_grid[0]) == 0:
            return False
            
        return True
    
    def apply_generic_transformations(self, input_grid):
        """Transformations génériques de dernier recours"""
        # En essayant les transformations les plus courantes
        # 1. Rotation à 90 degrés
        try:
            rotated = rotate_grid(input_grid, 90)
            return rotated
        except:
            pass
        
        # 2. Répétition 2x2
        try:
            arr = np.array(input_grid)
            repeated = np.tile(arr, (2, 2)).tolist()
            return repeated
        except:
            pass
        
        # 3. Retourner horizontalement
        try:
            flipped = flip_grid(input_grid, 'horizontal')
            return flipped
        except:
            pass
        
        # Fallback: retourner l'input inchangé
        return input_grid

if __name__ == "__main__":
    # Initialise le prédicteur
    predictor = ARCPredictor()
    
    # Chemin vers les données de test
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    test_path = os.path.join(base_dir, 'data', 'arc-agi_test_challenges.json')
    
    # Charge les tâches de test
    with open(test_path) as f:
        test_challenges = json.load(f)
    
    # Prédit les solutions
    solutions = {}
    for task_id, task_data in test_challenges.items():
        test_input = task_data["test"][0]["input"]
        solutions[task_id] = predictor.predict(task_id, test_input)
    
    # Sauvegarde
    save_path = os.path.join(base_dir, 'data', 'predicted_solutions.json')
    with open(save_path, "w") as f:
        json.dump(solutions, f, indent=2)
    

    print(f"Solutions prédites pour {len(solutions)} tâches")
