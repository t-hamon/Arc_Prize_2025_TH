import numpy as np
import json
import os
from analysis.pattern_detectors import detect_object_transformations
from primitives.transformations import apply_transformations

class ARCSolver:
    def __init__(self, transformations_db):
        self.transformations_db = transformations_db
    
    def solve_task(self, train_pairs, test_input):
        # Apprendre les transformations à partir des exemples d'entraînement
        learned_transformations = []
        
        for input_grid, output_grid in train_pairs:
            transformations = detect_object_transformations(input_grid, output_grid)
            learned_transformations.extend(transformations)
        
        # Appliquer les transformations apprises à l'entrée de test
        return apply_transformations(test_input, learned_transformations)

# Charger les données
with open('../data/arc-agi_training_challenges.json') as f:
    train_challenges = json.load(f)

with open('../data/arc-agi_test_challenges.json') as f:
    test_challenges = json.load(f)

# Préparer les solutions
solutions = {}

for task_id, task_data in test_challenges.items():
    # Récupérer les exemples d'entraînement pour cette tâche
    train_examples = train_challenges.get(task_id, {}).get("train", [])
    train_pairs = [(ex['input'], ex['output']) for ex in train_examples]
    
    test_input = task_data["test"][0]["input"]
    
    # Initialiser le solveur
    solver = ARCSolver({})
    predicted = solver.solve_task(train_pairs, test_input)
    
    solutions[task_id] = predicted

# Sauvegarder les solutions
with open("../data/improved_solutions.json", "w") as f:
    json.dump(solutions, f, indent=2)