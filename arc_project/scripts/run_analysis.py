import sys
import os
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.arc_utils import load_arc_dataset, save_transformations
from analysis.detectors import detect_transformations
from analysis.spatial_analysis import detect_spatial_changes

def analyze_all_tasks():
    # Charger toutes les tâches d'entraînement
    train_challenges = load_arc_dataset("train_challenges")
    all_results = {}
    
    # Analyse de chaque tâche
    for task_id, task_data in tqdm(train_challenges.items(), desc='Analyse des tâches ARC'):
        task_results = {"examples": []}
        
        # Analyse de chaque paire input/output
        for example in task_data["train"]:
            analysis = detect_transformations(example["input"], example["output"])
            spatial = detect_spatial_changes(example["input"], example["output"])
            
            # Conversion des types numpy pour sérialisation JSON
            for key, value in analysis.items():
                if isinstance(value, np.ndarray):
                    analysis[key] = value.tolist()
                elif isinstance(value, np.bool_):
                    analysis[key] = bool(value)
            
            task_results["examples"].append({
                "transformations": analysis,
                "spatial_changes": spatial
            })
        
        # Analyse globale de la tâche
        task_results["summary"] = summarize_task_analysis(task_results["examples"])
        all_results[task_id] = task_results
    
    # Sauvegarde des résultats
    save_transformations(all_results, "../data/detected_transformations.json")

def summarize_task_analysis(examples):
    """Crée un résumé des transformations pour une tâche"""
    summary = {}
    transformation_counts = {}
    
    for ex in examples:
        for trans, detected in ex["transformations"].items():
            if detected:
                transformation_counts[trans] = transformation_counts.get(trans, 0) + 1
    
    # Calcul des fréquences
    total_examples = len(examples)
    for trans, count in transformation_counts.items():
        summary[trans] = count / total_examples
    
    return summary

if __name__ == "__main__":
    analyze_all_tasks()