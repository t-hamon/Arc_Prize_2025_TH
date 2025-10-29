import json
import os
import numpy as np
from pathlib import Path

def load_arc_dataset(dataset_type: str):
    """Charge un dataset complet ARC"""
    base_dir = Path(__file__).resolve().parent
    file_map = {
        "train_challenges": "arc-agi_training_challenges.json",
        "train_solutions": "arc-agi_training_solutions.json",
        "test_challenges": "arc-agi_test_challenges.json",
        "evaluation_challenges": "arc-agi_evaluation_challenges.json",
        "evaluation_solutions": "arc-agi_evaluation_solutions.json",
    }
    
    if dataset_type not in file_map:
        raise ValueError(f"Type de dataset invalide. Options: {list(file_map.keys())}")
    
    file_path = base_dir / file_map[dataset_type]
    return load_json(file_path)

def load_json(file_path: Path):
    """Charge un fichier JSON avec gestion d'erreurs"""
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_transformations(transformations, file_path):
    """Sauvegarde les transformations détectées"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(transformations, f, indent=2)