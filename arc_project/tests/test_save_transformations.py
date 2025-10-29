import os
import sys
import json
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.arc_utils import load_arc_file

TRANSFORMATIONS_FILE = "data/detected_transformations.json"

def test_detected_transformations_exists():
    """
    Vérifie que le fichier detected_transformations.json a bien été généré.
    """
    assert os.path.exists(TRANSFORMATIONS_FILE), f"{TRANSFORMATIONS_FILE} manquant"

def test_detected_transformations_valid_json():
    """
    Vérifie que le fichier detected_transformations.json est un JSON valide.
    """
    try:
        data = load_arc_file(TRANSFORMATIONS_FILE)
    except Exception as e:
        pytest.fail(f"Erreur lors du chargement JSON {e}")

def test_each_entry_has_expected_keys():
    """
    Vérifie que chaque tâche contient les clés attendues (task_id + au moins une
    transformation détectée).
    """
    data = load_arc_file(TRANSFORMATIONS_FILE)
    for entry in data:
        assert "task_id" in entry, f"Clé 'task_id' manquante dans {entry}"
        # Il doit y avoir au moins une transformation détectée
        transformation_keys = [k for k in entry.keys() if k != "task_id"]
        assert transformation_keys, f"Aucune transformation détectée pour la tâche {entry.get('task_id')}"