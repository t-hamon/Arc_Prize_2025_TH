import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sys
import os

# Ajoute le chemin pour résoudre les problèmes d'import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Charge le catalogue de transformations
with open('data/transformation_catalog.json') as f:
    catalog = json.load(f)

# Charge les transformations détectées
with open('data/detected_transformations.json') as f:
    tasks = json.load(f)

# Crée une matrice de caractéristiques
feature_names = [t["transformation"] for t in catalog]
trans_features = []
task_ids = []

for task_id, task_data in tasks.items():
    features = []
    
    # Vérifie la structure des données
    if isinstance(task_data, list):
        # Ancien format: liste d'exemples
        summary = {}
        for example in task_data:
            for trans, detected in example["transformations"].items():
                if detected:
                    summary[trans] = summary.get(trans, 0) + 1
        
        # Convertit en fréquences
        total_examples = len(task_data)
        for trans in summary:
            summary[trans] = summary[trans] / total_examples
    else:
        # Nouveau format: dictionnaire avec résumé
        summary = task_data.get("summary", {})
    
    for trans in feature_names:
        features.append(summary.get(trans, 0))
    
    trans_features.append(features)
    task_ids.append(task_id)

# Normalisation des caractéristiques
scaler = StandardScaler()
X = scaler.fit_transform(trans_features)

# Clustering avec K-Means
kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# Sauvegarde la classification
classification = {}
for task_id, cluster_id in zip(task_ids, clusters):
    classification[task_id] = {
        "cluster": int(cluster_id),
        "transformations": {trans: freq for trans, freq in zip(feature_names, trans_features[task_ids.index(task_id)])}
    }

with open("data/task_classification.json", "w") as f:
    json.dump(classification, f, indent=2)


print(f"Tâches classifiées en {kmeans.n_clusters} clusters")
