# Breast_cancer_data
# 🩺 Prédiction du Cancer du Sein (Breast Cancer Wisconsin Diagnostic)

Ce projet implémente un modèle de Deep Learning basé sur un réseau de neurones artificiels (ANN) pour classifier les tumeurs du sein en **bénignes** ou **malignes** à partir du célèbre jeu de données **Breast Cancer Wisconsin (Diagnostic)**.

---

## 📋 Table des Matières
- [Aperçu du Projet](#aperçu-du-projet)
- [Architecture du Modèle](#architecture-du-modèle)
- [Prétraitement des Données](#prétraitement-des-données)
- [Structure du Projet](#structure-du-projet)
- [Installation et Utilisation](#installation-et-utilisation)

---

## 🔍 Aperçu du Projet
Le jeu de données contient 30 caractéristiques numériques calculées à partir d'images numérisées d'une ponction à l'aiguille fine (FNA) d'une masse mammaire (ex. : `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, etc.). L'objectif est de prédire avec précision la nature de la tumeur (bénigne ou maligne).

---

## 🧠 Architecture du Modèle
Le modèle est un réseau de neurones séquentiel (`Sequential`) développé avec **Keras / TensorFlow** :
* **Couche d'entrée :** 30 caractéristiques d'entrée (`input_shape: [30]`)
* **1ère couche cachée :** 16 neurones, fonction d'activation **ReLU**
* **2ème couche cachée :** 8 neurones, fonction d'activation **ReLU**
* **Couche de sortie :** 1 neurone, fonction d'activation **Sigmoid** (classification binaire)
* **Compilateur :** Optimiseur `Adam` (taux d'apprentissage : 0.001), fonction de perte `binary_crossentropy` avec suivi de l'`accuracy`.

---

## ⚙️ Prétraitement des Données
Un objet `StandardScaler` de **Scikit-Learn** est utilisé pour normaliser les données d'entrée (centrage et réduction : moyenne et écart-type) afin d'assurer une convergence optimale et rapide lors de l'entraînement du réseau de neurones.

---

## 📂 Structure du Projet
```text
├── Breast_cancer_data.ipynb
├── breast_cancer_model.keras          # Modèle Keras entraîné (architecture et poids)
├── scaler.joblib                      # Objet StandardScaler sauvegardé (joblib)
└── README.md                          # Documentation du projet
