# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
import pickle

# Fonction pour extraire les caractéristiques de l'image
def extract_features(image_path, size=(64, 64)):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize(size)  # Redimensionnement de l'image
        img_data = np.array(img)
        features = img_data.flatten()  # Transformation en vecteur
    return features

# Fonction pour charger les images et leurs étiquettes
def load_images_and_labels(labels_dict):
    features = []
    labels = []
    
    for label, image_path in labels_dict.items():
        if image_path.endswith((".jpg", ".jpeg", ".png", ".webp")):
            features.append(extract_features(image_path))
            labels.append(label)
    
    return np.array(features), np.array(labels)

# Fonction d'entraînement du modèle Naïve Bayes
def train_gaussian_nb(features, labels):
    clf = GaussianNB()
    clf.fit(features, labels)
    return clf

# Fonction d'entraînement du modèle Bayes Point Machine
def train_bayes_point_machine(features, labels):
    clf = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
    clf.fit(features, labels)
    return clf

# Fonction de prédiction
def predict_image(clf, image_path):
    features = extract_features(image_path).reshape(1, -1)
    prediction = clf.predict(features)
    return prediction

# Fonction pour sauvegarder le modèle
def save_model(clf, model_path):
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)

# Fonction pour charger le modèle
def load_model(model_path):
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    return clf

# Dossiers contenant les images
labels_dict = {
    'été': "C:/Users/coren/OneDrive/Documents/ete.jpg",  # Image d'été
    'hiver': "C:/Users/coren/OneDrive/Documents/hiver.jpg",  # Image d'hiver
    'printemps': "C:/Users/coren/OneDrive/Documents/printemps.jpg",  # Image de printemps
    'automne': "C:/Users/coren/OneDrive/Documents/automne.jpg"  # Image d'automne
}

# Chargement des images et des étiquettes
features, labels = load_images_and_labels(labels_dict)

# Choisir le classificateur
use_bayes_point_machine = True # Mettre False pour utiliser GaussianNB

if use_bayes_point_machine:
    clf = train_bayes_point_machine(features, labels)
    model_path = "saison_bpm_model.pkl"
else:
    clf = train_gaussian_nb(features, labels)
    model_path = "saison_gaussian_nb_model.pkl"

print("Modèle entraîné avec succès !")

# Sauvegarde du modèle
save_model(clf, model_path)
print("Modèle sauvegardé !")

# Chargement du modèle
loaded_clf = load_model(model_path)
print("Modèle chargé avec succès !")

# Test de prédiction sur une image de test
test_image_path = "C:/Users/coren/OneDrive/Documents/imagetest.webp"  # Première image test
prediction = predict_image(loaded_clf, test_image_path)
print(f"Prédiction 1 : {prediction[0]}")

# Deuxième test de prédiction
test_image_path2 = "C:/Users/coren/OneDrive/Documents/imagetest2.jpg"  # Deuxième image test
prediction2 = predict_image(loaded_clf, test_image_path2)
print(f"Prédiction 2 : {prediction2[0]}")

