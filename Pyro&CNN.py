# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import pyro
import pyro.distributions as dist
import torch
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

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

# Modèle Bayésien avec Pyro
def model_pyro(features):
    weight = pyro.sample("weight", dist.Normal(torch.zeros(features.shape[1]), torch.ones(features.shape[1])))
    bias = pyro.sample("bias", dist.Normal(torch.tensor(0.0), torch.tensor(1.0)))
    logits = torch.matmul(features, weight) + bias
    return pyro.sample("output", dist.Categorical(logits=logits))

# Entraînement du modèle Pyro
def train_pyro(features, labels):
    features = torch.tensor(features, dtype=torch.float)
    labels = torch.tensor([0 if l == "humain" else 1 for l in labels], dtype=torch.long)
    
    svi = pyro.infer.SVI(model=model_pyro,
                          guide=pyro.infer.autoguide.AutoDiagonalNormal(model_pyro),
                          optim=pyro.optim.Adam({"lr": 0.01}),
                          loss=pyro.infer.Trace_ELBO())
    
    for step in range(1000):
        loss = svi.step(features, labels)
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss}")
    return svi

# Fonction d'entraînement du modèle CNN
def train_cnn(train_images, train_labels, input_shape=(64, 64, 3)):
    train_images = np.array(train_images).reshape(-1, 64, 64, 3) / 255.0  # Normalisation
    label_map = {label: i for i, label in enumerate(set(train_labels))}
    train_labels = np.array([label_map[label] for label in train_labels])
    
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(label_map), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=10, verbose=1)
    return model, label_map

# Fonction de prédiction
def predict_image(clf, image_path, label_map=None):
    features = extract_features(image_path).reshape(1, -1)
    if isinstance(clf, keras.Sequential):
        features = features.reshape(-1, 64, 64, 3) / 255.0
        prediction = np.argmax(clf.predict(features), axis=1)[0]
        return [k for k, v in label_map.items() if v == prediction][0]
    else:
        return "humain" if pyro.sample("output", dist.Bernoulli(0.5)).item() == 0 else "non_humain"

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
    'humain': "C:/Users/coren/OneDrive/Documents/humain.png",  # Image d'un humain
    'non_humain': "C:/Users/coren/OneDrive/Documents/pashumain.webp"  # Image d'un objet ou animal
}

# Chargement des images et des étiquettes
features, labels = load_images_and_labels(labels_dict)

# Choisir le classificateur
model_type = "pyro"  # Options: "pyro", "cnn"

if model_type == "pyro":
    clf = train_pyro(features, labels)
    model_path = "humain_pyro_model.pkl"
elif model_type == "cnn":
    clf, label_map = train_cnn(features, labels)
    model_path = "humain_cnn_model.pkl"
    save_model(label_map, "humain_cnn_labels.pkl")

print("Modèle entraîné avec succès !")

# Sauvegarde du modèle
save_model(clf, model_path)
print("Modèle sauvegardé !")

# Chargement du modèle
loaded_clf = load_model(model_path)
print("Modèle chargé avec succès !")

# Test de prédiction sur une image de test
test_image_path = "C:/Users/coren/OneDrive/Documents/test.jpg"  # Première image test
prediction = predict_image(loaded_clf, test_image_path, label_map if model_type == "cnn" else None)
print(f"Prédiction 1 : {prediction}")

# Deuxième test de prédiction
test_image_path2 = "C:/Users/coren/OneDrive/Documents/test2.jpg"  # Deuxième image test
prediction2 = predict_image(loaded_clf, test_image_path2, label_map if model_type == "cnn" else None)
print(f"Prédiction 2 : {prediction2}")


