# ING4_PROJET2_PRAT  
# Bayes Point Machine et Classification d'Images

📌 Introduction
La **Bayes Point Machine (BPM)** est une méthode d’apprentissage automatique qui se distingue du **Perceptron** et du **modèle Bayésien naïf**. Contrairement à ces modèles, la BPM est une approche probabiliste avancée qui vise à trouver un équilibre optimal entre plusieurs séparateurs linéaires, plutôt que d’en choisir un seul. Cette approche offre une meilleure robustesse face aux variations des données et évite les erreurs de sur-apprentissage (**overfitting**). De plus, elle fonctionne efficacement même avec peu de données d’entraînement.

📖 Tutoriel : Implémentation d’un Classificateur Bayes Point Machine avec Infer.NET
Nous avons utilisé **Infer.NET**, une bibliothèque développée par **Microsoft**, pour implémenter un modèle **BPM** appliqué à la classification d’images.

🔹 Principe de fonctionnement
L’utilisateur est présenté avec plusieurs images qu’il peut classer en **bonnes (vert)** ou **mauvaises (rouge)**. À chaque étiquette attribuée, le modèle est mis à jour pour ajuster la classification des autres images non étiquetées. L'algorithme classe progressivement les images en fonction des similarités détectées :
- **Couleurs dominantes**
- **Textures**
- **Formes**

Le modèle extrait des **vecteurs de caractéristiques** des images, combinant des **descripteurs visuels** (ex: couleurs, textures) et des **attributs binaires** indiquant la présence de motifs spécifiques.

⚖️ Comparaison des Techniques : Bayes Point Machine vs. Gaussian Naïve Bayes
Nous avons testé deux approches distinctes pour classifier les images :

📌 Gaussian Naïve Bayes (GNB) :
✔ Fonctionne bien lorsque les données suivent une **distribution normale** (ex: classification des saisons : neige en hiver, feuilles en automne...).  
✔ Se base sur les **moyennes des couleurs** et leur répartition dans l’image.

📌 Bayes Point Machine (SGDClassifier) :
✔ Plus efficace pour les images où **les motifs et textures** sont importants (ex: classification des fruits).  
✔ Exploite les relations entre pixels pour mieux reconnaître des **formes complexes**.

👉 Résultat** :
- **GNB** est plus efficace pour distinguer les **saisons**.
- **BPM** est plus performant pour différencier des objets précis, comme des **fruits**.

🚀 Exploration de nouvelles approches : Pyro et CNN
Pour améliorer encore la classification d’images, nous avons exploré deux nouvelles méthodes :

🔹 Pyro (Modèle Bayésien avancé)
✔ **Pyro** est une bibliothèque de modélisation probabiliste développée par **Uber**.  
✔ Nous avons utilisé un **modèle Bayésien hiérarchique** pour estimer la probabilité qu’une image appartienne à une classe.  
✔ Cette approche est adaptée aux **données bruitées et incertaines**, permettant une classification plus souple qu’un simple modèle binaire.

🔹 CNN (Réseaux de Neurones Convolutifs avec TensorFlow)
✔ Un **CNN** extrait automatiquement des **descripteurs visuels complexes** à partir des images.  
✔ Il est performant pour identifier les objets où **les relations spatiales entre les pixels** sont essentielles (ex: reconnaître un humain versus un objet).

🔮 Perspectives et Améliorations
🔍 1️⃣ Combinaison des modèles hybrides** :
- Fusionner les approches **BPM et CNN** pour bénéficier à la fois d’une extraction de caractéristiques avancée et d’une prise en compte probabiliste des incertitudes.

🚀 2️⃣ Apprentissage auto-adaptatif** :
- Intégrer des techniques d’**Active Learning**, permettant au modèle de demander des annotations pour les images les plus incertaines.

📊 3️⃣ Optimisation de la robustesse aux biais et au bruit** :
- Améliorer la robustesse aux variations des images (bruit, angles, luminosité) via **data augmentation** et **transfer learning**.

🤖 4️⃣ Application à d’autres domaines** :
- Extension du modèle à la reconnaissance **vidéo** ou à l’analyse d’images médicales.
- Application en **finance** pour classifier des graphiques boursiers et détecter des anomalies.

🌍 5️⃣ Intégration d’une dimension éthique et explicable** :
- Rendre les décisions du modèle plus **explicables** grâce à des outils comme **SHAP** ou **LIME**.

---

📌 Conclusion
✅ **Bayes Point Machine et Gaussian Naïve Bayes** sont des solutions rapides et peu gourmandes en ressources.  
✅ **Pyro** offre une approche **probabiliste plus flexible**, idéale en présence d’incertitude.  
✅ **CNN** est la meilleure option avec un grand volume de données, permettant une classification précise.

📢 Le choix d’un modèle dépend du **type de données** et du **problème à résoudre**. Une approche hybride permet souvent d’obtenir les meilleurs résultats ! 🚀


Bienvenue sur le Git du projet **ING4_PROJET2_PRAT** 🎉  

## 🚀 Comment cloner ce dépôt ?  
Pour cloner ce projet sur votre machine, utilisez la commande suivante :  

```sh
git clone https://github.com/Corentin1109/ING4_PROJET2_PRAT.git

'
