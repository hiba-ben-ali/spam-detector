# Détecteur de spam SMS

Petit projet de machine learning qui classe un SMS en **spam** ou **ham** (message normal).

## Données
Jeu de données *SMS Spam Collection* ([lien de la source : à ajouter]), fichier `spam.csv`.

## Méthode
- Nettoyage du texte (minuscules, URL remplacées, ponctuation supprimée)
- Vectorisation TF-IDF (unigrammes + bigrammes)
- Classifieur Naive Bayes multinomial
- Séparation 80 % entraînement / 20 % test (stratifiée)

## Résultats
- Accuracy : 98,3 %
- Spam : précision 0,99 / rappel 0,89 / F1 0,93
- Matrice de confusion :

|            | Prédit ham | Prédit spam |
|------------|-----------:|------------:|
| Réel ham   | 964        | 2           |
| Réel spam  | 17         | 132         |

Le modèle se trompe très rarement sur les messages normaux (2 faux positifs),
mais laisse passer environ 11 % des spams (17 faux négatifs).

## Utilisation
pip install -r requirements.txt
python train.py      # entraîne et sauvegarde le modèle
python predict.py    # teste le modèle en ligne de commande

## Limites
- Données en anglais uniquement : un message en français ou en dialecte
  n'est pas reconnu et retombe sur la probabilité a priori (~13 %)
- Modèle simple (sac de mots), sans prise en compte du contexte