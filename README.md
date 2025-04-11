
#  PRÉDICTION DE LA CONSOMMATION D'ÉNERGIE

##  Objectif

Ce projet vise à prédire la consommation d’énergie à partir de données historiques (consommation électrique, météo, calendrier).  
Trois modèles de machine learning sont utilisés pour effectuer des prévisions de consommation à court terme (30 jours) :

-  Random Forest
-  XGBoost
-  LightGBM

##  Structure du projet

```
PRÉDICTION-CONSOMATION-ENERGIE/
│
├── data/
│   ├── 01_raw/                # Données brutes
│   ├── 02_processed/          # Données nettoyées
│   ├── 03_training_data/      # Données prêtes pour l'entraînement
│   ├── 04_visualisation/      # Résultats graphiques + fichiers CSV
│   └── 05_models/             # Modèles sauvegardés (.pkl)
│
├── notebooks/
│   ├── AED.ipynb              # Analyse exploratoire des données
│   └── ML.ipynb               # Entraînement des modèles
│
├── Forecast.py                # Script de prévision autonome
├── Makefile                   # Automatisation des tâches
├── requirements.txt           # Dépendances du projet
└── README.md                  
```

---

##  Utilisation avec `Makefile`


###  Installation des dépendances

```bash
make install
```

###  Entraînement (si tu veux relancer le notebook d’entraînement)

```bash
make train
```

###  Lancer les prévisions 30 jours

```bash
make forecast
```

###  Nettoyer les résultats générés

```bash
make clean
```

>  Les résultats seront sauvegardés dans : `data/04_visualisation/`



##  Résultats produits

| Fichier / Graphique                        | Description                                  |
|-------------------------------------------|----------------------------------------------|
| `prevision_directe_t_plus_30.csv`         | Prévision unique à J+30                      |
| `prevision_rolling_30_jours.csv`          | Prévision journalière sur 30 jours           |
| `plot_prevision_rolling_30_jours.png`     | Graphe des prévisions                        |
| `Eval_3 model.png`                         | Evaluation des 3 modèles          |
| `top 8 features for each model.png`       | les 8 top important features pour chaque modèle  |





