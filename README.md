# predection_mortalite_MIMICIII
L’apprentissage profond surMIMIC-III :Prédiction de la mortalité sous 24 H


Ce projet décrit la fouille de donéée sur la base  MIMIC-III .
L'objectif est de prédire le décès à l'hôpital sur la base MIMIC III.
On va suivre dans ce projet le processus KDD qui est :
    • Sélection et extraction d'un ensemble de données de séries chronologiques multivariées à partir d'une base de données de rangées de millons en écrivant des requêtes SQL.
    • Prétraité et nettoyé la série chronologique en un ensemble de données bien rangé en explorant les données, en gérant les données manquantes (taux de données manquantes> 50\%) et en supprimant le bruit / les valeurs aberrantes.
    • Développement d'un modèle prédictif permettant de classer les séries chronologiques biomédicales en mettant en œuvre plusieurs algorithmes tels que l'arbre de décision gradient boost et le KNN avec déformation temporelle dynamique.
    • Résultat de 30\% d'augmentation du score F1 par rapport à l'indice de notation médical.
 
