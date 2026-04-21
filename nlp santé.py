import pandas as pd

df_test = pd.read_csv("drugsComTest_raw.csv") # données test
df_train = pd.read_csv("drugsComTrain_raw.csv") # données d'entraînement

print(df_test.head())  # affiche les 5 premières lignes
print(df_train.head())  # affiche les 5 premières lignes

df_test = df_test.dropna()
df_train = df_train.dropna()

# Affichage des dimensions des DataFrames après suppression des valeurs manquantes

print("Dimensions du DataFrame de test après suppression des valeurs manquantes :", df_test.shape)
print("Dimensions du DataFrame d'entraînement après suppression des valeurs manquantes :", df_train.shape)

# Affichage des types de données des colonnes

df_test.describe()  # affiche les statistiques descriptives du DataFrame de test
df_test.info()

df_train.describe()  # affiche les statistiques descriptives du DataFrame d'entraînement
df_train.info()

#Visualisation des variables numériques
# Importation des librairies

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Visualisation de la variable 'rating' dans le DataFrame de test


