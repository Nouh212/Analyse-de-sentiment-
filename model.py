import pandas as pd

df_test = pd.read_csv("drugsComTest_raw.csv") # données test

df_train = pd.read_csv("drugsComTrain_raw.csv") # données d'entraînement

df_test = df_test.dropna()
df_train = df_train.dropna()
# Commencement des data set pour la prédictions et entraîner les données
X_train = df_train["review"]
y_train = df_train["condition"]

X_test = df_test["review"]
y_test = df_test["condition"]

#Transformation des données textuelles en vecteurs numériques à l'aide de TfidfVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=1000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)



# Entraîner un modèle de classification à l'aide de RandomForestClassifier 

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=300)
model.fit(X_train_vec, y_train)

#Test du modèle sur les données de test

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test_vec)

print(accuracy_score(y_test, y_pred))
print(y_pred[:10])
print(y_test[:10])