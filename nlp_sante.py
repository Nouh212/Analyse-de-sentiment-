# ============================================================
# MedSentiment Analyzer
# Analyse de sentiment sur des avis de médicaments
# ============================================================

# --- 1. IMPORTS ---
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import nltk

from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


# ============================================================
# 2. CHARGEMENT DES DONNÉES
# ============================================================

df_train = pd.read_csv("drugsComTrain_raw.csv").dropna()
df_test = pd.read_csv("drugsComTest_raw.csv").dropna()

print(f"Train : {df_train.shape} | Test : {df_test.shape}")


# ============================================================
# 3. NETTOYAGE DU TEXTE
# ============================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"&#039;", "'", text)        # remplacer HTML &'
    text = re.sub(r"<.*?>", "", text)           # enlever balises HTML
    text = re.sub(r"[^a-z0-9\s']", " ", text)  # garder lettres, chiffres, apostrophes
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

df_train["review_clean"] = df_train["review"].apply(clean_text)
df_test["review_clean"] = df_test["review"].apply(clean_text)

# Sauvegarde des données nettoyées
df_train.to_csv("df_train_clean.csv", index=False)
df_test.to_csv("df_test_clean.csv", index=False)


# ============================================================
# 4. CRÉATION DE LA CIBLE BINAIRE
# ============================================================

# rating >= 7 → positif (1), sinon négatif (0)
df_train["sentiment"] = df_train["rating"].apply(lambda x: 1 if x >= 7 else 0)
df_test["sentiment"] = df_test["rating"].apply(lambda x: 1 if x >= 7 else 0)

print("\nDistribution des sentiments (train) :")
print(df_train["sentiment"].value_counts())


# ============================================================
# 5. VISUALISATION EXPLORATOIRE
# ============================================================

# Distribution du rating
plt.figure(figsize=(10, 5))
sns.histplot(df_train['rating'], bins=20, kde=True)
plt.title('Distribution des ratings (train)')
plt.xlabel('Rating')
plt.ylabel('Fréquence')
plt.tight_layout()
plt.show()

# Boxplot des ratings
plt.figure(figsize=(6, 5))
sns.boxplot(y="rating", data=df_train)
plt.title("Distribution des ratings")
plt.tight_layout()
plt.show()


# ============================================================
# 6. VECTORISATION TF-IDF
# ============================================================

X_train = df_train["review_clean"]
y_train = df_train["sentiment"]
X_test = df_test["review_clean"]
y_test = df_test["sentiment"]

# Un seul vectorizer pour tout le projet
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# ============================================================
# 7. MODÈLES ET COMPARAISON
# ============================================================

# --- Logistic Regression ---
logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
logreg.fit(X_train_vec, y_train)
y_pred_logreg = logreg.predict(X_test_vec)

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))

# --- LinearSVC ---
svc = LinearSVC(class_weight='balanced', max_iter=1000)
svc.fit(X_train_vec, y_train)
y_pred_svc = svc.predict(X_test_vec)

print("\n--- LinearSVC ---")
print("Accuracy:", accuracy_score(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))


# ============================================================
# 8. VISUALISATION DES MOTS IMPORTANTS (meilleur modèle : LogReg)
# ============================================================

feature_names = vectorizer.get_feature_names_out()
coefs = logreg.coef_[0]

top_positifs_idx = np.argsort(coefs)[-15:]
top_negatifs_idx = np.argsort(coefs)[:15]

mots_positifs = [feature_names[i] for i in top_positifs_idx]
mots_negatifs = [feature_names[i] for i in top_negatifs_idx]
poids_positifs = coefs[top_positifs_idx]
poids_negatifs = coefs[top_negatifs_idx]

# Barplot positifs / négatifs
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].barh(mots_positifs, poids_positifs, color='green')
axes[0].set_title('Top 15 mots positifs')
axes[0].set_xlabel('Poids')
axes[1].barh(mots_negatifs, poids_negatifs, color='red')
axes[1].set_title('Top 15 mots négatifs')
axes[1].set_xlabel('Poids')
plt.suptitle('Mots les plus influents dans le modèle', fontsize=14)
plt.tight_layout()
plt.show()

# Wordclouds positifs / négatifs (pondérés par les poids du modèle)
wc_positifs = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
    dict(zip(mots_positifs, poids_positifs))
)
wc_negatifs = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
    dict(zip(mots_negatifs, abs(poids_negatifs)))
)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].imshow(wc_positifs, interpolation='bilinear')
axes[0].axis('off')
axes[0].set_title('Mots les plus positifs')
axes[1].imshow(wc_negatifs, interpolation='bilinear')
axes[1].axis('off')
axes[1].set_title('Mots les plus négatifs')
plt.tight_layout()
plt.show()

# Wordcloud des mots les plus fréquents dans les avis
text_positif = " ".join(df_train[df_train['sentiment'] == 1]['review_clean'])
text_negatif = " ".join(df_train[df_train['sentiment'] == 0]['review_clean'])

wc_pos = WordCloud(width=800, height=400, background_color='white').generate(text_positif)
wc_neg = WordCloud(width=800, height=400, background_color='white').generate(text_negatif)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].imshow(wc_pos, interpolation='bilinear')
axes[0].axis('off')
axes[0].set_title('Mots fréquents — Avis positifs')
axes[1].imshow(wc_neg, interpolation='bilinear')
axes[1].axis('off')
axes[1].set_title('Mots fréquents — Avis négatifs')
plt.tight_layout()
plt.show()


# ============================================================
# 9. ANALYSE DES ERREURS
# ============================================================

df_test["sentiment_predit"] = y_pred_logreg
df_test["sentiment_reel"] = y_test.values

erreurs = df_test[df_test["sentiment_predit"] != df_test["sentiment_reel"]]
faux_positifs = df_test[(df_test["sentiment_reel"] == 0) & (df_test["sentiment_predit"] == 1)]
faux_negatifs = df_test[(df_test["sentiment_reel"] == 1) & (df_test["sentiment_predit"] == 0)]

print(f"\nNombre d'erreurs : {len(erreurs)} sur {len(df_test)}")
print(f"Faux positifs : {len(faux_positifs)}")
print(f"Faux négatifs : {len(faux_negatifs)}")

print("\nExemples de faux positifs (avis négatifs mal classés) :")
for avis in faux_positifs["review"].head(3).values:
    print("-", avis[:200])
    print()

print("\nExemples de faux négatifs (avis positifs mal classés) :")
for avis in faux_negatifs["review"].head(3).values:
    print("-", avis[:200])
    print()

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred_logreg)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Négatif", "Positif"],
            yticklabels=["Négatif", "Positif"])
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.title("Matrice de confusion — LogisticRegression")
plt.tight_layout()
plt.show()


# ============================================================
# 10. TEST SUR DES AVIS INVENTÉS
# ============================================================

def predire_sentiment(avis):
    avis_vec = vectorizer.transform([avis])
    prediction = logreg.predict(avis_vec)[0]
    return "Positif 😊" if prediction == 1 else "Négatif 😡"

print("\n--- Tests sur des avis inventés ---")
print(predire_sentiment("this medication is absolutely amazing, no side effects at all"))
print(predire_sentiment("this medication is horrible, terrible side effects, i stopped taking it"))
print(predire_sentiment("it helped with my pain but the side effects were really bad"))


# ============================================================
# 11. SAUVEGARDE DU MODÈLE
# ============================================================

joblib.dump(logreg, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("\nModèle et vectorizer sauvegardés ✅")


# ============================================================
# 12. MODELE BERT AVEC PEU DE LIGNES (optionnel)
# ============================================================

from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    truncation=True,
    max_length=512
)

avis_tests = [
    "this medication is absolutely amazing, no side effects at all",
    "this medication is horrible, terrible side effects, i stopped taking it",
    "it helped with my pain but the side effects were really bad"
]

for avis in avis_tests:
    result = classifier(avis)
    print(f"Avis : {avis[:60]}")
    print(f"Résultat : {result}\n")
    
    
    
def stars_to_sentiment(label):
    stars = int(label[0])  # "5 stars" → 5
    return 1 if stars >= 4 else 0

for avis in avis_tests:
    result = classifier(avis)
    sentiment = stars_to_sentiment(result[0]['label'])
    print(f"Avis : {avis[:60]}")
    print(f"Sentiment : {'Positif 😊' if sentiment == 1 else 'Négatif 😡'}\n")
    
    

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

df_test = pd.read_csv("df_test_clean.csv").dropna()
df_test["sentiment"] = df_test["rating"].apply(lambda x: 1 if x >= 7 else 0)

# Echantillon de 1000 lignes pour aller vite
df_sample = df_test.sample(1000, random_state=42)

# Prédictions BERT
predictions = classifier(
    list(df_sample["review_clean"]),
    truncation=True,
    max_length=512,
    batch_size=32
)

# Conversion étoiles → sentiment
df_sample["bert_pred"] = [
    1 if int(p['label'][0]) >= 4 else 0
    for p in predictions
]

# Évaluation
print("Accuracy BERT :", accuracy_score(df_sample["sentiment"], df_sample["bert_pred"]))
print(classification_report(df_sample["sentiment"], df_sample["bert_pred"]))

# On prend un modèle BERT de base
# et on l'entraîne directement sur nos avis médicaux

from transformers import pipeline

classifier_medical = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=512
)

# Ce modèle prédit directement POSITIVE/NEGATIVE
predictions = classifier_medical(
    list(df_sample["review_clean"]),
    truncation=True,
    max_length=512,
    batch_size=32
)

df_sample["bert_pred2"] = [
    1 if p['label'] == 'POSITIVE' else 0
    for p in predictions
]

print("Accuracy DistilBERT :", accuracy_score(df_sample["sentiment"], df_sample["bert_pred2"]))
print(classification_report(df_sample["sentiment"], df_sample["bert_pred2"]))