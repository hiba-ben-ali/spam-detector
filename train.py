from utils import nettoyer
import joblib
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix

#1.load data 
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]
df["y"] = (df["label"] == "spam").astype(int)
print("Dataset size : ",  df.shape)
print(df["label"].value_counts())

#2. Nettoyage du texte
df["clean"] = df["text"].apply(nettoyer)

#3.Split
X_train ,X_test ,Y_train ,Y_test =train_test_split(df["clean"],df["y"], test_size=0.2 ,random_state=42,stratify=df["y"])

#4.Model pipeline 
model =Pipeline([("tfidf" , TfidfVectorizer(stop_words="english" , ngram_range=(1 ,2), min_df=2)),("nb" , MultinomialNB(alpha=0.1)),]) 
model.fit(X_train ,Y_train)

#5.Evaluate
pred = model.predict(X_test)
print("\nAccuracy:", round(accuracy_score(Y_test, pred), 4))
print(classification_report(Y_test, pred, target_names=["ham", "spam"]))
print("Confusion matrix:\n", confusion_matrix(Y_test, pred))

#6. Save
joblib.dump(model, "spam_model.joblib")
print("Model Saved as spam_model.joblib")