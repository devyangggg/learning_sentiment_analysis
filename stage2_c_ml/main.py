import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


df = pd.read_csv("../data/imdb_data.csv")

df["sentiment_num"] = df["sentiment"].map(lambda elem: 1 if elem == "positive" else 0)

text = df["review"]
label = df["sentiment_num"]

x_train, x_test, y_train, y_test = train_test_split(text, label ,test_size=0.2,train_size=0.8, random_state=42)

vectorizer = CountVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

model = LogisticRegression(max_iter=1000)

model.fit(x_train_vec, y_train)

pred = model.predict(x_test_vec)

accuracy = accuracy_score(y_test, pred)

print("Accuracy: ", accuracy)

feature_names = vectorizer.get_feature_names_out()
weights = model.coef_[0]

top_positive = sorted(
    zip(feature_names, weights),
    key=lambda x: x[1],
    reverse=True
)[:10]

print(top_positive)


