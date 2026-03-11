import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv("../data/imdb_data.csv")

df["sentiment_num"] = df["sentiment"].map(lambda elem: 1 if elem == "positive" else 0)

x_temp, x_test,y_temp, y_test = train_test_split(df["review"], df["sentiment_num"],test_size=0.2,random_state=42,stratify=df["sentiment_num"])

x_train, x_val, y_train, y_val = train_test_split( #validation data split
    x_temp,
    y_temp,
    test_size=0.1,
    random_state=42
)

