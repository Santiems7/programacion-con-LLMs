import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def entrenar_clasificador_libros(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = DecisionTreeClassifier()
    model.fit(X_scaled, y)

    predictions = model.predict(X_scaled)

    accuracy = accuracy_score(y, predictions)

    return accuracy, predictions
