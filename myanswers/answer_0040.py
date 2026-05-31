import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def clasificador_balanceado(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = RandomForestClassifier(class_weight="balanced", random_state=42)

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    f1 = f1_score(y_test, y_pred)

    return float(f1)
