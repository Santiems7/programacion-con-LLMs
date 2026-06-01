import numpy as np
from sklearn.impute import SimpleImputer
import random


def imputar_mediana_y_devolver(X):
    imputer = SimpleImputer(strategy="median")
    return imputer.fit_transform(X)
