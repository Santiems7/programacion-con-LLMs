import numpy as np
from sklearn.decomposition import TruncatedSVD


def calcular_error_reconstruccion_svd(X, n_componentes):
    svd = TruncatedSVD(n_components=n_componentes, random_state=42)
    X_reducido = svd.fit_transform(X)

    X_reconstruido = svd.inverse_transform(X_reducido)  # (n_usuarios, n_items)

    error_por_fila = np.mean((X - X_reconstruido) ** 2, axis=1)

    return error_por_fila
