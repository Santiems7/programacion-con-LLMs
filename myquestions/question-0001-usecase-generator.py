import pandas as pd
import numpy as np
import random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def generar_caso_de_uso_procesar_observaciones_estelares():
    """
    Genera un caso de uso aleatorio para la función
    procesar_observaciones_estelares(df, columna_objetivo).

    Retorna:
        (
            input_data: dict,
            output_data: tuple
        )
    """

    seed = random.randint(1, 100000)
    np.random.seed(seed)
    random.seed(seed)

    # Cantidad de estrellas
    n_filas = random.randint(6, 15)

    # Variables científicas con escalas distintas
    df = pd.DataFrame(
        {
            "brillo_medio": np.random.uniform(100, 10000, n_filas),
            "temperatura_superficial": np.random.uniform(2500, 12000, n_filas),
            "distancia_parsec": np.random.uniform(1, 5000, n_filas),
            "metalicidad": np.random.uniform(-2.5, 0.8, n_filas),
            "radio_estelar": np.random.uniform(0.1, 50, n_filas),
        }
    )

    # Introducir NaNs aleatorios
    n_nans = random.randint(1, max(2, n_filas))
    for _ in range(n_nans):
        i = random.randint(0, n_filas - 1)
        j = random.randint(0, df.shape[1] - 1)
        df.iat[i, j] = np.nan

    # Variable objetivo
    columna_objetivo = "estrella_variable"
    df[columna_objetivo] = np.random.randint(0, 2, size=n_filas)

    # INPUT
    input_data = {"df": df.copy(), "columna_objetivo": columna_objetivo}

    # OUTPUT ESPERADO
    X = df.drop(columns=[columna_objetivo])
    y = df[columna_objetivo].to_numpy()

    imputador = SimpleImputer(strategy="mean")
    X_imputada = imputador.fit_transform(X)

    escalador = StandardScaler()
    X_escalada = escalador.fit_transform(X_imputada)

    output_data = (X_escalada, y)

    return input_data, output_data


# Ejemplo de uso
# if __name__ == "__main__":
#     entrada, salida_esperada = generar_caso_de_uso_procesar_observaciones_estelares()

#     print("=== INPUT ===")
#     print("Nombre de la función esperada: procesar_observaciones_estelares")
#     print("Argumentos:")
#     print("columna_objetivo =", entrada["columna_objetivo"])
#     print(entrada["df"])

#     print("\n=== OUTPUT ESPERADO ===")
#     X_res, y_res = salida_esperada
#     print("X procesada:")
#     print(X_res)
#     print("\ny:")
#     print(y_res)
