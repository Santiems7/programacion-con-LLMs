import pandas as pd
import numpy as np
import random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def generar_caso_de_uso_procesar_datos_meteorologicos():
    """
    Genera un caso de uso aleatorio para la función
    procesar_datos_meteorologicos(df, target_col).
    """

    seed = random.randint(1, 100000)
    np.random.seed(seed)
    random.seed(seed)

    n_filas = random.randint(6, 12)

    df = pd.DataFrame(
        {
            "temperatura_max": np.random.uniform(20, 38, n_filas),
            "humedad": np.random.uniform(40, 100, n_filas),
            "velocidad_viento": np.random.uniform(0, 80, n_filas),
            "presion": np.random.uniform(950, 1050, n_filas),
        }
    )

    # Introducir algunos NaN aleatorios
    n_nans = random.randint(1, max(2, n_filas // 2))
    for _ in range(n_nans):
        fila = random.randint(0, n_filas - 1)
        col = random.choice(df.columns.tolist())
        df.loc[fila, col] = np.nan

    target_col = "llovera_manana"
    df[target_col] = np.random.randint(0, 2, size=n_filas)

    # INPUT
    input_data = {"df": df.copy(), "target_col": target_col}

    # OUTPUT ESPERADO
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    imputer = SimpleImputer(strategy="mean")
    X_imputada = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_escalada = scaler.fit_transform(X_imputada)

    output_data = (X_escalada, y)

    return input_data, output_data


# Ejemplo de uso
# if __name__ == "__main__":
#     entrada, salida_esperada = generar_caso_de_uso_procesar_datos_meteorologicos()

#     print("=== INPUT ===")
#     print("Nombre de la función esperada: procesar_datos_meteorologicos")
#     print("target_col:", entrada["target_col"])
#     print(entrada["df"])

#     print("\n=== OUTPUT ESPERADO ===")
#     X_res, y_res = salida_esperada
#     print("X procesada:")
#     print(X_res)
#     print("\ny:")
#     print(y_res)
