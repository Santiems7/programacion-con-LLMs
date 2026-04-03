import pandas as pd
import numpy as np
import random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def generar_caso_de_uso_preparar_muestras_quimicas():
    """
    Genera un caso de uso aleatorio para la función
    preparar_muestras_quimicas(df, target_col).

    Retorna una tupla:
    (
        input_data: dict,
        output_data: tuple
    )
    """

    seed = random.randint(1, 100000)
    np.random.seed(seed)
    random.seed(seed)

    n_filas = random.randint(8, 15)

    df = pd.DataFrame(
        {
            "acidez": np.random.uniform(1.0, 14.0, n_filas),
            "sales_disueltas": np.random.uniform(10, 500, n_filas),
            "conductividad": np.random.uniform(50, 2000, n_filas),
            "densidad": np.random.uniform(0.8, 1.5, n_filas),
            "columna_constante": np.full(n_filas, 7.0),
        }
    )

    # Introducir NaNs aleatorios solo en columnas no constantes
    columnas_con_nans = ["acidez", "sales_disueltas", "conductividad", "densidad"]
    n_nans = random.randint(1, max(2, n_filas // 2))

    for _ in range(n_nans):
        fila = random.randint(0, n_filas - 1)
        col = random.choice(columnas_con_nans)
        df.loc[fila, col] = np.nan

    target_col = "riesgo_quimico"
    df[target_col] = np.random.randint(0, 2, size=n_filas)

    # INPUT
    input_data = {"df": df.copy(), "target_col": target_col}

    # OUTPUT ESPERADO
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    imputer = SimpleImputer(strategy="mean")
    X_imputada = imputer.fit_transform(X)

    # Eliminar columnas con varianza 0
    varianzas = np.var(X_imputada, axis=0)
    X_filtrada = X_imputada[:, varianzas > 0]

    scaler = StandardScaler()
    X_escalada = scaler.fit_transform(X_filtrada)

    output_data = (X_escalada, y)

    return input_data, output_data


# # Ejemplo de uso
# if __name__ == "__main__":
#     entrada, salida_esperada = generar_caso_de_uso_preparar_muestras_quimicas()

#     print("=== INPUT ===")
#     print("Nombre de la función esperada: preparar_muestras_quimicas")
#     print("target_col:", entrada["target_col"])
#     print(entrada["df"])

#     print("\n=== OUTPUT ESPERADO ===")
#     X_res, y_res = salida_esperada
#     print("X procesada:")
#     print(X_res)
#     print("\ny:")
#     print(y_res)
