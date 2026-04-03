import pandas as pd
import numpy as np
import random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def generar_caso_de_uso_preprocesar_muestras_celulares():
    """
    Genera un caso de uso aleatorio para la función
    preprocesar_muestras_celulares(df, target_col).

    Retorna una tupla:
    (
        input_data: dict,
        output_data: tuple
    )
    """

    seed = random.randint(1, 100000)
    np.random.seed(seed)
    random.seed(seed)

    n_filas = random.randint(8, 16)

    df = pd.DataFrame(
        {
            "expresion_gen_1": np.random.uniform(0, 500, n_filas),
            "expresion_gen_2": np.random.uniform(100, 10000, n_filas),
            "tamano_celular": np.random.uniform(5, 50, n_filas),
            "densidad_nuclear": np.random.uniform(0.1, 2.5, n_filas),
            "tipo_muestra": np.random.choice(
                ["tejido_A", "tejido_B", "tejido_C"], size=n_filas
            ),
            "laboratorio_origen": np.random.choice(
                ["lab_norte", "lab_sur"], size=n_filas
            ),
        }
    )

    # Introducir NaNs solo en columnas numéricas
    columnas_numericas = [
        "expresion_gen_1",
        "expresion_gen_2",
        "tamano_celular",
        "densidad_nuclear",
    ]

    n_nans = random.randint(1, max(2, n_filas))
    for _ in range(n_nans):
        fila = random.randint(0, n_filas - 1)
        col = random.choice(columnas_numericas)
        df.loc[fila, col] = np.nan

    target_col = "mutacion_objetivo"
    df[target_col] = np.random.randint(0, 2, size=n_filas)

    # INPUT
    input_data = {"df": df.copy(), "target_col": target_col}

    # OUTPUT ESPERADO
    y = df[target_col].to_numpy()

    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number])

    imputer = SimpleImputer(strategy="mean")
    X_imputada = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_escalada = scaler.fit_transform(X_imputada)

    output_data = (X_escalada, y)

    return input_data, output_data


# Ejemplo de uso
# if __name__ == "__main__":
#     entrada, salida_esperada = generar_caso_de_uso_preprocesar_muestras_celulares()

#     print("=== INPUT ===")
#     print("Nombre de la función esperada: preprocesar_muestras_celulares")
#     print("target_col:", entrada["target_col"])
#     print(entrada["df"])

#     print("\n=== OUTPUT ESPERADO ===")
#     X_res, y_res = salida_esperada
#     print("X procesada:")
#     print(X_res)
#     print("\ny:")
#     print(y_res)
