import random
import numpy as np
import pandas as pd


def read_csv_to_dataframe(csv_path):
    """
    Lee un archivo CSV y lo convierte en un DataFrame de pandas.

    :param csv_path: Ruta del archivo CSV.
    :return: DataFrame de pandas.
    """
    try:
        # Intentar leer el archivo CSV
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        # En caso de un error, retornar el mensaje de error
        return str(e)


# Usamos la función para leer el archivo CSV que se cargó anteriormente
csv_path = "trabajo_final/genetico\Libro1.csv"
df = read_csv_to_dataframe(csv_path)
df
