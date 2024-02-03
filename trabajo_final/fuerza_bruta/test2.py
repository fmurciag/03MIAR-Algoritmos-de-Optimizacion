import itertools
import numpy as np

TABLA_ESCENAS = np.genfromtxt("trabajo_final/tabla_escenas.csv", delimiter=",")
N_ESCENAS, N_ACTORES = TABLA_ESCENAS.shape
N_DIAS = (N_ESCENAS + 5) // 6
MAX_TOMAS_POR_DIA = 6
COSTO_POR_ACTOR_POR_DIA = 30

# Define las variables globales mejor_total y mejor_tomas
mejor_total = float("inf")
mejor_tomas = []

# Define la función para obtener combinaciones de tomas
def obtener_combinaciones_tomas(tomas_restantes, n):
    return list(itertools.combinations(tomas_restantes, n))


# Define la función para calcular la cantidad de actores en un conjunto de tomas
def calcular_actores(tomas):
    actores = len(np.unique(TABLA_ESCENAS[tomas].nonzero()[1]))
    return actores


# Define la función para procesar una combinación de tomas
def procesar_combinacion(tomas_seleccionadas, total_actores):
    global mejor_total
    global mejor_tomas

    if total_actores < mejor_total:
        mejor_total = total_actores
        mejor_tomas = tomas_seleccionadas
        print(
            "El menor número de desplazamientos al estudio de actores hasta ahora:",
            mejor_total * 30,
        )
        print(
            "Tomas seleccionadas:",
            [[toma + 1 for toma in tomas] for tomas in mejor_tomas],
        )


# Define la función principal para procesar tomas
def procesar_tomas(tomas_restantes, tomas_seleccionadas, total_actores):
    if len(tomas_restantes) == 0:
        procesar_combinacion(tomas_seleccionadas, total_actores)
        return

    combinaciones_tomas = obtener_combinaciones_tomas(tomas_restantes, 6)

    for tomas in combinaciones_tomas:
        actores = calcular_actores(tomas)
        nuevas_tomas_seleccionadas = tomas_seleccionadas + [tomas]
        nuevos_total_actores = total_actores + actores
        nuevas_tomas_restantes = [toma for toma in tomas_restantes if toma not in tomas]
        procesar_tomas(
            nuevas_tomas_restantes, nuevas_tomas_seleccionadas, nuevos_total_actores
        )


# Aquí deberías incluir la definición de TABLA_ESCENAS si no está definida anteriormente

if __name__ == "__main__":

try:
    array_enteros = np.arange(30)
    np.random.shuffle(array_enteros)
    procesar_tomas(array_enteros, [], 0)
except KeyboardInterrupt:
    print("Búsqueda interrumpida.")
