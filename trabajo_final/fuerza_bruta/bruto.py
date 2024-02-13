import numpy as np
import time
import itertools
import pandas as pd

TABLA_ESCENAS = np.genfromtxt("trabajo_final/tabla_escenas.csv", delimiter=",")
N_ESCENAS, N_ACTORES = TABLA_ESCENAS.shape
N_DIAS = (N_ESCENAS + 5) // 6
MAX_TOMAS_POR_DIA = 6
COSTO_POR_ACTOR_POR_DIA = 30
N_INTENTOS_MAX = 10

mejor_total = float("inf")  # Variable global para rastrear el mejor total de actores
mejor_tomas = None  # Variable global para rastrear las mejores tomas seleccionadas


def procesar_tomas(tomas_restantes, tomas_seleccionadas, total_actores, intentos):
    global mejor_total
    global mejor_tomas
    intentos += 1
    print(intentos)
    if intentos > N_INTENTOS_MAX:
        raise ("Intentos maximos")
    if len(tomas_restantes) == 0:  # Si no hay más tomas restantes
        if total_actores < mejor_total:
            # Si el total de actores es menor que el mejor total actual, actualizamos las variables mejor_total y mejor_tomas.
            mejor_total = total_actores
            mejor_tomas = tomas_seleccionadas
            # Imprimimos la mejor combinación de tomas encontrada hasta ahora.
            print("Coste:", mejor_total * COSTO_POR_ACTOR_POR_DIA, "€")
            print(
                "Orden de sesiones:",
                [[toma + 1 for toma in tomas] for tomas in mejor_tomas],
            )
        intentos = 0
        return

    # Obtenemos todas las combinaciones posibles de MAX_TOMAS_POR_DIA tomas restantes.
    combinaciones_tomas = np.array(list(itertools.combinations(tomas_restantes, MAX_TOMAS_POR_DIA)), dtype=int)

    # Itera sobre las combinaciones de tomas.
    for tomas in combinaciones_tomas:
        # Calcula la cantidad de actores en la combinación actual.
        # actores = calcular_actores(tomas)
        actores = len(np.unique(TABLA_ESCENAS[tomas].nonzero()[1]))
        # Actualiza las variables para la próxima llamada recursiva.
        nuevas_tomas_seleccionadas = tomas_seleccionadas + [tomas]
        nuevos_total_actores = total_actores + actores

        # Genera la lista de tomas restantes sin las tomas seleccionadas.
        nuevas_tomas_restantes = [toma for toma in tomas_restantes if toma not in tomas]

        # Realiza una llamada recursiva con las nuevas tomas seleccionadas y restantes.
        procesar_tomas(
            nuevas_tomas_restantes,
            nuevas_tomas_seleccionadas,
            nuevos_total_actores,
            intentos,
        )


def print_schedule(session_order=list(range(N_ESCENAS))):
    """Prints the details of the best schedule found in table format."""
    schedule = [session_order[n : n + MAX_TOMAS_POR_DIA] for n in range(0, len(session_order), MAX_TOMAS_POR_DIA)]
    schedule_details = []

    print("\n\nCalendario de sesiones:")
    print(f"Orden de sesiones: {[x + 1 for x in session_order]}")

    total_cost = 0
    for day_num, day in enumerate(schedule, start=1):
        shots = [x + 1 for x in day]
        sum_shots = np.sum(TABLA_ESCENAS[day, :], axis=0)
        num_actors = np.count_nonzero(sum_shots)
        cost_per_day = num_actors * COSTO_POR_ACTOR_POR_DIA
        total_cost += cost_per_day

        schedule_details.append(
            {
                "Día": day_num,
                "Tomas": ", ".join(map(str, shots)),
                "Numero de actores": num_actors,
                "Coste de la sesion (€)": cost_per_day,
            }
        )

    df_schedule = pd.DataFrame(schedule_details)

    print("Horario:")
    print(df_schedule.to_string(index=False))

    print(f"\nCoste Total:: {total_cost}€")


try:
    intentos = 0
    array_enteros = np.arange(30)
    np.random.shuffle(array_enteros)
    procesar_tomas(array_enteros, [], 0, intentos)

except KeyboardInterrupt:
    print_schedule(np.array(mejor_tomas).flatten())


# Calendario de sesiones:
# Orden de sesiones: [18, 24, 7, 2, 9, 28, 26, 21, 16, 8, 13, 11, 25, 19, 30, 17, 22, 23, 29, 1, 20, 12, 14, 10, 15, 4, 3, 5, 27, 6]
# Horario:
#  Día                  Tomas  Numero de actores  Coste de la sesion (€)
#    1    18, 24, 7, 2, 9, 28                  6                     180
#    2  26, 21, 16, 8, 13, 11                  9                     270
#    3 25, 19, 30, 17, 22, 23                  5                     150
#    4  29, 1, 20, 12, 14, 10                  7                     210
#    5     15, 4, 3, 5, 27, 6                  6                     180

# Coste Total:: 990€
