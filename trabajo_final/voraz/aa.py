import numpy as np
from copy import copy, deepcopy
import pandas as pd

TABLA_ESCENAS = np.array(
    [
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
        [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 0, 1, 0, 0, 1, 0, 0],
        [1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    ]
)
N_ESCENAS, N_ACTORES = TABLA_ESCENAS.shape
N_DIAS = (N_ESCENAS + 5) // 6
MAX_TOMAS_POR_DIA = 6
COSTO_POR_ACTOR_POR_DIA = 30
N_INTENTOS_MAX = 10
N_ESCENAS, N_ACTORES = TABLA_ESCENAS.shape
N_DIAS = (N_ESCENAS + 5) // 6
MAX_TOMAS_POR_DIA = 6
COSTO_POR_ACTOR_POR_DIA = 30
N_INTENTOS_MAX = 10


def evaluar_solucion(solucion, data):
    coste = 0
    for sesion in solucion:
        coste += np.any(data[sesion, :] != 0, axis=0).sum()

    return coste


def ordenar_tomas(tomas, data):
    suma_filas = np.sum(data, axis=1)
    indices_ordenados = np.argsort(suma_filas)[::-1]
    return tomas[indices_ordenados]


def seleccion_voraz(solucion, sesion, tomas, data):
    mejor_toma = tomas[0]

    solucion_temp = deepcopy(solucion)
    sesion_temp = deepcopy(sesion)

    solucion_temp.append(sesion_temp + [mejor_toma])
    menor_coste = evaluar_solucion(solucion_temp, data)
    for i in range(1, len(tomas)):
        solucion_temp = deepcopy(solucion)
        sesion_temp = deepcopy(sesion)
        solucion_temp.append(sesion_temp + [tomas[i]])
        coste = evaluar_solucion(solucion_temp, data)

        if coste < menor_coste:
            mejor_toma = tomas[i]
            menor_coste = coste

    return mejor_toma


def busqueda_voraz(data):

    solucion = []
    tomas = np.arange(data.shape[0])
    tomas = ordenar_tomas(tomas, data)

    ses_cont = 0
    while tomas.shape[0] > 0:
        print("Entrando en sesión: ", ses_cont)
        sesion = []
        for _ in range(6):
            print("Generando toma: ", _)
            mejor_toma = seleccion_voraz(solucion, sesion, tomas, data)

            sesion += [mejor_toma]
            idx = np.argwhere(tomas == mejor_toma)
            tomas = np.delete(tomas, idx)
        solucion.append(sesion)
        ses_cont += 1
        print(solucion)

    return solucion


def print_schedule(session_order=list(range(N_ESCENAS))):
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


TABLA_ESCENAS = np.genfromtxt("trabajo_final/tabla_escenas.csv", delimiter=",")
sol = busqueda_voraz(TABLA_ESCENAS)
print("con coste: ", evaluar_solucion(sol, TABLA_ESCENAS))
print_schedule(np.array(sol).flatten())
