import numpy as np
from copy import deepcopy

MAX_TOMAS_POR_DIA = 6


def evaluar_solucion(solucion, data):
    # coste = sum(np.any(data != 0, axis=1)[toma] for sesion in solucion for toma in sesion)
    coste = sum(np.any(data != 0, axis=1)[np.array(sesion)] for sesion in solucion)
    return coste


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
    tomas = np.argsort(-data.sum(axis=1))

    for ses_cont in range(len(tomas) // MAX_TOMAS_POR_DIA + 1):
        print("Entrando en sesión: ", ses_cont)
        sesion = []

        for _ in range(MAX_TOMAS_POR_DIA):
            if not tomas.size:
                break  # Sal del bucle si no quedan tomas disponibles
            mejor_toma = seleccion_voraz(solucion, sesion, tomas, data)

            sesion.append(mejor_toma)
            idx = np.where(tomas == mejor_toma)
            tomas = np.delete(tomas, idx)
        solucion.append(sesion)
        print(solucion)

    return solucion


TABLA_ESCENAS = np.genfromtxt("trabajo_final/tabla_escenas.csv", delimiter=",")
sol = busqueda_voraz(TABLA_ESCENAS)
print("La solución es: ", sol)
print("con coste: ", evaluar_solucion(sol, TABLA_ESCENAS) * 30)
