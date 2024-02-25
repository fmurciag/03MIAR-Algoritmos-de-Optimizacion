import numpy as np
from copy import deepcopy

MAX_TOMAS_POR_DIA = 6


def obtener_tomas(data):
    return np.arange(data.shape[0])


def ordenar_actores(actores, data):
    suma_columnas = np.sum(data, axis=0)
    indices_ordenados = np.argsort(suma_columnas)[::-1]
    return actores[indices_ordenados]


def ordenar_tomas(tomas, data):
    suma_filas = np.sum(data, axis=1)
    indices_ordenados = np.argsort(suma_filas)[::-1]
    return tomas[indices_ordenados]


def evaluar_solucion(solucion, data):
    coste = 0
    for sesion in solucion:
        # Accede a las filas especificadas por los índices de 'sesion' en el array 'data'
        data_sesion = data[sesion, :]
        # Calcula si hay algún valor no cero en cada columna de 'data_sesion' y suma esos valores
        # np.any(data_sesion != 0, axis=0) devuelve un array booleano a lo largo del eje de las columnas
        # .sum() suma los valores True (considerados como 1)
        coste += np.any(data_sesion != 0, axis=0).sum()

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
    actores = np.arange(data.shape[1])
    actores = ordenar_actores(actores, data)
    tomas = obtener_tomas(data)
    tomas = ordenar_tomas(tomas, data)

    ses_cont = 0
    while tomas.size > 0:
        print("Entrando en sesión: ", ses_cont)
        sesion = []
        for _ in range(MAX_TOMAS_POR_DIA):
            print("Generando toma: ", _)
            mejor_toma = seleccion_voraz(solucion, sesion, tomas, data)

            sesion += [mejor_toma]
            idx = np.where(tomas == mejor_toma)
            tomas = np.delete(tomas, idx)
        solucion.append(sesion)
        ses_cont += 1
        print(solucion)

    return solucion


TABLA_ESCENAS = np.genfromtxt("trabajo_final/tabla_escenas.csv", delimiter=",")
sol = busqueda_voraz(TABLA_ESCENAS)
print("La solución es: ", sol)
print("con coste: ", evaluar_solucion(sol, TABLA_ESCENAS))
