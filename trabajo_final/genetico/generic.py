import random
import numpy as np

# Toma la tabla de las escenas
tabla_escenas = np.array(
    [
        # 1  2  3  4  5  6  7  8  9 10 actor / toma
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # 1
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],  # 2
        [0, 1, 0, 0, 1, 0, 1, 0, 0, 0],  # 3
        [1, 1, 0, 0, 0, 0, 1, 1, 0, 0],  # 4
        [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],  # 5
        [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],  # 6
        [1, 1, 0, 1, 1, 0, 0, 0, 0, 0],  # 7
        [1, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # 8
        [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],  # 9
        [1, 1, 0, 0, 0, 1, 0, 0, 1, 0],  # 10
        [1, 1, 1, 0, 1, 0, 0, 1, 0, 0],  # 11
        [1, 1, 1, 1, 0, 1, 0, 0, 0, 0],  # 12
        [1, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # 13
        [1, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # 14
        [1, 1, 0, 0, 0, 0, 1, 0, 0, 0],  # 15
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],  # 16
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 17
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # 18
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 19
        [1, 0, 1, 1, 1, 0, 0, 0, 0, 0],  # 20
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],  # 21
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # 22
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 23
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # 24
        [1, 1, 0, 1, 0, 0, 0, 0, 0, 1],  # 25
        [1, 0, 1, 0, 1, 0, 0, 0, 1, 0],  # 26
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # 27
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 28
        [1, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # 29
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 30
    ]
)

N_ESCENAS, N_ACTORES = tabla_escenas.shape
N_DIAS = (N_ESCENAS + 5) // 6  # Cálculo del número máximo de días
max_tomas_por_dia = 6
cost_por_actor_por_dia = 30
# Generar una población inicial
def generate_population(tamano):
    return [random.sample(range(N_ESCENAS), N_ESCENAS) for _ in range(tamano)]


# Calcular aptitud
def fitness(individual):
    horario = [individual[n : n + 6] for n in range(0, len(individual), 6)]
    costo = 0
    for dia in horario:
        # suma las filas correspondientes a las escenas del día
        sum_tomas = np.sum(tabla_escenas[dia, :], axis=0)
        # cuenta cuántos actores tienen al menos una aparición
        actores = np.count_nonzero(sum_tomas)
        costo += (
            actores * cost_por_actor_por_dia
        )  # se multiplica por el costo por actor por día
    print(f"Costo: {costo} ")
    return (costo,)


# Cruzamiento
def crossover(ind1, ind2):
    idx = range(len(ind1))
    cross_points = sorted(random.sample(idx, 2))
    return (
        ind1[: cross_points[0]]
        + ind2[cross_points[0] : cross_points[1]]
        + ind1[cross_points[1] :],
    )


# Mutación
def mutate(individual):
    idx = range(len(individual))
    puntos_mutar = random.sample(idx, 2)
    individual[puntos_mutar[0]], individual[puntos_mutar[1]] = (
        individual[puntos_mutar[1]],
        individual[puntos_mutar[0]],
    )
    return (individual,)


# Selección de los mejores individuos
def selection(poblacion, fitnesses, num_padres, num_direct_copias):
    parents_idx = np.argsort(fitnesses)[:num_padres]
    padres = [poblacion[int(i)] for i in parents_idx]
    offsprings = (
        padres[:num_direct_copias]
        + [
            crossover(padres[i], padres[i + 1])
            for i in range(num_direct_copias, num_padres - 1, 2)
        ]
        + [mutate(padres[i]) for i in range(num_direct_copias + 1, num_padres, 2)]
    )
    return offsprings


# Algoritmo genético
def genetic_algorithm():
    num_generaciones = 500
    poblacion_tamano = 500
    num_padres = poblacion_tamano // 2
    num_direct_copias = num_padres // 2

    poblacion = generate_population(poblacion_tamano)

    for _ in range(num_generaciones):
        fitnesses = [fitness(ind) for ind in poblacion]
        # print(f"salida: {fitnesses} ")
        poblacion = selection(poblacion, fitnesses, num_padres, num_direct_copias)
    return min(poblacion, key=fitness)


def imprimirDetalle(individual):
    print("\nRESULTADOS:")
    horario = [individual[n : n + 6] for n in range(0, len(individual), 6)]
    print(f"Mejor cromosoma: {[x + 1 for x in individual]}")
    costo = 0
    num_dia = 0
    print("\nHorario:")
    for day in horario:
        num_dia += 1
        print(f"Día {num_dia}:")
        print(f"Tomas: {[x + 1 for x in day]}")
        # suma las filas correspondientes a las escenas del día
        sum_tomas = np.sum(tabla_escenas[day, :], axis=0)
        # cuenta cuántos actores tienen al menos una aparición
        actors = np.count_nonzero(sum_tomas)
        print(f"Actores: {actors}")
        costo += (
            actors * cost_por_actor_por_dia
        )  # se multiplica por el costo por actor por día
    print(f"\nCoste: {costo}€")


mejor_horario = genetic_algorithm()
imprimirDetalle(mejor_horario)
# RESULTADOS:
# Mejor cromosoma: [6, 25, 2, 9, 1, 22, 23, 16, 10, 13, 24, 18, 29, 3, 20, 28, 21, 30, 27, 12, 17, 11, 14, 5, 15, 19, 7, 26, 8, 4]
#
# Horario:
# Día 1:
# Tomas: [6, 25, 2, 9, 1, 22]
# Actores: 6
# Día 2:
# Tomas: [23, 16, 10, 13, 24, 18]
# Actores: 8
# Día 3:
# Tomas: [29, 3, 20, 28, 21, 30]
# Actores: 8
# Día 4:
# Tomas: [27, 12, 17, 11, 14, 5]
# Actores: 7
# Día 5:
# Tomas: [15, 19, 7, 26, 8, 4]
# Actores: 9
#
# Coste: 1140€
#
