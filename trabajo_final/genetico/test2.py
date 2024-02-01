import random
import numpy as np
import pandas as pd

# Definiciones iniciales
TABLA_ESCENAS = np.genfromtxt("trabajo_final/tabla_escenas.csv", delimiter=",")
N_ESCENAS, N_ACTORES = TABLA_ESCENAS.shape
N_DIAS = (N_ESCENAS + 5) // 6
MAX_TOMAS_POR_DIA = 6
COSTO_POR_ACTOR_POR_DIA = 30
np.random.seed(0)
random.seed(0)


def generate_population(size):
    """Genera una población de secuencias de escenas aleatorias."""
    return [np.random.permutation(N_ESCENAS).tolist() for _ in range(size)]


def calculate_cost(schedule):
    """Calcula el costo total basado en el número de actores por día."""
    day_sums = np.sum(TABLA_ESCENAS[schedule, :], axis=0)
    actors = np.count_nonzero(day_sums)
    return actors * COSTO_POR_ACTOR_POR_DIA


def fitness(individual):
    """Calcula la aptitud de un individuo basada en el costo total."""
    horario = [
        individual[n : n + MAX_TOMAS_POR_DIA]
        for n in range(0, N_ESCENAS, MAX_TOMAS_POR_DIA)
    ]
    costo_total = sum(calculate_cost(dia) for dia in horario)
    print(f"Costo: {costo_total} ")
    return (costo_total,)


# def fitness(individual):
#     horario = [
#         individual[n : n + MAX_TOMAS_POR_DIA]
#         for n in range(0, N_ESCENAS, MAX_TOMAS_POR_DIA)
#     ]
#     costo = 0
#     for dia in horario:
#         # suma las filas correspondientes a las escenas del día
#         sum_tomas = np.sum(TABLA_ESCENAS[dia, :], axis=0)
#         # cuenta cuántos actores tienen al menos una aparición
#         actores = np.count_nonzero(sum_tomas)
#         costo += (
#             actores * COSTO_POR_ACTOR_POR_DIA
#         )  # se multiplica por el costo por actor por día
#     print(f"Costo: {costo} ")
#     return (costo,)


def crossover(ind1, ind2):
    """Realiza el cruzamiento entre dos individuos."""
    size = len(ind1)
    pto1, pto2 = sorted(random.sample(range(size), 2))
    new_ind = ind1[:pto1] + ind2[pto1:pto2] + ind1[pto2:]
    return new_ind


def mutate(individual):
    """Aplica una mutación intercambiando dos elementos aleatorios."""
    idx1, idx2 = random.sample(range(len(individual)), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual


def selection(population, fitnesses, num_parents, num_direct_copies):
    """Selecciona los mejores individuos para la siguiente generación."""
    sorted_idx = np.argsort(fitnesses)[:num_parents]
    parents = [population[int(i)] for i in sorted_idx]
    offsprings = parents[:num_direct_copies]
    for i in range(num_direct_copies, num_parents - 1, 2):
        offsprings.append(crossover(parents[i], parents[i + 1]))
    for i in range(num_direct_copies + 1, num_parents, 2):
        offsprings.append(mutate(parents[i]))
    return offsprings


# def selection(population, fitnesses, num_parents, num_direct_copies):
#     """Selecciona los mejores individuos para la siguiente generación."""
#     sorted_idx = np.argsort(fitnesses)
#     parents = [population[i] for i in sorted_idx[:num_parents]]
#     offspring = parents[:num_direct_copies]
#     for i in range(num_direct_copies, num_parents, 2):
#         offspring.append(crossover(parents[i], parents[(i + 1) % num_parents]))
#     for i in range(num_direct_copies):
#         offspring.append(mutate(parents[i].copy()))
#     return offspring

# def selection(poblacion, fitnesses, num_padres, num_direct_copias):
#     parents_idx = np.argsort(fitnesses)[:num_padres]
#     padres = [poblacion[int(i)] for i in parents_idx]
#     offsprings = (
#         padres[:num_direct_copias]
#         + [
#             crossover(padres[i], padres[i + 1])
#             for i in range(num_direct_copias, num_padres - 1, 2)
#         ]
#         + [mutate(padres[i]) for i in range(num_direct_copias + 1, num_padres, 2)]
#     )
#     return offsprings


def genetic_algorithm(population_size=500, generations=500):
    """Ejecuta el algoritmo genético."""
    population = generate_population(population_size)
    for _ in range(generations):
        fitnesses = [fitness(ind) for ind in population]
        population = selection(
            population, fitnesses, population_size // 2, population_size // 4
        )
    best_individual = min(population, key=fitness)
    return best_individual


# def genetic_algorithm():
#     num_generaciones = 500
#     poblacion_tamano = 500
#     num_padres = poblacion_tamano // 2
#     num_direct_copias = num_padres // 2

#     poblacion = generate_population(poblacion_tamano)

#     for _ in range(num_generaciones):
#         fitnesses = [fitness(ind) for ind in poblacion]
#         # print(f"salida: {fitnesses} ")
#         poblacion = selection(poblacion, fitnesses, num_padres, num_direct_copias)
#     return min(poblacion, key=fitness)


def imprimirDetalle(individual):
    """Imprime los detalles del mejor horario encontrado."""
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
        sum_tomas = np.sum(TABLA_ESCENAS[day, :], axis=0)
        # cuenta cuántos actores tienen al menos una aparición
        actors = np.count_nonzero(sum_tomas)
        print(f"Actores: {actors}")
        costo += (
            actors * COSTO_POR_ACTOR_POR_DIA
        )  # se multiplica por el costo por actor por día
    print(f"\nCoste: {costo}€")


def imprimirDetalle(individual):
    """Imprime los detalles del mejor horario encontrado en forma de tabla."""
    # Asegúrate de que TABLA_ESCENAS y COSTO_POR_ACTOR_POR_DIA estén definidos
    global TABLA_ESCENAS, COSTO_POR_ACTOR_POR_DIA

    horario = [individual[n : n + 6] for n in range(0, len(individual), 6)]
    detalles_horario = []

    print("\nRESULTADOS:")
    print(f"Mejor cromosoma: {[x + 1 for x in individual]}")

    costo_total = 0
    for num_dia, day in enumerate(horario, start=1):
        tomas = [x + 1 for x in day]
        sum_tomas = np.sum(TABLA_ESCENAS[day, :], axis=0)
        actores = np.count_nonzero(sum_tomas)
        costo_dia = actores * COSTO_POR_ACTOR_POR_DIA
        costo_total += costo_dia

        # Añade los detalles del día a la lista para construir la tabla
        detalles_horario.append(
            {
                "Día": num_dia,
                "Tomas": ", ".join(map(str, tomas)),
                "Actores": actores,
                "Costo del Día (€)": costo_dia,
            }
        )

    # Crea un DataFrame de pandas con los detalles del horario
    df_horario = pd.DataFrame(detalles_horario)

    # Imprimir la tabla de horario
    print("\nHorario:")
    print(df_horario.to_string(index=False))

    print(f"\nCoste Total: {costo_total}€")


# Ejecución del algoritmo y presentación de resultados
mejor_horario = genetic_algorithm()
imprimirDetalle(mejor_horario)
print(sorted(mejor_horario))

# RESULTADOS:
# Mejor cromosoma: [1, 12, 20, 28, 19, 5, 14, 6, 3, 24, 8, 9, 7, 23, 17, 25, 22, 30, 10, 21, 11, 15, 13, 4, 29, 26, 18, 27, 2, 16]

# Horario:
# Día 1:
# Tomas: [1, 12, 20, 28, 19, 5]
# Actores: 7
# Día 2:
# Tomas: [14, 6, 3, 24, 8, 9]
# Actores: 7
# Día 3:
# Tomas: [7, 23, 17, 25, 22, 30]
# Actores: 6
# Día 4:
# Tomas: [10, 21, 11, 15, 13, 4]
# Actores: 9
# Día 5:
# Tomas: [29, 26, 18, 27, 2, 16]
# Actores: 7

# Coste: 1080€


# RESULTADOS:
# Mejor cromosoma: [14, 20, 19, 26, 16, 2, 11, 9, 17, 29, 3, 1, 22, 15, 12, 4, 30, 23, 24, 21, 18, 10, 8, 25, 5, 13, 28, 6, 27, 7]

# Horario:
# Día 1:
# Tomas: [14, 20, 19, 26, 16, 2]
# Actores: 7
# Día 2:
# Tomas: [11, 9, 17, 29, 3, 1]
# Actores: 8
# Día 3:
# Tomas: [22, 15, 12, 4, 30, 23]
# Actores: 7
# Día 4:
# Tomas: [24, 21, 18, 10, 8, 25]
# Actores: 8
# Día 5:
# Tomas: [5, 13, 28, 6, 27, 7]
# Actores: 5

# Coste: 1050€
