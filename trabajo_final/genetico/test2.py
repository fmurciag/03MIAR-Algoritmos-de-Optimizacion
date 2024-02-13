import random
import numpy as np
import pandas as pd

# Definiciones iniciales
TABLA_ESCENAS = np.genfromtxt("trabajo_final/tabla_escenas.csv", delimiter=",")
N_ESCENAS, N_ACTORES = TABLA_ESCENAS.shape
N_DIAS = (N_ESCENAS + 5) // 6
MAX_TOMAS_POR_DIA = 6
COSTO_POR_ACTOR_POR_DIA = 30
# np.random.seed(0)
# random.seed(0)


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
    horario = [individual[n : n + MAX_TOMAS_POR_DIA] for n in range(0, N_ESCENAS, MAX_TOMAS_POR_DIA)]
    costo_total = sum(calculate_cost(dia) for dia in horario)
    print(f"Costo: {costo_total} ")
    return (costo_total,)


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


def generate_schedule_genetic_algorithm(population_size=500, generations=500):
    """Ejecuta el algoritmo genético."""
    population = generate_population(population_size)
    for _ in range(generations):
        fitnesses = [fitness(ind) for ind in population]
        population = selection(population, fitnesses, population_size // 2, population_size // 4)
    best_session = min(population, key=fitness)
    print_schedule(best_session)
    return best_session


def print_schedule(session_order=list(range(N_ESCENAS))):
    """Prints the details of the best schedule found in table format."""
    schedule = [session_order[n : n + 6] for n in range(0, len(session_order), 6)]
    schedule_details = []

    print("\n\n Calendario de sesiones:")
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


# Ejecución del algoritmo y presentación de resultados
generate_schedule_genetic_algorithm(population_size=100, generations=1000)


# RESULTADOS:
# Mejor cromosoma: [7, 17, 4, 18, 3, 11, 21, 24, 26, 10, 14, 23, 27, 13, 22, 19, 15, 9, 5, 1, 25, 12, 2, 8, 6, 28, 30, 29, 20, 16]

# Horario:
#  Día                  Tomas  Actores  Costo del Día (€)
#    1    7, 17, 4, 18, 3, 11        8                240
#    2 21, 24, 26, 10, 14, 23        7                210
#    3  27, 13, 22, 19, 15, 9        6                180
#    4     5, 1, 25, 12, 2, 8        8                240
#    5  6, 28, 30, 29, 20, 16        7                210

# Coste Total: 1080€
