import numpy as np
import itertools
import pandas as pd

TABLA_ESCENAS = np.genfromtxt("trabajo_final/tabla_escenas.csv", delimiter=",")
N_ESCENAS, N_ACTORES = TABLA_ESCENAS.shape
N_DIAS = (N_ESCENAS + 5) // 6
MAX_TOMAS_POR_DIA = 6
COSTO_POR_ACTOR_POR_DIA = 30
N_INTENTOS_MAX = 10

BEST_COST = float("inf")  # Global variable to track the best total number of actors
BEST_SCHEDULE = np.array([])


def get_shot_combinations(remaining_shots, n):
    """Genera todas las combinaciones posibles de tomas a partir de las tomas restantes."""
    return np.array(list(itertools.combinations(remaining_shots, n)), dtype=int)


def calculate_actors_in_shots(shots):
    """Calcula el número de actores que participan en un conjunto de tomas."""
    return len(np.unique(TABLA_ESCENAS[shots].nonzero()[1]))


def save_posible_best_schedule(selected_shots, total_actors):
    global BEST_COST
    global BEST_SCHEDULE

    if total_actors < BEST_COST:
        BEST_COST = total_actors
        BEST_SCHEDULE = selected_shots.copy()
        print_schedule(BEST_SCHEDULE.flatten())


def generate_schedule_brute_algorithm(remaining_shots, selected_shots, total_actors):
    """Ejecuta el algoritmo por fuerza bruta"""

    if len(remaining_shots) == 0:
        save_posible_best_schedule(selected_shots, total_actors)
        return

    shot_combinations = get_shot_combinations(remaining_shots, MAX_TOMAS_POR_DIA)

    for shots in shot_combinations:
        actors = calculate_actors_in_shots(shots)
        new_selected_shots = np.vstack((selected_shots, shots))
        new_total_actors = total_actors + actors
        new_remaining_shots = np.array([shot for shot in remaining_shots if shot not in shots])

        generate_schedule_brute_algorithm(new_remaining_shots, new_selected_shots, new_total_actors)


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


try:
    initial_schedule = np.arange(N_ESCENAS)
    np.random.shuffle(initial_schedule)
    generate_schedule_brute_algorithm(initial_schedule, np.empty((0, MAX_TOMAS_POR_DIA), dtype=int), 0)
except KeyboardInterrupt:
    print_schedule(BEST_SCHEDULE.flatten())
