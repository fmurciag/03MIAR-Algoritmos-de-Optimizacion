import numpy as np
import itertools
import pandas as pd

TABLA_ESCENAS = np.genfromtxt("trabajo_final/tabla_escenas.csv", delimiter=",")
N_ESCENAS, N_ACTORES = TABLA_ESCENAS.shape
N_DIAS = (N_ESCENAS + 5) // 6
MAX_TOMAS_POR_DIA = 6
COSTO_POR_ACTOR_POR_DIA = 30
N_INTENTOS_MAX = 10

best_total = float("inf")  # Global variable to track the best total number of actors
best_shots = None  # Global variable to track the best selected shots


def get_shot_combinations(remaining_shots, n):
    return np.array(list(itertools.combinations(remaining_shots, n)), dtype=int)


def calculate_actors_in_shots(shots):
    return len(np.unique(TABLA_ESCENAS[shots].nonzero()[1]))


def process_combination(selected_shots, total_actors):
    global best_total
    global best_shots

    if total_actors < best_total:
        best_total = total_actors
        best_shots = selected_shots
        print("Cost:", best_total * COSTO_POR_ACTOR_POR_DIA, "€")
        print(
            "Session order:",
            [[shot + 1 for shot in shots] for shots in best_shots],
        )


def process_shots(remaining_shots, selected_shots, total_actors, attempts):
    global best_total
    global best_shots
    attempts += 1

    if attempts > N_INTENTOS_MAX:
        raise Exception("Maximum attempts reached")

    if len(remaining_shots) == 0:
        process_combination(selected_shots, total_actors)
        attempts = 0
        return

    shot_combinations = get_shot_combinations(remaining_shots, MAX_TOMAS_POR_DIA)

    for shots in shot_combinations:
        actors = calculate_actors_in_shots(shots)
        new_selected_shots = selected_shots + [shots]
        new_total_actors = total_actors + actors
        new_remaining_shots = [shot for shot in remaining_shots if shot not in shots]
        process_shots(
            new_remaining_shots,
            new_selected_shots,
            new_total_actors,
            attempts,
        )


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
    intentos = 0
    array_enteros = np.arange(30)
    np.random.shuffle(array_enteros)
    process_shots(array_enteros, [], 0, intentos)

except KeyboardInterrupt:
    print_schedule(np.array(best_shots).flatten())
except Exception as e:
    print("Error:", str(e))
    print_schedule(np.array(best_shots).flatten())
