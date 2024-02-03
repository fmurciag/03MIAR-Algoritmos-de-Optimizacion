import numpy as np


def calculate_cost(solution, matriz_shot_actors):
    num_days = len(solution)
    num_actors = len(matriz_shot_actors[0])
    actor_daily_cost = 5  # Costo diario de desplazamiento para cada actor

    cost = 0
    for d in range(num_days):
        unique_actors = set()
        for i in range(len(solution[d])):
            if solution[d][i] == 1:
                for j in range(num_actors):
                    if matriz_shot_actors[i][j] == 1:
                        unique_actors.add(j)
        daily_cost = len(unique_actors) * actor_daily_cost
        cost += daily_cost

    return cost


def find_optimal_solution(
    matriz_shot_actors, num_days, current_day, current_solution, best_solution, min_cost
):
    if current_day == num_days:
        cost = calculate_cost(current_solution, matriz_shot_actors)
        if cost < min_cost:
            min_cost = cost
            best_solution = current_solution.copy()
        return best_solution, min_cost

    num_shots = len(matriz_shot_actors)
    for i in range(2**num_shots):
        new_solution = np.copy(current_solution)
        # Crear una representación binaria de i con longitud igual a num_shots
        binary_representation = list(bin(i)[2:].zfill(num_shots))
        binary_representation = [int(bit) for bit in binary_representation][-num_shots:]
        new_solution[:, current_day] = binary_representation
        best_solution, min_cost = find_optimal_solution(
            matriz_shot_actors,
            num_days,
            current_day + 1,
            new_solution,
            best_solution,
            min_cost,
        )

    return best_solution, min_cost


# Definición de la matriz_shot_actors y otros parámetros iniciales
# (Utiliza tu matriz_shot_actors original aquí)
matriz_shot_actors = np.array(
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

num_days = 2  # Número de días de grabación
num_shots = len(matriz_shot_actors)

# Inicialización de la solución y el costo mínimo
initial_solution = np.zeros((num_shots, num_days), dtype=int)
best_solution, min_cost = find_optimal_solution(
    matriz_shot_actors, num_days, 0, initial_solution, None, float("inf")
)

print("Planificación óptima encontrada:")
total_cost = calculate_cost(best_solution, matriz_shot_actors)
for d in range(num_days):
    print(f"Día {d+1}:")
    for i in range(len(matriz_shot_actors)):
        if best_solution[i, d] == 1:
            print(f"Toma {i+1} programada")
print(f"Costo total de la planificación: {total_cost}")
