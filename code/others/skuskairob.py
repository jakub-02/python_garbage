import random
import time

def fitness(path, game_map):
    visited = set()
    coins_collected = 0
    x, y = 0, 0

    for move in path:
        if move == 'U':
            x = max(0, x - 1)
        elif move == 'D':
            x = min(len(game_map) - 1, x + 1)
        elif move == 'L':
            y = max(0, y - 1)
        elif move == 'R':
            y = min(len(game_map[0]) - 1, y + 1)

        if (x, y) not in visited and game_map[x][y] == 'C':
            coins_collected += 1
            visited.add((x, y))

    return coins_collected

def generate_random_path(length=100):
    return [random.choice(['U', 'D', 'L', 'R']) for _ in range(length)]

def crossover(parent1, parent2):
    split = random.randint(1, len(parent1) - 1)
    child1 = parent1[:split] + parent2[split:]
    child2 = parent2[:split] + parent1[split:]
    return child1, child2

def mutate(path, mutation_rate=0.01):
    for i in range(len(path)):
        if random.random() < mutation_rate:
            path[i] = random.choice(['U', 'D', 'L', 'R'])
    return path

def genetic_algorithm(game_map, population_size, generations, path_length, parent1, parent2):
    population = [
        parent1,
        parent2,
        *[generate_random_path(path_length) for _ in range(population_size - 2)]
    ]

    for generation in range(generations):
        fitness_scores = [(path, fitness(path, game_map)) for path in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        population = [path for path, _ in fitness_scores[:population_size // 2]]

        new_population = []
        while len(new_population) < population_size:
            p1, p2 = random.sample(population, 2)
            child1, child2 = crossover(p1, p2)
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))

        population = new_population[:population_size]

        best_path, best_score = fitness_scores[0]
        print(f"Generation {generation + 1}: Best fitness = {best_score}")

        if best_score == len([cell for row in game_map for cell in row if cell == 'C']):
            print("All coins collected!")
            break

    return fitness_scores[0]

def load_map_from_file(filename):
    with open(filename, 'r') as f:
        return [list(line.strip()) for line in f.readlines()]

def visualize_path(game_map, path):
    x, y = 0, 0
    for move in path:
        if move == 'U':
            x = max(0, x - 1)
        elif move == 'D':
            x = min(len(game_map) - 1, x + 1)
        elif move == 'L':
            y = max(0, y - 1)
        elif move == 'R':
            y = min(len(game_map[0]) - 1, y + 1)

        if game_map[x][y] == 'C':
            game_map[x][y] = '.'

        game_map_copy = [row[:] for row in game_map]
        game_map_copy[x][y] = 'P'
        print("\n".join(" ".join(row) for row in game_map_copy))
        print("\n---\n")
        time.sleep(0.5)

if __name__ == "__main__":
    game_map = load_map_from_file("data/map.txt")

    generations = int(input("Enter number of generations: "))
    path_length = 100
    population_size = 20

    parent1 = list(input("Enter first parent path (U/D/L/R only): "))
    parent2 = list(input("Enter second parent path (U/D/L/R only): "))

    best_path, best_fitness = genetic_algorithm(
        game_map, population_size, generations, path_length, parent1, parent2
    )

    print("Visualization of the best path:")
    visualize_path(game_map, best_path)

    print("Best path:", ''.join(best_path))
    print("Coins collected:", best_fitness)
