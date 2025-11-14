import random

# ================================
# Problem Setup
# ================================
distance_matrix = [
    [0, 10, 15, 20, 25],
    [10, 0, 35, 25, 30],
    [15, 35, 0, 30, 10],
    [20, 25, 30, 0, 15],
    [25, 30, 10, 15, 0]
]

NUM_CUSTOMERS = 4           # Customers 1..4
NUM_VEHICLES = 2
POPULATION_SIZE = 10
GENERATIONS = 50
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.9

customers = list(range(1, NUM_CUSTOMERS + 1))


# ================================
# Decode: split route into vehicles
# ================================
def decode(chromosome):
    """Convert permutation into vehicle routes by splitting evenly."""
    size = len(chromosome) // NUM_VEHICLES
    return [chromosome[i*size:(i+1)*size] for i in range(NUM_VEHICLES)]


# ================================
# Fitness = total VRP distance
# ================================
def fitness_function(chromosome):
    routes = decode(chromosome)
    total_distance = 0

    for route in routes:
        current = 0  # depot
        for c in route:
            total_distance += distance_matrix[current][c]
            current = c
        total_distance += distance_matrix[current][0]  # return to depot

    return -total_distance    # NEGATIVE because GA maximizes fitness


# ================================
# Evaluate population
# ================================
def evaluate_population(population):
    return [fitness_function(ind) for ind in population]


# ================================
# Selection: Roulette Wheel
# ================================
def select(population, fitnesses):
    total = sum(fitnesses)
    pick = random.uniform(0, total)
    current = 0
    for indiv, fit in zip(population, fitnesses):
        current += fit
        if current >= pick:
            return indiv
    return random.choice(population)


# ================================
# Crossover: Order Crossover (OX)
# ================================
def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1, parent2

    size = len(parent1)
    p1, p2 = sorted(random.sample(range(size), 2))

    child1 = parent1[p1:p2] + [x for x in parent2 if x not in parent1[p1:p2]]
    child2 = parent2[p1:p2] + [x for x in parent1 if x not in parent2[p1:p2]]

    return child1, child2


# ================================
# Mutation: swap two customers
# ================================
def mutate(chromosome):
    chromosome = chromosome[:]
    if random.random() < MUTATION_RATE:
        a, b = random.sample(range(len(chromosome)), 2)
        chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
    return chromosome


# ================================
# Initial Population
# ================================
def initial_population():
    population = []
    for _ in range(POPULATION_SIZE):
        chrom = customers[:]
        random.shuffle(chrom)
        population.append(chrom)
    return population


# ================================
# Genetic Algorithm
# ================================
def genetic_algorithm():
    population = initial_population()
    best_solution = None
    best_fitness = float('-inf')

    for gen in range(GENERATIONS):
        fitnesses = evaluate_population(population)

        # Track best
        for i, chrom in enumerate(population):
            if fitnesses[i] > best_fitness:
                best_fitness = fitnesses[i]
                best_solution = chrom

        print(f"Generation {gen+1}: Best Fitness = {best_fitness}")

        # Create next generation
        new_pop = []
        while len(new_pop) < POPULATION_SIZE:
            p1 = select(population, fitnesses)
            p2 = select(population, fitnesses)

            o1, o2 = crossover(p1, p2)
            o1 = mutate(o1)
            o2 = mutate(o2)

            new_pop.extend([o1, o2])

        population = new_pop[:POPULATION_SIZE]

    return best_solution, -best_fitness   # return positive distance


# ================================
# RUN
# ================================
solution, distance = genetic_algorithm()

print("\nBest Chromosome (Permutation):", solution)
routes = decode(solution)

print("\nVehicle Routes:")
for v, route in enumerate(routes):
    print(f"Vehicle {v+1}: {route}")

print("\nTotal Distance:", distance)
