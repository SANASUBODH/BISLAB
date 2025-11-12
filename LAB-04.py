import random
import math

# Fitness function (objective function to minimize)
def fitness(x):
    return x**2  # f(x) = x^2, global minimum at x = 0

# Levy flight for generating new solutions
def levy_flight(beta=1.5):
    # Generate a random step using Levy flight (simplified version)
    u = random.gauss(0, 1)  # Gaussian distribution for u
    v = random.gauss(0, 1)  # Gaussian distribution for v
    step = u / abs(v) ** (1 / beta)  # Levy flight step
    return step

# Cuckoo Search Algorithm
def cuckoo_search(nests=10, max_iter=100, pa=0.25):
    # Step 1: Initialize nests (solutions)
    nests_positions = [random.uniform(-10, 10) for _ in range(nests)]
    fitness_values = [fitness(x) for x in nests_positions]
    
    # Best solution initially
    best_nest = nests_positions[fitness_values.index(min(fitness_values))]
    best_fitness = min(fitness_values)
    
    # Iteration loop
    for t in range(max_iter):
        # Generate new solutions via Levy flight
        new_nests = [nest + levy_flight() * (nest - best_nest) for nest in nests_positions]
        
        # Evaluate fitness of new solutions
        new_fitness = [fitness(x) for x in new_nests]
        
        # Update nests with better solutions
        for i in range(nests):
            if new_fitness[i] < fitness_values[i]:
                nests_positions[i] = new_nests[i]
                fitness_values[i] = new_fitness[i]
        
        # Abandon worst nests with probability Pa
        for i in range(nests):
            if random.random() < pa:
                nests_positions[i] = random.uniform(-10, 10)  # Generate a new random solution
                fitness_values[i] = fitness(nests_positions[i])
        
        # Update best solution
        current_best_nest = nests_positions[fitness_values.index(min(fitness_values))]
        current_best_fitness = min(fitness_values)
        
        if current_best_fitness < best_fitness:
            best_nest = current_best_nest
            best_fitness = current_best_fitness
        
        print(f"Iteration {t+1}, Best Fitness: {best_fitness}, Best Nest: {best_nest}")
    
    return best_nest, best_fitness

# Run the cuckoo search algorithm
best_solution, best_value = cuckoo_search(nests=10, max_iter=100, pa=0.25)
print(f"Global minimum found at x = {best_solution}, f(x) = {best_value}")
