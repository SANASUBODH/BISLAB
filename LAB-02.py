import random

def objective(pos):
    return pos[0]**2 + pos[1]**2

# PSO parameters
num_particles = 3
num_iterations = 2
w = 0.5      # inertia
c1 = 1.0    # cognitive
c2 = 1.0    # social
r1 = r2 = 0.5  # fixed random factors for simplicity

# Initial positions and velocities
positions = [[2, 3], [-1, 4], [0, -2]]
velocities = [[0, 0] for _ in range(num_particles)]
pbests = [pos[:] for pos in positions]
pbest_scores = [objective(p) for p in positions]
gbest = pbests[pbest_scores.index(min(pbest_scores))][:]
gbest_score = min(pbest_scores)

for it in range(num_iterations):
    for i in range(num_particles):
        for d in range(2):  # for x and y
            velocities[i][d] = (
                w * velocities[i][d]
                + c1 * r1 * (pbests[i][d] - positions[i][d])
                + c2 * r2 * (gbest[d] - positions[i][d])
            )
            positions[i][d] += velocities[i][d]
        score = objective(positions[i])
        if score < pbest_scores[i]:
            pbests[i] = positions[i][:]
            pbest_scores[i] = score
    gbest = pbests[pbest_scores.index(min(pbest_scores))][:]
    gbest_score = min(pbest_scores)
    print(f"Iteration {it+1}: gbest position = {gbest}, gbest score = {gbest_score}")
