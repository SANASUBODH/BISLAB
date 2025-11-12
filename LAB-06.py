import numpy as np
from multiprocessing import Pool

def objective(x):
    return np.sum(x ** 2)

def update(cell, neighbor, bounds):
    new_cell = cell + 0.1 * (neighbor - cell)
    new_cell = np.clip(new_cell, bounds[0], bounds[1])
    return new_cell

def parallel_cellular(dim=3, bounds=(-5,5), grid_shape=(4,4), max_iters=50):
    np.random.seed(0)
    grid = np.random.uniform(bounds[0], bounds[1], size=(grid_shape[0], grid_shape[1], dim))
    pool = Pool()
    
    for t in range(1, max_iters + 1):
        tasks = []
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                ni, nj = np.random.randint(0, grid_shape[0]), np.random.randint(0, grid_shape[1])
                tasks.append((grid[i, j], grid[ni, nj], bounds))
        results = pool.starmap(update, tasks)
        grid = np.array(results).reshape(grid_shape[0], grid_shape[1], dim)

        fits = np.array([[objective(grid[i,j]) for j in range(grid_shape[1])] for i in range(grid_shape[0])])
        best_idx = np.unravel_index(np.argmin(fits), fits.shape)
        best_fit = fits[best_idx]
        print(f"Iteration {t:3d}/{max_iters} | Best fitness = {best_fit:.6f}")

    pool.close(); pool.join()
    best_sol = grid[best_idx]
    return best_sol, best_fit

if __name__ == "__main__":
    best_sol, best_fit = parallel_cellular(dim=3, bounds=(-5,5), grid_shape=(4,4), max_iters=50)
    print("\nBest solution found:", best_sol)
    print("Best fitness:", best_fit)
