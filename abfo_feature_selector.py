import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

# --------------------------------------------------------------
# Utility: Ensure artifacts folder exists
# --------------------------------------------------------------
def ensure_artifacts():
    if not os.path.exists("artifacts"):
        os.makedirs("artifacts")
        print("[INFO] Created 'artifacts/' directory.")

# --------------------------------------------------------------
# Fitness evaluation function
# --------------------------------------------------------------
def compute_fitness(solution, X, y):
    """
    Computes fitness of a binary feature-selection solution.

    Fitness = Cross-validated accuracy on selected features
    """
    idx = np.where(solution == 1)[0]

    if len(idx) == 0:
        return 0  # invalid solution

    model = RandomForestClassifier()

    try:
        acc = cross_val_score(model, X[:, idx], y, cv=5).mean()
    except Exception:
        acc = 0

    return acc

# --------------------------------------------------------------
# Adaptive Bitterling Fish Optimization (ABFO)
# --------------------------------------------------------------
class ABFOFeatureSelector:

    def __init__(
        self,
        population_size=20,
        iterations=50,
        mutation_rate=0.1,
        adaptive_factor=0.9,
    ):
        self.population_size = population_size
        self.iterations = iterations
        self.mutation_rate = mutation_rate
        self.adaptive_factor = adaptive_factor

    # --------------------------------------------
    # Population initialization
    # --------------------------------------------
    def initialize_population(self, dim):
        return np.random.randint(0, 2, (self.population_size, dim))

    # --------------------------------------------
    # Mutation operator
    # --------------------------------------------
    def mutate(self, solution):
        mutant = solution.copy()
        for i in range(len(mutant)):
            if np.random.rand() < self.mutation_rate:
                mutant[i] = 1 - mutant[i]  # flip bit
        return mutant

    # --------------------------------------------
    # Random exploration step
    # --------------------------------------------
    def random_walk(self, solution):
        pos = solution.copy()
        random_idx = np.random.randint(0, len(solution))
        pos[random_idx] = 1 - pos[random_idx]
        return pos

    # --------------------------------------------
    # Exploitation step (attraction to best)
    # --------------------------------------------
    def attraction(self, solution, best_solution):
        new_sol = solution.copy()
        diff = best_solution - solution

        for i in range(len(solution)):
            if diff[i] != 0 and np.random.rand() < 0.5:
                new_sol[i] = best_solution[i]

        return new_sol

    # --------------------------------------------
    # Main ABFO Optimization
    # --------------------------------------------
    def optimize(self, X, y):
        dim = X.shape[1]
        population = self.initialize_population(dim)

        best_solution = None
        best_score = -1

        print("\n[INFO] Starting ABFO Feature Selection...\n")

        for iteration in tqdm(range(self.iterations)):

            for i in range(self.population_size):

                # Fitness of current
                score = compute_fitness(population[i], X, y)

                if score > best_score:
                    best_score = score
                    best_solution = population[i].copy()

                # Random exploration
                exploratory = self.random_walk(population[i])
                exp_score = compute_fitness(exploratory, X, y)

                if exp_score > score:
                    population[i] = exploratory

                # Attraction toward best
                attracted = self.attraction(population[i], best_solution)
                att_score = compute_fitness(attracted, X, y)

                if att_score > score:
                    population[i] = attracted

                # Mutation
                mutated = self.mutate(population[i])
                mut_score = compute_fitness(mutated, X, y)

                if mut_score > score:
                    population[i] = mutated

            # Adaptive mutation decrease
            self.mutation_rate *= self.adaptive_factor

        return best_solution, best_score

# --------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------
if __name__ == "__main__":
    ensure_artifacts()

    print("[INFO] Loading preprocessed dataset...")
    X_train = np.loadtxt("artifacts/X_train.csv", delimiter=",")
    y_train = np.loadtxt("artifacts/y_train.csv", delimiter=",").astype(int)

    selector = ABFOFeatureSelector(
        population_size=30,
        iterations=40,
        mutation_rate=0.12,
        adaptive_factor=0.92
    )

    best_features, score = selector.optimize(X_train, y_train)

    np.savetxt("artifacts/selected_features.txt", best_features, fmt="%d")

    print("\n======================================")
    print(" ABFO Feature Selection Completed")
    print(" Selected Features:", sum(best_features))
    print(" Best Fitness Score:", score)
    print("======================================\n")
