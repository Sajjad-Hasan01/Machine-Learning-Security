import numpy as np

# Define the fitness function
def fitness_function(solution):
    # Calculate the fitness value of a solution
    # This can be any function that evaluates the quality of a solution
    # The higher the fitness value, the better the solution
    return np.sum(solution)

# Create the initial population
def create_population(population_size, solution_size):
    # Generate a random population of solutions
    population = np.random.randint(2, size=(population_size, solution_size))
    return population


# Select parents for mating
def selection(population, fitness_values):
    # Perform roulette wheel selection
    probabilities = fitness_values / np.sum(fitness_values)
    selected_indices = np.random.choice(range(len(population)), size=len(population), p=probabilities)
    selected_population = population[selected_indices]
    return selected_population


# Perform crossover between parents
def crossover(parents):
    # Perform single-point crossover
    offspring = []
    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        parent2 = parents[i + 1]
        crossover_point = np.random.randint(1, len(parent1))
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        offspring.extend([child1, child2])
    return np.array(offspring)



# Perform mutation on offspring
def mutation(offspring, mutation_rate):
    # Perform bit-flip mutation
    for i in range(len(offspring)):
        # populatin
        for j in range(len(offspring[i])):
            # iterates over each gene in the current individual offspring[i].
            if np.random.rand() < mutation_rate:
                # np.random.rand generates a random number between
                # 0 and 1
                #  checks if the random number is less than the mutation rate. If it is, the gene will be mutated;
                #  otherwise, it remains unchanged.
                offspring[i][j] = 1 - offspring[i][j]
                """ return offspring returns the mutated population after all individuals have been processed.
Overall, this function loops through each individual in the population and for each gene, it randomly decides
 whether to flip the bit (0 to 1 or 1 to 0) based on the mutation rate provided
    return offspring"""


# Genetic Algorithm
def genetic_algorithm(population_size, solution_size, generations, mutation_rate):
    population = create_population(population_size, solution_size)
    # initializes a population of a specified size (population_size) with individuals of a specified size (solution_size).
    # This is typically done randomly or based on some initialization strategy
    for _ in range(generations):
        # iterates over the specified number of generations.
        fitness_values = np.array([fitness_function(solution) for solution in population])
        # calculates the fitness value for each individual in the population using a fitness_function
        parents = selection(population, fitness_values)
        offspring = crossover(parents)
        # applies mutation to the offspring population using the provided mutation rate.
        mutated_offspring = mutation(offspring, mutation_rate)
        population = mutated_offspring
    best_solution = population[np.argmax(fitness_values)]
    return best_solution


# Example usage
population_size = 100
solution_size = 10
generations = 100
mutation_rate = 0.01

best_solution = genetic_algorithm(population_size, solution_size, generations, mutation_rate)
print("Best solution:", best_solution)
print("Fitness value:", fitness_function(best_solution))