# Explanation
# Sample Data Creation: The script creates a sample dataset containing lease portfolio data with lease ID, lease cost, lease area, market value, and business priority. This dataset is saved as lease_portfolio.csv.
# Data Loading: The script reads the sample data from the CSV file.
# Genetic Algorithm Components: The script defines the components of the Genetic Algorithm, including the fitness function, individual creation, and genetic operators (crossover, mutation, and selection).
# Fitness Function: The fitness function evaluates individuals based on lease cost, lease area, market value, and business priority. The objective is to maximize value and priority while minimizing cost.
# Genetic Algorithm Execution: The script runs the Genetic Algorithm using the DEAP library, evolving a population of individuals over a specified number of generations to optimize the lease portfolio.
# Best Solution Output: The script outputs the best solution found by the Genetic Algorithm, including the selected leases.
# Performance Plot: The script plots the performance of the Genetic Algorithm over generations, showing the maximum and mean fitness values.
# Before running the script, ensure you have the necessary libraries installed:

# pip install pandas numpy matplotlib deap


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from deap import base, creator, tools, algorithms

# Sample data creation
data = {
    'Lease ID': ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10'],
    'Lease Cost': np.random.randint(1000, 5000, size=10),
    'Lease Area': np.random.randint(500, 2000, size=10),
    'Market Value': np.random.randint(2000, 6000, size=10),
    'Business Priority': np.random.randint(1, 10, size=10)
}

# Create DataFrame
data = pd.DataFrame(data)

# Save the sample data to a CSV file
data.to_csv('lease_portfolio.csv', index=False)

# Load the data
data = pd.read_csv('lease_portfolio.csv')

# Define Genetic Algorithm components
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def create_individual():
    return [random.randint(0, 1) for _ in range(len(data))]

def evaluate(individual):
    cost = np.sum(individual * data['Lease Cost'])
    area = np.sum(individual * data['Lease Area'])
    value = np.sum(individual * data['Market Value'])
    priority = np.sum(individual * data['Business Priority'])
    
    # Objective: maximize value and priority, minimize cost
    return value + priority - cost,

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Genetic Algorithm
def run_ga(pop_size=100, cx_prob=0.5, mut_prob=0.2, n_gen=50):
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cx_prob, mutpb=mut_prob, ngen=n_gen, stats=stats, halloffame=hof, verbose=True)
    
    return pop, log, hof

# Run Genetic Algorithm
pop, log, hof = run_ga()

# Best solution
best_solution = hof[0]
best_solution_indices = [i for i, x in enumerate(best_solution) if x == 1]

# Output best solution
print("Best solution indices:", best_solution_indices)
print("Selected leases:")
print(data.iloc[best_solution_indices])

# Plotting the log
gen = log.select("gen")
max_fitness_values = log.select("max")
mean_fitness_values = log.select("avg")

plt.plot(gen, max_fitness_values, label="Max Fitness")
plt.plot(gen, mean_fitness_values, label="Mean Fitness")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend()
plt.grid(True)
plt.title("Genetic Algorithm Performance")
plt.show()
