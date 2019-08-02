# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.6
#   kernelspec:
#     display_name: Python [conda env:ea_thesis] *
#     language: python
#     name: conda-env-ea_thesis-py
# ---

# %% [raw]
# \author{Eloy Ruiz-Donayre}
# \title{TESTCASE B - One Parameter Set}
# \date{\today}
# \maketitle

# %% [raw]
# \tableofcontents

# %% [markdown]
# # Preliminaries

# %% [markdown]
# Importing python packages and setting display parameters

# %%
import math as mt
import random as rnd
import numpy as np

from deap import base, creator, tools

import numba
from numba import jit
import joblib

import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import pandas as pd
import statistics as stats

# %%
# %matplotlib inline
# #%config InlineBackend.figure_format = "retina"

plt.style.use("default")
plt.style.use("bmh")
# plt.rcParams.update({"figure.autolayout": True})
plt.rcParams["figure.figsize"] = (12, 9)
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.dpi"] = 100

# %%
pd.set_option("display.latex.repr", True)

# %%
pd.set_option("display.latex.longtable", True)

# %% [markdown] {"toc-hr-collapsed": false}
# # Fitness Landscape Definition

# %%
# Problem domain
x_min = -15
x_max = 15
y_min = -15
y_max = 15

# Known minimum
x_point = -1
y_point = -1


domain = (x_min, x_max, y_min, y_max)
point = (x_point, y_point)
img_size = (8.5, 4.25)

# Problem definition


@jit
def g_fun(x, y):
    mag = np.sqrt(x ** 2.0 + y ** 2.0)
    val = -(50.0 * np.sinc(mag / np.pi) - mag)
    return val.item()


@jit
def f_fun(x, y):
    D = 2
    alpha = 1 / 8

    x = (x - 5) / 6
    y = (y - 5) / 6

    a = np.abs(x ** 2 + y ** 2 - D) ** (alpha * D)
    b = (0.5 * (x ** 2 + y ** 2) + (x + y)) / D

    return a + b + 0.5


# %%
@jit
def evaluate(individual):
    x = individual[0]
    y = individual[1]
    fitness = f_fun(x, y)
    return (fitness,)


# %%
# Testing the minimum
print(f_fun(-1, -1))

# %%
# Testing the function
print(f_fun(-1.0, -1.0), f_fun(-11.0, -9.0), f_fun(11.0, 3.0), f_fun(-6.0, 9.0))

# %% [markdown] {"toc-hr-collapsed": false}
# # Running the Evolutionary Algorithm

# %% [markdown] {"toc-hr-collapsed": true}
# ## Setting the EA's parameters

# %% [markdown]
# From TLFL

# %%
# Algorithm parameters
# Number of replicates, and generations per experiment
rep_end = 40
births_end = 120e3

# Genes
gen_size = 2
# Population size
pop_size = 160
# Progeny and parents size
b_ratio = 2
par_size = b_ratio * pop_size

# Progeny parameters
## Crossover probability per gene
cx_pb = 0.3
## Mutation probability per gene
mut_pb = 0.5
## Mutation strength
mut_sig = 0.5

# Selection by tournament
# Tournament size parent selection
k_par = 4
# Tournament size survivor selection
k_sur = 6

# %% [markdown]
# ## Defining the EA elements

# %% [markdown]
# We define that the fitness is related to a minimizing problem, and that each individual is represented with a list of numbers

# %%
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# %% [markdown]
# We start the DEAP toolset. At creation, each individual will have 2 genes of type "float" that are randomly initialized in the range [-15; 15].

# %%
toolbox = base.Toolbox()

# %%
toolbox.register("attr_float", rnd.uniform, -15, 15)
toolbox.register(
    "individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=gen_size
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# %% [markdown]
# We add our landscape to the toolset, indicate that the mating will use a uniform crossover on a per-gene basis, that the mutation will be also on a per-gene basis with a value taken from a gaussian distribution, and that parent and survivor selections will use tournament selection.

# %%
toolbox.register("evaluate", evaluate)

toolbox.register("mate", tools.cxUniform)
toolbox.register("mutate", tools.mutGaussian, mu=0)

toolbox.register("par_select", tools.selTournament)
toolbox.register("sur_select", tools.selTournament)

# %% [markdown]
# We define that for each generation we'll summarize the fitnesses with median, mean, standard deviation, and store the best and worst fitnesses in the generation.

# %%
stat = tools.Statistics(key=lambda ind: ind.fitness.values[0])

stat.register("med", stats.median)
stat.register("avg", stats.mean)
stat.register("std", stats.stdev)
stat.register("best", min)
stat.register("worst", max)

# %% [markdown]
# We invoque the data recording logbook.

# %%
logbook = tools.Logbook()

# %% [markdown] {"toc-hr-collapsed": false}
# ## Single Run of the EA Experiments
# 1 Experiment
# L-> 1 Parameter set for the experiment.
# >L-> 1 Replicate.
# >>L-> The replicate is affected due to the randomness seed.

# %% [markdown] {"toc-hr-collapsed": false}
# ## 100 Executions of the EA
# 1 Experiment
# >L-> 1 Parameter set for the experiment.
# >>L-> 100 Replicate.
# >>>L-> Each replicate is different due to randomness effects.

# %% [markdown]
# ### Changing parameters

# %%
# Restarting seed
rnd.seed(42)


# %% [markdown]
# ### Execution

# %%
# %%time
logbook.clear()

if __name__ == "__main__":
    for rep_n in range(rep_end):
        rep_seed = rnd.randint(0, 999)
        rnd.seed(rep_seed)
        # We initialize the population and evaluate the individuals' fitnesses
        pop = toolbox.population(n=pop_size)
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        # We start the logbook
        record = stat.compile(pop)
        births = len(pop)
        logbook.record(
            rep=rep_n + 1, 
            seed=rep_seed, 
            births=births, 
            **record)

        while births < births_end:
            # Select Parents and clone them as base for offsprings
            parents = toolbox.par_select(pop, k=par_size, tournsize=k_par)
            offspring = [toolbox.clone(ind) for ind in parents]
            births = births + len(offspring)

            # Aplly crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                toolbox.mate(child1, child2, indpb=cx_pb)

            # Apply mutation
            for mutant in offspring:
                toolbox.mutate(mutant, sigma=mut_sig, indpb=mut_pb)
                del mutant.fitness.values

            fitnesses = toolbox.map(toolbox.evaluate, offspring)
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit

            pop = toolbox.sur_select((pop + offspring), k=pop_size, tournsize=k_sur)

            record = stat.compile(pop)
            logbook.record(
                rep=rep_n + 1, 
                seed=rep_seed, 
                births=births + 1, 
                **record)

# %% [markdown] {"toc-hr-collapsed": false}
# ### Data Analysis

# %% [markdown]
# We transform the records into a Data Frame

# %%
fitness_res = pd.DataFrame.from_dict(logbook)
fitness_res = fitness_res[
    ["rep", "seed", "births", "avg", "std", "med", "worst", "best"]
]
fitness_res.head()

# %% [markdown]
# We filter the values of the last generation

# %%
query = fitness_res["births"] >= births_end
fit_120k = fitness_res[query]
query = fitness_res["births"] == 60001
fit_60k = fitness_res[query]
query = fitness_res["births"] == 30241
fit_30k = fitness_res[query]

query_exact = fit_30k["best"] < 1e-6
fit_30k_exact = fit_30k[query_exact]
query_exact = fit_60k["best"] < 1e-6
fit_60k_exact = fit_60k[query_exact]
query_exact = fit_120k["best"] < 1e-6
fit_120k_exact = fit_120k[query_exact]


# %%
print(len(fit_30k))
print(len(fit_60k))
print(len(fit_120k))

print(len(fit_30k_exact))
print(len(fit_60k_exact))
print(len(fit_120k_exact))

# %%
print(fit_120k.best.min())
print(fit_120k.best.max())

# %%

# %%
## Mutation strength
mut_sig = 5e-4
# Restarting seed
rnd.seed(42)

# %%
# %%time
logbook.clear()

if __name__ == "__main__":
    for rep_n in range(rep_end):
        rep_seed = rnd.randint(0, 999)
        rnd.seed(rep_seed)
        # We initialize the population and evaluate the individuals' fitnesses
        pop = toolbox.population(n=pop_size)
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        # We start the logbook
        record = stat.compile(pop)
        births = len(pop)
        logbook.record(
            rep=rep_n + 1, 
            seed=rep_seed, 
            births=births, 
            **record)

        while births < births_end:
            # Select Parents and clone them as base for offsprings
            parents = toolbox.par_select(pop, k=par_size, tournsize=k_par)
            offspring = [toolbox.clone(ind) for ind in parents]
            births = births + len(offspring)

            # Aplly crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                toolbox.mate(child1, child2, indpb=cx_pb)

            # Apply mutation
            for mutant in offspring:
                toolbox.mutate(mutant, sigma=mut_sig, indpb=mut_pb)
                del mutant.fitness.values

            fitnesses = toolbox.map(toolbox.evaluate, offspring)
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit

            pop = toolbox.sur_select((pop + offspring), k=pop_size, tournsize=k_sur)

            record = stat.compile(pop)
            logbook.record(
                rep=rep_n + 1, 
                seed=rep_seed, 
                births=births + 1, 
                **record)

# %%
fitness_res = pd.DataFrame.from_dict(logbook)
fitness_res = fitness_res[
    ["rep", "seed", "births", "avg", "std", "med", "worst", "best"]
]
fitness_res.head()

# %%
query = fitness_res["births"] >= births_end
fit_120k = fitness_res[query]
query = fitness_res["births"] == 60001
fit_60k = fitness_res[query]
query = fitness_res["births"] == 30241
fit_30k = fitness_res[query]

query_exact = fit_30k["best"] < 1e-6
fit_30k_exact = fit_30k[query_exact]
query_exact = fit_60k["best"] < 1e-6
fit_60k_exact = fit_60k[query_exact]
query_exact = fit_120k["best"] < 1e-6
fit_120k_exact = fit_120k[query_exact]

# %%
print(len(fit_30k))
print(len(fit_60k))
print(len(fit_120k))

print(len(fit_30k_exact))
print(len(fit_60k_exact))
print(len(fit_120k_exact))

# %%
print(fit_120k.best.min())
print(fit_120k.best.max())

# %%

# %% [markdown]
# From OFAT

# %%
# Algorithm parameters
# Number of replicates, and generations per experiment
rep_end = 40
births_end = 120e3

# Genes
gen_size = 2
# Population size
pop_size = 160
# Progeny and parents size
b_ratio = 3
par_size = b_ratio * pop_size

# Progeny parameters
## Crossover probability per gene
cx_pb = 0.25
## Mutation probability per gene
mut_pb = 0.5
## Mutation strength
mut_sig = 1.25

# Selection by tournament
# Tournament size parent selection
k_par = 4
# Tournament size survivor selection
k_sur = 6

# %%
# Restarting seed
rnd.seed(42)

# %%
# %%time
logbook.clear()

if __name__ == "__main__":
    for rep_n in range(rep_end):
        rep_seed = rnd.randint(0, 999)
        rnd.seed(rep_seed)
        # We initialize the population and evaluate the individuals' fitnesses
        pop = toolbox.population(n=pop_size)
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        # We start the logbook
        record = stat.compile(pop)
        births = len(pop)
        logbook.record(
            rep=rep_n + 1, 
            seed=rep_seed, 
            births=births, 
            **record)

        while births < births_end:
            # Select Parents and clone them as base for offsprings
            parents = toolbox.par_select(pop, k=par_size, tournsize=k_par)
            offspring = [toolbox.clone(ind) for ind in parents]
            births = births + len(offspring)

            # Aplly crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                toolbox.mate(child1, child2, indpb=cx_pb)

            # Apply mutation
            for mutant in offspring:
                toolbox.mutate(mutant, sigma=mut_sig, indpb=mut_pb)
                del mutant.fitness.values

            fitnesses = toolbox.map(toolbox.evaluate, offspring)
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit

            pop = toolbox.sur_select((pop + offspring), k=pop_size, tournsize=k_sur)

            record = stat.compile(pop)
            logbook.record(
                rep=rep_n + 1, 
                seed=rep_seed, 
                births=births + 1, 
                **record)

# %%
fitness_res = pd.DataFrame.from_dict(logbook)
fitness_res = fitness_res[
    ["rep", "seed", "births", "avg", "std", "med", "worst", "best"]
]
fitness_res.head()

# %%
query = fitness_res["births"] >= births_end
fit_120k = fitness_res[query]
query = fitness_res["births"] == 60001
fit_60k = fitness_res[query]
query = fitness_res["births"] == 30241
fit_30k = fitness_res[query]

query_exact = fit_30k["best"] < 1e-6
fit_30k_exact = fit_30k[query_exact]
query_exact = fit_60k["best"] < 1e-6
fit_60k_exact = fit_60k[query_exact]
query_exact = fit_120k["best"] < 1e-6
fit_120k_exact = fit_120k[query_exact]

# %%
print(len(fit_30k))
print(len(fit_60k))
print(len(fit_120k))

print(len(fit_30k_exact))
print(len(fit_60k_exact))
print(len(fit_120k_exact))

# %%
print(fit_120k.best.min())
print(fit_120k.best.max())

# %%

# %%
## Mutation strength
mut_sig = 5e-4
# Restarting seed
rnd.seed(42)

# %%
# %%time
logbook.clear()

if __name__ == "__main__":
    for rep_n in range(rep_end):
        rep_seed = rnd.randint(0, 999)
        rnd.seed(rep_seed)
        # We initialize the population and evaluate the individuals' fitnesses
        pop = toolbox.population(n=pop_size)
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        # We start the logbook
        record = stat.compile(pop)
        births = len(pop)
        logbook.record(
            rep=rep_n + 1, 
            seed=rep_seed, 
            births=births, 
            **record)

        while births < births_end:
            # Select Parents and clone them as base for offsprings
            parents = toolbox.par_select(pop, k=par_size, tournsize=k_par)
            offspring = [toolbox.clone(ind) for ind in parents]
            births = births + len(offspring)

            # Aplly crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                toolbox.mate(child1, child2, indpb=cx_pb)

            # Apply mutation
            for mutant in offspring:
                toolbox.mutate(mutant, sigma=mut_sig, indpb=mut_pb)
                del mutant.fitness.values

            fitnesses = toolbox.map(toolbox.evaluate, offspring)
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit

            pop = toolbox.sur_select((pop + offspring), k=pop_size, tournsize=k_sur)

            record = stat.compile(pop)
            logbook.record(
                rep=rep_n + 1, 
                seed=rep_seed, 
                births=births + 1, 
                **record)

# %%
fitness_res = pd.DataFrame.from_dict(logbook)
fitness_res = fitness_res[
    ["rep", "seed", "births", "avg", "std", "med", "worst", "best"]
]
fitness_res.head()

# %%
query = fitness_res["births"] >= births_end
fit_120k = fitness_res[query]
query = fitness_res["births"] == 60001
fit_60k = fitness_res[query]
query = fitness_res["births"] == 30241
fit_30k = fitness_res[query]

query_exact = fit_30k["best"] < 1e-6
fit_30k_exact = fit_30k[query_exact]
query_exact = fit_60k["best"] < 1e-6
fit_60k_exact = fit_60k[query_exact]
query_exact = fit_120k["best"] < 1e-6
fit_120k_exact = fit_120k[query_exact]

# %%
print(len(fit_30k))
print(len(fit_60k))
print(len(fit_120k))

print(len(fit_30k_exact))
print(len(fit_60k_exact))
print(len(fit_120k_exact))

# %%
print(fit_120k.best.min())
print(fit_120k.best.max())

# %%

# %%

# %%

# %%
