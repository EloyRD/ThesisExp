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
# \title{TESTCASE B - One-Factor-at-a-Time Analysis}
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
import itertools as it

from deap import base, creator, tools

import numba
from numba import jit
import joblib

import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import statistics as stats

import thesis_EAfunc as EAf
import thesis_visfunc as EAv


# %%
plt.style.use("bmh")
plt.rcParams.update({"figure.autolayout": True})
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

pd.set_option("display.latex.repr", True)
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
x_point = -6.01717
y_point = 9.06022

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
    x_min = -6.01717
    y_min = 9.06022
    f_min = (
        g_fun(x_min + 11.0, y_min + 9.0)
        + g_fun(x_min - 11.0, y_min - 3.0)
        + g_fun(x_min + 6.0, y_min - 9.0)
    )
    tripsinc = (
        g_fun(x + 11.0, y + 9.0)
        + g_fun(x - 11.0, y - 3.0)
        + g_fun(x + 6.0, y - 9.0)
        - (f_min)
    )
    return tripsinc


# %%
@jit
def evaluate(individual):
    x = individual[0]
    y = individual[1]
    fitness = f_fun(x, y)
    return (fitness,)


# %%
# Testing the minimum
print(f_fun(-6.01717, 9.06022))

# %%
# Testing the function
print(f_fun(-1.0, -1.0), f_fun(-11.0, -9.0), f_fun(11.0, 3.0), f_fun(-6.0, 9.0))

# %% [markdown] {"toc-hr-collapsed": false}
# # Setting up the experiment

# %% [markdown] {"toc-hr-collapsed": false}
# ## Common parameters

# %%
# Algorithm parameters
# Number of replicates, and generations per experiment
rep_end = 40
gen_end = 5000

# Genes
gen_size = 2
# Population size
pop_size_lvl = [20, 10, 40, 80, 160]
# Progeny and parents size ratio to population size
b_ratio_lvl = [3, 0.5, 1, 2, 5]

# Progeny parameters
## Crossover probability per gene
cx_pb_lvl = [0.5, 0.1, 0.25, 0.75, 0.9]
## Mutation probability per gene
mut_pb_lvl = [0.5, 0.1, 0.25, 0.75, 0.9]
## Mutation strength
mut_sig_lvl = [2.5, 0.5, 1.25, 5, 7.5, 10]

# Selection by tournament
# Tournament size parent selection
k_par_lvl = [2, 1, 4, 6, 7]
# Tournament size survivor selection
k_sur_lvl = [6, 1, 3, 4, 7]

# %% [markdown]
# ### Factor levels

# %%
inputs_labels = {
    "pop_size": "Population size",
    "b_ratio": "Progeny-to-pop ratio",
    "cx_pb": "Crossover prob",
    "mut_pb": "Mutation prob",
    "mut_sig": "Mutation sigma",
    "k_par": "Parent tourn size",
    "k_sur": "Surviv tourn size",
}

dat = [
    ("pop_size", "Integer +", pop_size_lvl, pop_size_lvl[0]),
    ("b_ratio", "Real +", b_ratio_lvl, b_ratio_lvl[0]),
    ("cx_pb", "Real [0,1]", cx_pb_lvl, cx_pb_lvl[0]),
    ("mut_pb", "Real [0,1]", mut_pb_lvl, mut_pb_lvl[0]),
    ("mut_sig", "Real +", mut_sig_lvl, mut_sig_lvl[0]),
    ("k_par", "Integer +", k_par_lvl, k_par_lvl[0]),
    ("k_sur", "Integer +", k_sur_lvl, k_sur_lvl[0]),
]

inputs_df = pd.DataFrame(dat, columns=["Factor", "Range", "Levels", "Default"])
inputs_df = inputs_df.set_index(["Factor"])
inputs_df["Label"] = inputs_df.index.map(lambda z: inputs_labels[z])
inputs_df = inputs_df[["Label", "Range", "Levels", "Default"]]

# %% [markdown] {"toc-hr-collapsed": false}
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
# ## Experiment description

# %% [markdown]
# 7 Experiment
# >L-> In each experiment, one factor will be varied through its levels or categories. Other factors stay at default value.
# >>L-> 40 Replicate per experiment.
# >>>L-> Each replicate is different due to randomness effects.

# %% [markdown] {"toc-hr-collapsed": false}
# # Running the experiments

# %% [markdown] {"toc-hr-collapsed": false}
# ## 1st experiment: Varying population size (Factor pop_size)

# %% [markdown]
# Experiment number

# %%
exp_n = 1

# %% [markdown]
# We create all the possible combinations of the experiment's dynamic factor levels with the static factors default values

# %%
exp_par = list(
    it.product(
        pop_size_lvl,
        [b_ratio_lvl[0]],
        [cx_pb_lvl[0]],
        [mut_pb_lvl[0]],
        [mut_sig_lvl[0]],
        [k_par_lvl[0]],
        [k_sur_lvl[0]],
    )
)
exp_par

# %% [markdown]
# ### Executing the experiment

# %%
# %%time

if __name__ == "__main__":
    for (pop_size, b_ratio, cx_pb, mut_pb, mut_sig, k_par, k_sur) in exp_par:
        par_size = b_ratio * pop_size
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
            logbook.record(
                exp=exp_n,
                pop=pop_size,
                b_ratio=b_ratio,
                cx_pb=cx_pb,
                mut_pb=mut_pb,
                mut_sig=mut_sig,
                k_par=k_par,
                k_sur=k_sur,
                rep=rep_n + 1,
                seed=rep_seed,
                gen=0,
                **record
            )

            for gen_n in range(gen_end):
                # Select Parents and clone them as base for offsprings
                parents = toolbox.par_select(pop, k=par_size, tournsize=k_par)
                offspring = [toolbox.clone(ind) for ind in pop]

                # Aplly crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    toolbox.mate(child1, child2, indpb=cx_pb)
                    del child1.fitness.values, child2.fitness.values

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
                    exp=exp_n,
                    pop=pop_size,
                    b_ratio=b_ratio,
                    cx_pb=cx_pb,
                    mut_pb=mut_pb,
                    mut_sig=mut_sig,
                    k_par=k_par,
                    k_sur=k_sur,
                    rep=rep_n + 1,
                    seed=rep_seed,
                    gen=gen_n + 1,
                    **record
                )

# %% [markdown]
# Storing the logbook to a file

# %%
pickle_file = "./pickle/C_02.joblib"

# %%
with open(pickle_file, "wb") as handle:
    joblib.dump(logbook, handle, compress="xz")

# %% [markdown] {"toc-hr-collapsed": true}
# ## 2nd experiment: Varying progeny-to-population ratio (Factor b_ratio)

# %% [markdown]
# Experiment number

# %%
exp_n = 2

# %% [markdown]
# We create all the possible combinations of the experiment's dynamic factor levels with the static factors default values

# %%
exp_par = list(
    it.product(
        [pop_size_lvl[0]],
        b_ratio_lvl,
        [cx_pb_lvl[0]],
        [mut_pb_lvl[0]],
        [mut_sig_lvl[0]],
        [k_par_lvl[0]],
        [k_sur_lvl[0]],
    )
)
exp_par

# %% [markdown]
# ### Executing the experiment

# %%
# %%time

if __name__ == "__main__":
    for (pop_size, b_ratio, cx_pb, mut_pb, mut_sig, k_par, k_sur) in exp_par:
        par_size = b_ratio * pop_size
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
            logbook.record(
                exp=exp_n,
                pop=pop_size,
                b_ratio=b_ratio,
                cx_pb=cx_pb,
                mut_pb=mut_pb,
                mut_sig=mut_sig,
                k_par=k_par,
                k_sur=k_sur,
                rep=rep_n + 1,
                seed=rep_seed,
                gen=0,
                **record
            )

            for gen_n in range(gen_end):
                # Select Parents and clone them as base for offsprings
                parents = toolbox.par_select(pop, k=par_size, tournsize=k_par)
                offspring = [toolbox.clone(ind) for ind in pop]

                # Aplly crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    toolbox.mate(child1, child2, indpb=cx_pb)
                    del child1.fitness.values, child2.fitness.values

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
                    exp=exp_n,
                    pop=pop_size,
                    b_ratio=b_ratio,
                    cx_pb=cx_pb,
                    mut_pb=mut_pb,
                    mut_sig=mut_sig,
                    k_par=k_par,
                    k_sur=k_sur,
                    rep=rep_n + 1,
                    seed=rep_seed,
                    gen=gen_n + 1,
                    **record
                )

# %% [markdown]
# Storing the logbook to a file

# %%
with open(pickle_file, "wb") as handle:
    joblib.dump(logbook, handle, compress="xz")

# %% [markdown] {"toc-hr-collapsed": true}
# ## 3rd experiment: Varying crossover probability (Factor cx_pb)

# %% [markdown]
# Experiment number

# %%
exp_n = 3

# %% [markdown]
# We create all the possible combinations of the experiment's dynamic factor levels with the static factors default values

# %%
exp_par = list(
    it.product(
        [pop_size_lvl[0]],
        [b_ratio_lvl[0]],
        cx_pb_lvl,
        [mut_pb_lvl[0]],
        [mut_sig_lvl[0]],
        [k_par_lvl[0]],
        [k_sur_lvl[0]],
    )
)
exp_par

# %% [markdown]
# ### Executing the experiment

# %%
# %%time

if __name__ == "__main__":
    for (pop_size, b_ratio, cx_pb, mut_pb, mut_sig, k_par, k_sur) in exp_par:
        par_size = b_ratio * pop_size
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
            logbook.record(
                exp=exp_n,
                pop=pop_size,
                b_ratio=b_ratio,
                cx_pb=cx_pb,
                mut_pb=mut_pb,
                mut_sig=mut_sig,
                k_par=k_par,
                k_sur=k_sur,
                rep=rep_n + 1,
                seed=rep_seed,
                gen=0,
                **record
            )

            for gen_n in range(gen_end):
                # Select Parents and clone them as base for offsprings
                parents = toolbox.par_select(pop, k=par_size, tournsize=k_par)
                offspring = [toolbox.clone(ind) for ind in pop]

                # Aplly crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    toolbox.mate(child1, child2, indpb=cx_pb)
                    del child1.fitness.values, child2.fitness.values

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
                    exp=exp_n,
                    pop=pop_size,
                    b_ratio=b_ratio,
                    cx_pb=cx_pb,
                    mut_pb=mut_pb,
                    mut_sig=mut_sig,
                    k_par=k_par,
                    k_sur=k_sur,
                    rep=rep_n + 1,
                    seed=rep_seed,
                    gen=gen_n + 1,
                    **record
                )

# %% [markdown]
# Storing the logbook to a file

# %%
with open(pickle_file, "wb") as handle:
    joblib.dump(logbook, handle, compress="xz")

# %% [markdown] {"toc-hr-collapsed": true}
# ## 4th experiment: Varying mutation probability (Factor mut_pb)

# %% [markdown]
# Experiment number

# %%
exp_n = 4

# %% [markdown]
# We create all the possible combinations of the experiment's dynamic factor levels with the static factors default values

# %%
exp_par = list(
    it.product(
        [pop_size_lvl[0]],
        [b_ratio_lvl[0]],
        [cx_pb_lvl[0]],
        mut_pb_lvl,
        [mut_sig_lvl[0]],
        [k_par_lvl[0]],
        [k_sur_lvl[0]],
    )
)
exp_par

# %% [markdown]
# ### Executing the experiment

# %%
# %%time

if __name__ == "__main__":
    for (pop_size, b_ratio, cx_pb, mut_pb, mut_sig, k_par, k_sur) in exp_par:
        par_size = b_ratio * pop_size
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
            logbook.record(
                exp=exp_n,
                pop=pop_size,
                b_ratio=b_ratio,
                cx_pb=cx_pb,
                mut_pb=mut_pb,
                mut_sig=mut_sig,
                k_par=k_par,
                k_sur=k_sur,
                rep=rep_n + 1,
                seed=rep_seed,
                gen=0,
                **record
            )

            for gen_n in range(gen_end):
                # Select Parents and clone them as base for offsprings
                parents = toolbox.par_select(pop, k=par_size, tournsize=k_par)
                offspring = [toolbox.clone(ind) for ind in pop]

                # Aplly crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    toolbox.mate(child1, child2, indpb=cx_pb)
                    del child1.fitness.values, child2.fitness.values

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
                    exp=exp_n,
                    pop=pop_size,
                    b_ratio=b_ratio,
                    cx_pb=cx_pb,
                    mut_pb=mut_pb,
                    mut_sig=mut_sig,
                    k_par=k_par,
                    k_sur=k_sur,
                    rep=rep_n + 1,
                    seed=rep_seed,
                    gen=gen_n + 1,
                    **record
                )

# %% [markdown]
# Storing the logbook to a file

# %%
with open(pickle_file, "wb") as handle:
    joblib.dump(logbook, handle, compress="xz")

# %% [markdown] {"toc-hr-collapsed": true}
# ## 5th experiment: Varying mutation strength (Factor mut_sig)

# %% [markdown]
# Experiment number

# %%
exp_n = 5

# %% [markdown]
# We create all the possible combinations of the experiment's dynamic factor levels with the static factors default values

# %%
exp_par = list(
    it.product(
        [pop_size_lvl[0]],
        [b_ratio_lvl[0]],
        [cx_pb_lvl[0]],
        [mut_pb_lvl[0]],
        mut_sig_lvl,
        [k_par_lvl[0]],
        [k_sur_lvl[0]],
    )
)
exp_par

# %% [markdown]
# ### Executing the experiment

# %%
# %%time

if __name__ == "__main__":
    for (pop_size, b_ratio, cx_pb, mut_pb, mut_sig, k_par, k_sur) in exp_par:
        par_size = b_ratio * pop_size
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
            logbook.record(
                exp=exp_n,
                pop=pop_size,
                b_ratio=b_ratio,
                cx_pb=cx_pb,
                mut_pb=mut_pb,
                mut_sig=mut_sig,
                k_par=k_par,
                k_sur=k_sur,
                rep=rep_n + 1,
                seed=rep_seed,
                gen=0,
                **record
            )

            for gen_n in range(gen_end):
                # Select Parents and clone them as base for offsprings
                parents = toolbox.par_select(pop, k=par_size, tournsize=k_par)
                offspring = [toolbox.clone(ind) for ind in pop]

                # Aplly crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    toolbox.mate(child1, child2, indpb=cx_pb)
                    del child1.fitness.values, child2.fitness.values

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
                    exp=exp_n,
                    pop=pop_size,
                    b_ratio=b_ratio,
                    cx_pb=cx_pb,
                    mut_pb=mut_pb,
                    mut_sig=mut_sig,
                    k_par=k_par,
                    k_sur=k_sur,
                    rep=rep_n + 1,
                    seed=rep_seed,
                    gen=gen_n + 1,
                    **record
                )

# %% [markdown]
# Storing the logbook to a file

# %%
with open(pickle_file, "wb") as handle:
    joblib.dump(logbook, handle, compress="xz")

# %% [markdown] {"toc-hr-collapsed": true}
# ## 6th experiment: Varying parent selection (Factor k_par)

# %% [markdown]
# Experiment number

# %%
exp_n = 6

# %% [markdown]
# We create all the possible combinations of the experiment's dynamic factor levels with the static factors default values

# %%
exp_par = list(
    it.product(
        [pop_size_lvl[0]],
        [b_ratio_lvl[0]],
        [cx_pb_lvl[0]],
        [mut_pb_lvl[0]],
        [mut_sig_lvl[0]],
        k_par_lvl,
        [k_sur_lvl[0]],
    )
)
exp_par

# %% [markdown]
# ### Executing the experiment

# %%
# %%time

if __name__ == "__main__":
    for (pop_size, b_ratio, cx_pb, mut_pb, mut_sig, k_par, k_sur) in exp_par:
        par_size = b_ratio * pop_size
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
            logbook.record(
                exp=exp_n,
                pop=pop_size,
                b_ratio=b_ratio,
                cx_pb=cx_pb,
                mut_pb=mut_pb,
                mut_sig=mut_sig,
                k_par=k_par,
                k_sur=k_sur,
                rep=rep_n + 1,
                seed=rep_seed,
                gen=0,
                **record
            )

            for gen_n in range(gen_end):
                # Select Parents and clone them as base for offsprings
                parents = toolbox.par_select(pop, k=par_size, tournsize=k_par)
                offspring = [toolbox.clone(ind) for ind in pop]

                # Aplly crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    toolbox.mate(child1, child2, indpb=cx_pb)
                    del child1.fitness.values, child2.fitness.values

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
                    exp=exp_n,
                    pop=pop_size,
                    b_ratio=b_ratio,
                    cx_pb=cx_pb,
                    mut_pb=mut_pb,
                    mut_sig=mut_sig,
                    k_par=k_par,
                    k_sur=k_sur,
                    rep=rep_n + 1,
                    seed=rep_seed,
                    gen=gen_n + 1,
                    **record
                )

# %% [markdown]
# Storing the logbook to a file

# %%
with open(pickle_file, "wb") as handle:
    joblib.dump(logbook, handle, compress="xz")

# %% [markdown] {"toc-hr-collapsed": true}
# ## 7th experiment: Varying survivor selection (Factor k_sur)

# %% [markdown]
# Experiment number

# %%
exp_n = 7

# %% [markdown]
# We create all the possible combinations of the experiment's dynamic factor levels with the static factors default values

# %%
exp_par = list(
    it.product(
        [pop_size_lvl[0]],
        [b_ratio_lvl[0]],
        [cx_pb_lvl[0]],
        [mut_pb_lvl[0]],
        [mut_sig_lvl[0]],
        [k_par_lvl[0]],
        k_sur_lvl,
    )
)
exp_par

# %% [markdown]
# ### Executing the experiment

# %%
# %%time

if __name__ == "__main__":
    for (pop_size, b_ratio, cx_pb, mut_pb, mut_sig, k_par, k_sur) in exp_par:
        par_size = b_ratio * pop_size
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
            logbook.record(
                exp=exp_n,
                pop=pop_size,
                b_ratio=b_ratio,
                cx_pb=cx_pb,
                mut_pb=mut_pb,
                mut_sig=mut_sig,
                k_par=k_par,
                k_sur=k_sur,
                rep=rep_n + 1,
                seed=rep_seed,
                gen=0,
                **record
            )

            for gen_n in range(gen_end):
                # Select Parents and clone them as base for offsprings
                parents = toolbox.par_select(pop, k=par_size, tournsize=k_par)
                offspring = [toolbox.clone(ind) for ind in pop]

                # Aplly crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    toolbox.mate(child1, child2, indpb=cx_pb)
                    del child1.fitness.values, child2.fitness.values

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
                    exp=exp_n,
                    pop=pop_size,
                    b_ratio=b_ratio,
                    cx_pb=cx_pb,
                    mut_pb=mut_pb,
                    mut_sig=mut_sig,
                    k_par=k_par,
                    k_sur=k_sur,
                    rep=rep_n + 1,
                    seed=rep_seed,
                    gen=gen_n + 1,
                    **record
                )

# %% [markdown]
# Storing the logbook to a file

# %%
with open(pickle_file, "wb") as handle:
    joblib.dump(logbook, handle, compress="xz")

# %% [markdown]
# # Data Analysis

# %% [markdown]
# Reading the logbook from the file

# %%
# %%time
with open(pickle_file, "rb") as handle:
    logbook = joblib.load(handle)

# %% [markdown]
# We transform the records into a Data Frame

# %%
pop_records = [record for record in logb]
fitness_res = pd.DataFrame.from_dict(pop_records)

# %% [markdown]
# We filter the values of the last generation of each experiment into a new Data Frame

# %%
last_gen = fitness_res["gen"].max()
query = fitness_res["gen"] == last_gen
fin_fit_res = fitness_res[query]

# %% [markdown]
# # Visualization

# %% [markdown]
# Factors to iterate in the visualization

# %%
factors = ["pop_s"] + ["b"] + ["mut_p"] + ["mut_s"] + ["p_sel"] + ["s_sel"]
factors

# %% [markdown]
# Development of average minimum (best) fitness for each experiment (Each experiment has one factor at different levels)

# %%
# %%time
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 4), sharey=True, sharex=True)
palette = it.cycle(sns.color_palette("tab10"))

for ax, i in zip(axs.flatten(), range(6)):
    query = multi_fit["exp"] == (i + 1)
    c = next(palette)
    sns.lineplot(
        x="generation",
        y="fitness_min",
        label=("Average of: " + factors[i]),
        color=c,
        data=multi_fit[query],
        ax=ax,
    )
    ax.legend(facecolor="white")

axs[0, 0].set_ylim((2, 5.5))
axs[0, 0].set_xlim((0, 200))

plt.tight_layout()
plt.show()

# %% [markdown]
# Development of minimum (best) fitness for each level of each experiment

# %%
# %%time
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 4), sharey=True, sharex=True)

for ax, i in zip(axs.flatten(), range(6)):
    query = multi_fit["exp"] == (i + 1)
    if i in [4, 5]:
        palette = sns.color_palette("tab10", 4)
    elif i == 3:
        palette = sns.color_palette("tab10", 6)
    else:
        palette = sns.color_palette("tab10", 5)
    sns.lineplot(
        x="generation",
        y="fitness_min",
        style=factors[i],
        hue=factors[i],
        palette=palette,
        data=multi_fit[query],
        ax=ax,
    )
    ax.legend(facecolor="white", fontsize="x-small", ncol=2)

axs[0, 0].set_ylim((1.5, 4))
axs[0, 0].set_xlim((0, 200))

plt.tight_layout()
plt.show()

# %% [markdown]
# Average of minimum (best) fitness for each level of each experiment at 50, 100 and 200 generations

# %%
# %%time
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 4), sharey=True)

for ax, i in zip(axs.flatten(), range(6)):
    query = (multi_fit["exp"] == (i + 1)) & (
        (multi_fit["generation"] == gen_f)
        | (multi_fit["generation"] == (gen_f - 100))
        | (multi_fit["generation"] == (gen_f - 150))
    )
    palette = sns.color_palette("tab10", 3)
    sns.lineplot(
        x=factors[i],
        y="fitness_min",
        style="generation",
        hue="generation",
        markers=True,
        palette=palette,
        data=multi_fit[query],
        ax=ax,
    )
    ax.legend(facecolor="white", fontsize="x-small")
    if i == 0:
        ax.xaxis.set_ticks(pop_s_levels)
    elif i == 1:
        ax.xaxis.set_ticks(b_levels)
    elif i == 2:
        ax.xaxis.set_ticks(mut_p_levels)
    elif i == 3:
        ax.xaxis.set_ticks(mut_s_levels)
    elif i == 4 or i == 5:
        labels = ["fit_prop", "tourn_3", "trunc", "unif"]
        ax.set_xticklabels(labels)

axs[0, 0].set_ylim((0, 5))

plt.tight_layout()
plt.show()

# %%
