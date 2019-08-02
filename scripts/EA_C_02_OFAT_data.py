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

# %%
# #!pip install deap

# %%
# from google.colab import drive
# drive.mount('/content/gdrive')

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


# %%
# %matplotlib inline
# #%config InlineBackend.figure_format = "retina"

plt.style.use("bmh")
plt.rcParams.update({"figure.autolayout": True})
plt.rcParams["figure.figsize"] = (12, 9)
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.dpi"] = 100

pd.set_option("display.latex.repr", True)
pd.set_option("display.latex.longtable", True)

# %%
pickle_dir = "./pickle/"
file_sufix = "C_02"

# pickle_dir = '/content/gdrive/My Drive/Colab Notebooks/thesis/'

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

# %% [markdown] {"toc-hr-collapsed": true}
# # Setting up the experiment

# %% [markdown] {"toc-hr-collapsed": false}
# ## Common parameters

# %%
# Algorithm parameters
# Number of replicates, and generations per experiment
rep_end = 40
births_end = 120e3

# Genes
gen_size = 2
# Population size
pop_size_lvl = [20, 10, 40, 80, 160]
# Progeny and parents size ratio to population size
b_ratio_lvl = [3, 0.6, 1, 2, 5]

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
factors_levels = [
    ("pop", "Population size", "Integer +", pop_size_lvl, pop_size_lvl[0]),
    ("b_ratio", "Progeny-to-pop ratio", "Real +", b_ratio_lvl, b_ratio_lvl[0]),
    ("cx_pb", "Crossover prob", "Real [0,1]", cx_pb_lvl, cx_pb_lvl[0]),
    ("mut_pb", "Mutation prob", "Real [0,1]", mut_pb_lvl, mut_pb_lvl[0]),
    ("mut_sig", "Mutation sigma", "Real +", mut_sig_lvl, mut_sig_lvl[0]),
    ("k_par", "Parent tourn size", "Integer +", k_par_lvl, k_par_lvl[0]),
    ("k_sur", "Surviv tourn size", "Integer +", k_sur_lvl, k_sur_lvl[0]),
]

factors_df = pd.DataFrame(
    factors_levels, columns=["Factor", "Label", "Range", "Levels", "Default"]
)
factors_df = factors_df.set_index(["Factor"])

factors_df

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

# %% [markdown] {"toc-hr-collapsed": true}
# # Running the experiments

# %%
rnd.seed(42)

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
            births = len(pop)
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
                births=births,
                **record
            )

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
                    births=births,
                    **record
                )

# %% [markdown]
# Storing the logbook to a file

# %%
pickle_file = pickle_dir + file_sufix + ".joblib"

# %%
# %%time
with open(pickle_file, "wb") as handle:
    joblib.dump(logbook, handle, compress=("xz", 2))

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
            births = len(pop)
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
                births=births,
                **record
            )

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
                    births=births,
                    **record
                )

# %% [markdown]
# Storing the logbook to a file

# %%
# %%time
with open(pickle_file, "wb") as handle:
    joblib.dump(logbook, handle, compress=("xz", 2))

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
            births = len(pop)
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
                births=births,
                **record
            )

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
                    births=births,
                    **record
                )

# %% [markdown]
# Storing the logbook to a file

# %%
# %%time
with open(pickle_file, "wb") as handle:
    joblib.dump(logbook, handle, compress=("xz", 2))

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
            births = len(pop)
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
                births=births,
                **record
            )

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
                    births=births,
                    **record
                )

# %% [markdown]
# Storing the logbook to a file

# %%
# %%time
with open(pickle_file, "wb") as handle:
    joblib.dump(logbook, handle, compress=("xz", 2))

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
            births = len(pop)
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
                births=births,
                **record
            )

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
                    births=births,
                    **record
                )

# %% [markdown]
# Storing the logbook to a file

# %%
# %%time
with open(pickle_file, "wb") as handle:
    joblib.dump(logbook, handle, compress=("xz", 2))

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
            births = len(pop)
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
                births=births,
                **record
            )

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
                    births=births,
                    **record
                )

# %% [markdown]
# Storing the logbook to a file

# %%
# %%time
with open(pickle_file, "wb") as handle:
    joblib.dump(logbook, handle, compress=("xz", 2))

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
            births = len(pop)
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
                births=births,
                **record
            )

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
                    births=births,
                    **record
                )

# %% [markdown]
# Storing the logbook to a file

# %%
# %%time
with open(pickle_file, "wb") as handle:
    joblib.dump(logbook, handle, compress=("xz", 2))

# %% [markdown] {"colab_type": "text", "id": "pJSLtKCLr_jE"}
# # Data Storage

# %% [markdown] {"colab_type": "text", "id": "WBbPOEflr_jF"}
# Reading the logbook from the file

# %% {"colab": {}, "colab_type": "code", "id": "p6TRyN7waarQ"}
pickle_file = pickle_dir + file_sufix + ".joblib"

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 53}, "colab_type": "code", "executionInfo": {"elapsed": 217219, "status": "ok", "timestamp": 1561378312899, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "qe9aP-ykr_jG", "outputId": "7f15b958-a23f-475d-fc76-92dd8cf287cf"}
# %%time
with open(pickle_file, "rb") as handle:
    logbook = joblib.load(handle)

# %% [markdown] {"colab_type": "text", "id": "H5KaYfLUMEe3"}
# ## Storing DataFrame of the Logbook

# %% [markdown] {"colab_type": "text", "id": "LGtbsGKhegPT"}
# We transform the records into a data frame

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 53}, "colab_type": "code", "executionInfo": {"elapsed": 225425, "status": "ok", "timestamp": 1561378328322, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "Yqd-a1hoLecl", "outputId": "f80486df-05c6-499b-c516-4a23dde0f93c"}
# %%time
fitness_res = pd.DataFrame.from_dict(pop_records)

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 206}, "colab_type": "code", "executionInfo": {"elapsed": 224003, "status": "ok", "timestamp": 1561378328334, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "SiZ9BZI_Lkze", "outputId": "4211db38-4426-42ef-9581-cd8ea7e001df"}
cols = [
    "exp",
    "pop",
    "b_ratio",
    "cx_pb",
    "mut_pb",
    "mut_sig",
    "k_par",
    "k_sur",
    "rep",
    "seed",
    "births",
    "avg",
    "best",
    "med",
    "std",
    "worst",
]
fitness_res = fitness_res[cols]
fitness_res.head()

# %% [markdown]
# We store the DataFrame in an external file

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 35}, "colab_type": "code", "executionInfo": {"elapsed": 274343, "status": "ok", "timestamp": 1561378380422, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "_pCCjaBQeeja", "outputId": "01bc3f1e-ac35-4c95-e52a-985413d5e120"}
fit_log_df_file = pickle_dir + file_sufix + "_fit_log_df.xz"
fitness_res.to_pickle(fit_log_df_file)
print(len(fitness_res))

# %% [markdown]
# We read them if necessary

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 35}, "colab_type": "code", "executionInfo": {"elapsed": 6661, "status": "ok", "timestamp": 1561410735289, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "NZUOKTLqUzD2", "outputId": "486033f0-952e-4c99-e84f-f08c0c6bdc37"}
fit_log_df_file = pickle_dir + file_sufix + "_fit_log_df.xz"
fitness_res = pd.read_pickle(fit_log_df_file)
print(len(fitness_res))

# %% [markdown] {"colab_type": "text", "id": "krdl63e4Mi4v"}
# ## Storing DataFrame of ca. 400 samples of each generation

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 1178}, "colab_type": "code", "executionInfo": {"elapsed": 252346, "status": "ok", "timestamp": 1561378381299, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "JwTFNB4wLkYx", "outputId": "cf44d987-5e7c-46b2-92c4-d48b7bf41a36"}
# %%time
index = [
    "exp",
    "pop",
    "b_ratio",
    "cx_pb",
    "mut_pb",
    "mut_sig",
    "k_par",
    "k_sur",
    "rep",
    "seed",
]
print(fitness_res.groupby(index).size())


# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 53}, "colab_type": "code", "executionInfo": {"elapsed": 251059, "status": "ok", "timestamp": 1561378381312, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "MzL1TW0UpqzB", "outputId": "65f2126b-4029-4de1-e559-80be73e95de5"}
def filter_group(dfg, col, qty):
    col_min = dfg[col].min()
    col_max = dfg[col].max()
    col_length = dfg[col].size
    jumps = col_length - 1
    jump_size = int((col_max - col_min) / jumps)
    new_jump_size = jumps / qty
    if new_jump_size > 1:
        new_jump_size = int(new_jump_size) * jump_size
    else:
        new_jump_size = jump_size

    col_select = list(range(col_min, col_max, new_jump_size))
    col_select.append(col_max)

    return dfg[dfg[col].isin(col_select)]


# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 1178}, "colab_type": "code", "executionInfo": {"elapsed": 254156, "status": "ok", "timestamp": 1561378386180, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "enPjlSiJ5SLS", "outputId": "e44b5872-aa52-4ea8-e2cb-24908c19565c"}
# %%time
index = [
    "exp",
    "pop",
    "b_ratio",
    "cx_pb",
    "mut_pb",
    "mut_sig",
    "k_par",
    "k_sur",
    "rep",
    "seed",
]
grouped = fitness_res.groupby(index, group_keys=False).apply(
    lambda x: filter_group(x, "births", 400)
)
print(grouped.groupby(index).size())

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 71}, "colab_type": "code", "executionInfo": {"elapsed": 263567, "status": "ok", "timestamp": 1561378397723, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "PbCor2-gk_ro", "outputId": "928dc84e-648a-4572-fda6-92623760553f"}
# %%time
fit_df_file = pickle_dir + file_sufix + "_fit_df.xz"
grouped.to_pickle(fit_df_file)
print(len(grouped))

# %% [markdown] {"colab_type": "text", "id": "oeiufrGvYTEt"}
# ## Storing DataFrame of final values of each replicate

# %% [markdown] {"colab_type": "text", "id": "UNQn8LtmfO6y"}
# We filter the final values of each replicate and store the Data Frame

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 71}, "colab_type": "code", "executionInfo": {"elapsed": 257921, "status": "ok", "timestamp": 1561378398113, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "glPt1U9tfPgW", "outputId": "85d72df1-1363-4ed6-8ef3-45d0460bb4f9"}
# %%time
query = fitness_res["births"] >= births_end
fit_fin_res = fitness_res[query]

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 71}, "colab_type": "code", "executionInfo": {"elapsed": 257921, "status": "ok", "timestamp": 1561378398113, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "glPt1U9tfPgW", "outputId": "85d72df1-1363-4ed6-8ef3-45d0460bb4f9"}
fit_fin_df_file = pickle_dir + file_sufix + "_fit_fin_df.xz"
fit_fin_res.to_pickle(fit_fin_df_file)
print(len(fit_fin_res))

# %% [markdown] {"colab_type": "text", "id": "dR20FtqVVjJY"}
# ## Storing DataFrame of values after birth 60K

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 71}, "colab_type": "code", "executionInfo": {"elapsed": 1566, "status": "ok", "timestamp": 1561438200551, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "aK8z-wsUY64t", "outputId": "faab8381-c34b-4bb0-ab86-b854002441f9"}
# %%time
query = fitness_res["births"] > 60e3
fit_60k_res = fitness_res[query]

print(len(fit_60k_res))

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 35}, "colab_type": "code", "executionInfo": {"elapsed": 1879, "status": "ok", "timestamp": 1561438212523, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "v8E6MF7PWXeS", "outputId": "2bb7e5be-10f3-410a-9dd8-fe943424ad32"}
index = [
    "exp",
    "pop",
    "b_ratio",
    "cx_pb",
    "mut_pb",
    "mut_sig",
    "k_par",
    "k_sur",
    "rep",
    "seed",
]
fit_60k_res = fit_60k_res.sort_values("births").groupby(index, as_index=False).first()

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 35}, "colab_type": "code", "executionInfo": {"elapsed": 1879, "status": "ok", "timestamp": 1561438212523, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "v8E6MF7PWXeS", "outputId": "2bb7e5be-10f3-410a-9dd8-fe943424ad32"}
fit_60k_df_file = pickle_dir + file_sufix + "_fit_60k_df.xz"
fit_60k_res.to_pickle(fit_60k_df_file)
len(fit_60k_res)

# %% [markdown] {"colab_type": "text", "id": "dR20FtqVVjJY"}
# ## Storing DataFrame of values after birth 30K

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 71}, "colab_type": "code", "executionInfo": {"elapsed": 1566, "status": "ok", "timestamp": 1561438200551, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "aK8z-wsUY64t", "outputId": "faab8381-c34b-4bb0-ab86-b854002441f9"}
# %%time
query = fitness_res["births"] > 30e3
fit_30k_res = fitness_res[query]

print(len(fit_30k_res))

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 35}, "colab_type": "code", "executionInfo": {"elapsed": 1879, "status": "ok", "timestamp": 1561438212523, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "v8E6MF7PWXeS", "outputId": "2bb7e5be-10f3-410a-9dd8-fe943424ad32"}
index = [
    "exp",
    "pop",
    "b_ratio",
    "cx_pb",
    "mut_pb",
    "mut_sig",
    "k_par",
    "k_sur",
    "rep",
    "seed",
]
fit_30k_res = fit_30k_res.sort_values("births").groupby(index, as_index=False).first()

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 35}, "colab_type": "code", "executionInfo": {"elapsed": 1879, "status": "ok", "timestamp": 1561438212523, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "v8E6MF7PWXeS", "outputId": "2bb7e5be-10f3-410a-9dd8-fe943424ad32"}
fit_30k_df_file = pickle_dir + file_sufix + "_fit_30k_df.xz"
fit_30k_res.to_pickle(fit_30k_df_file)
len(fit_30k_res)
