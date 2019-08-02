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

# %%
x_min = -6.01717
y_min = 9.06022
f_min = (
        g_fun(x_min + 11.0, y_min + 9.0)
        + g_fun(x_min - 11.0, y_min - 3.0)
        + g_fun(x_min + 6.0, y_min - 9.0)
    )
f_min

# %%
plot_x = np.linspace(-50,50,100)
plot_y = np.sinc(plot_x/np.pi)

fig,ax = plt.subplots(figsize=(6,2))
plt.plot(plot_x,plot_y)


# %%
def g_fun_mono(z):
    mag = np.sqrt(z ** 2.0)
    val = -(50.0 * np.sinc(mag / np.pi) - mag)
    return val

plot_x = np.linspace(-50,50,100)
plot_y = g_fun_mono(plot_x)

fix,ax = plt.subplots(figsize=(6,2))
plt.plot(plot_x,plot_y)

# %%
plot_x = np.linspace(-60,60,100)
plot_y = g_fun_mono(plot_x) + g_fun_mono(plot_x-15) + g_fun_mono(plot_x+15)

fix,ax = plt.subplots(figsize=(6,2))
plt.plot(plot_x,plot_y)

# %% [markdown] {"toc-hr-collapsed": false}
# # Running the Evolutionary Algorithm

# %% [markdown] {"toc-hr-collapsed": true}
# ## Setting the EA's parameters

# %%
# Algorithm parameters
# Number of replicates, and generations per experiment
rep_end = 1
births_end = 120e3

# Genes
gen_size = 2
# Population size
pop_size = 20
# Progeny and parents size
b_ratio = 3
par_size = b_ratio * pop_size

# Progeny parameters
## Crossover probability per gene
cx_pb = 0.5
## Mutation probability per gene
mut_pb = 0.5
## Mutation strength
sigma = 2.5

# Selection by tournament
# Tournament size parent selection
k_p = 2
# Tournament size survivor selection
k_s = 6

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

toolbox.register("mate", tools.cxUniform, indpb=cx_pb)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=sigma, indpb=mut_pb)

toolbox.register("par_select", tools.selTournament, tournsize=k_p)
toolbox.register("sur_select", tools.selTournament, tournsize=k_s)

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

# %%
# starting seed
rnd.seed(42)

# %%
# %%timeit
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
        logbook.record(rep=rep_n + 1, seed=rep_seed, births=births, **record)

        while births < births_end:
            # Select Parents and clone them as base for offsprings
            parents = toolbox.par_select(pop, par_size)
            offspring = [toolbox.clone(ind) for ind in parents]
            births = births + len(offspring)

            # Aplly crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                toolbox.mate(child1, child2)

            # Apply mutation
            for mutant in offspring:
                toolbox.mutate(mutant)
                del mutant.fitness.values

            fitnesses = toolbox.map(toolbox.evaluate, offspring)
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit

            pop = toolbox.sur_select((pop + offspring), pop_size)

            record = stat.compile(pop)
            logbook.record(rep=rep_n + 1, seed=rep_seed, births=births, **record)

# %% [markdown] {"toc-hr-collapsed": false}
# ### Data Analysis

# %% [markdown]
# We transform the records into a Data Frame

# %%
pop_records = [record for record in logbook]
fitness_res = pd.DataFrame.from_dict(pop_records)

# %% [markdown] {"toc-hr-collapsed": false}
# #### Fitness development

# %%
# %%time
fig, ax = plt.subplots(figsize=(6, 2))
ax = sns.lineplot(x="births", y="best", data=fitness_res, label="best fitness")

ax.set_axisbelow(True)
ax.minorticks_on()

ax.set(xscale="log")

ax.set_xlabel("births", fontsize="medium")
ax.set_ylabel("fitness", fontsize="medium")

ax.set_xlim((2e1, 120e3))
ax.set_ylim((0, 30))

ax.grid(True, axis="x", which="major", alpha=1, color="w", ls="-")
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.grid(True, axis="y", which="major", alpha=0.5, ls="-")
ax.grid(True, axis="y", which="minor", alpha=0.25, ls="--")

ax.legend(fontsize="small", facecolor="white", framealpha=0.5)

ax.plot()

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

# Algorithm parameters
# Number of replicates
rep_end = 50

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
        logbook.record(rep=rep_n + 1, seed=rep_seed, births=births, **record)

        while births < births_end:
            # Select Parents and clone them as base for offsprings
            parents = toolbox.par_select(pop, par_size)
            offspring = [toolbox.clone(ind) for ind in parents]
            births = births + len(offspring)

            # Aplly crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                toolbox.mate(child1, child2)

            # Apply mutation
            for mutant in offspring:
                toolbox.mutate(mutant)
                del mutant.fitness.values

            fitnesses = toolbox.map(toolbox.evaluate, offspring)
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit

            pop = toolbox.sur_select((pop + offspring), pop_size)

            record = stat.compile(pop)
            logbook.record(rep=rep_n + 1, seed=rep_seed, births=births + 1, **record)

# %%
pickle_file = "./pickle/C_01.joblib"

# %%
# %%time
with open(pickle_file, "wb") as handle:
    joblib.dump(logbook, handle, compress=("xz", 2))

# %%
# %%time
with open(pickle_file, "rb") as handle:
    logb = joblib.load(handle)

# %% [markdown] {"toc-hr-collapsed": false}
# ### Data Analysis

# %% [markdown]
# We transform the records into a Data Frame

# %%
fitness_res = pd.DataFrame.from_dict(logb)
fitness_res = fitness_res[
    ["rep", "seed", "births", "avg", "std", "med", "worst", "best"]
]

# %% [markdown]
# We filter the values of the last generation

# %%
query = fitness_res["births"] >= births_end
fin_fit_120k = fitness_res[query]
query = fitness_res["births"] == 60021
fin_fit_60k = fitness_res[query]
query = fitness_res["births"] == 30021
fin_fit_30k = fitness_res[query]


# %% [markdown]
# Top 10 best (lowest-fitness) individuals

# %%
fin_fit_120k.sort_values(by=["best"]).head(10)

# %% [markdown]
# Top 10 worst (highest-fitness) individuals

# %%
fin_fit_120k.sort_values(by=["best"], ascending=False).head(10)

# %% [markdown] {"toc-hr-collapsed": false}
# ### Visualization

# %% [markdown]
# #### Aggregated results

# %%
fin_fit_best_120k = fin_fit_120k["best"]
fin_fit_best_60k = fin_fit_60k["best"]
fin_fit_best_30k = fin_fit_30k["best"]


# %%
# %%time
fig, axs = plt.subplots(
    nrows=1,
    ncols=3,
    sharey="row",
    sharex="row",
    constrained_layout=True,
    figsize=(12, 2),
)

bins = np.linspace(-0.01, 6.99, 7 + 1)
axs[0].hist(fin_fit_best_30k, bins, label="best", alpha=0.5)
axs[1].hist(fin_fit_best_60k, bins, label="best", alpha=0.5)
axs[2].hist(fin_fit_best_120k, bins, label="best", alpha=0.5)

axs[0].set_title("(a) at 30k", fontsize="medium")
axs[1].set_title("(a) at 60k", fontsize="medium")
axs[2].set_title("(a) at 120k", fontsize="medium")

axs[0].set_ylabel("frequency", fontsize="medium")

for ax in axs.flatten():
    ax.vlines(bins, 0, 40, colors="xkcd:teal", linestyles=(0, (5, 10)), linewidths=0.75)

    ax.set_ylim((0, 40 + 1))
    ax.set_xlim((-0.05, 7.05))

    ax.set_xlabel("fitness", fontsize="medium")

    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.grid(True, axis="y", which="major", alpha=0.5, ls="-")
    ax.grid(True, axis="y", which="minor", alpha=0.25, ls="--")
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.grid(True, axis="x", which="major", alpha=1, color="w", ls="-")

    ax.legend(fontsize="small", facecolor="white", framealpha=0.5)

# %%
# %%time
fig, axs = plt.subplots(
    nrows=1,
    ncols=3,
    sharey="row",
    sharex="row",
    constrained_layout=True,
    figsize=(12, 2),
)
bins = np.linspace(-1e-8, 1e-6 - 1e-8, num=6)
axs[0].hist(fin_fit_best_30k, bins, label="best", alpha=0.5)
axs[1].hist(fin_fit_best_60k, bins, label="best", alpha=0.5)
axs[2].hist(fin_fit_best_120k, bins, label="best", alpha=0.5)

axs[0].set_title("(a) at 30k", fontsize="medium")
axs[1].set_title("(a) at 60k", fontsize="medium")
axs[2].set_title("(a) at 120k", fontsize="medium")

axs[0].set_ylabel("frequency", fontsize="medium")

for ax in axs.flatten():
    ax.vlines(bins, 0, 10, colors="xkcd:teal", linestyles=(0, (5, 10)), linewidths=0.75)

    ax.set_ylim((0, None))

    ax.set_xlabel("fitness", fontsize="medium")

    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(True, axis="y", which="major", alpha=0.5, ls="-")
    ax.grid(True, axis="y", which="minor", alpha=0.25, ls="--")
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.grid(True, axis="x", which="major", alpha=1, color="w", ls="-")
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    ax.legend(fontsize="small", facecolor="white", framealpha=0.5)

# %%
# %%time
fig, ax = plt.subplots(figsize=(6, 2))
ax = sns.lineplot(x="births", y="best", data=fitness_res, label="best fitness")

ax.set_axisbelow(True)
ax.minorticks_on()

ax.set(xscale="log")

ax.set_xlabel("births", fontsize="medium")
ax.set_ylabel("fitness", fontsize="medium")

ax.set_xlim((2e1, 120e3))
ax.set_ylim((0, 30))

ax.grid(True, axis="x", which="major", alpha=1, color="w", ls="-")
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.grid(True, axis="y", which="major", alpha=0.5, ls="-")
ax.grid(True, axis="y", which="minor", alpha=0.25, ls="--")

ax.legend(fontsize="small", facecolor="white", framealpha=0.5)
ax.plot()

# %%
# %%time
query = fitness_res["births"] <= 1e3
fig, ax = plt.subplots(figsize=(6, 2))
ax = sns.lineplot(x="births", y="best", data=fitness_res[query], label="best fitness")

ax.set_axisbelow(True)
ax.minorticks_on()

ax.set(xscale="log")

ax.set_xlabel("births", fontsize="medium")
ax.set_ylabel("fitness", fontsize="medium")

ax.set_xlim((2e1, 1e3))
ax.set_ylim((0, 30))

ax.grid(True, axis="x", which="major", alpha=1, color="w", ls="-")
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.grid(True, axis="y", which="major", alpha=0.5, ls="-")
ax.grid(True, axis="y", which="minor", alpha=0.25, ls="--")

ax.legend(fontsize="small", facecolor="white", framealpha=0.5)

ax.plot()

# %%
# %%time
query = fitness_res["births"] >= 1e3
fig, ax = plt.subplots(figsize=(6, 2))
ax = sns.lineplot(x="births", y="best", data=fitness_res[query], label="best fitness")

ax.set_axisbelow(True)
ax.minorticks_on()

ax.set(xscale="log")

ax.set_xlabel("births", fontsize="medium")
ax.set_ylabel("fitness", fontsize="medium")

ax.set_xlim((1e3, 120e3))
# ax.set_ylim((2.13, 2.18))

ax.grid(True, axis="x", which="major", alpha=1, color="w", ls="-")
# ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
# ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.grid(True, axis="y", which="major", alpha=0.5, ls="-")
ax.grid(True, axis="y", which="minor", alpha=0.25, ls="--")

ax.legend(fontsize="small", facecolor="white", framealpha=0.5)

ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

ax.plot()

# %%
# %%time
query = fitness_res["births"] >= 1e3
fig, ax = plt.subplots(figsize=(6, 2))
ax = sns.lineplot(x="births", y="best", data=fitness_res[query], label="best fitness")

ax.set_axisbelow(True)
ax.minorticks_on()

ax.set(xscale="log")

ax.set_xlabel("births", fontsize="medium")
ax.set_ylabel("fitness", fontsize="medium")

ax.set_xlim((1e3, 120e3))
ax.set_ylim((2.135, 2.17))

ax.grid(True, axis="x", which="major", alpha=1, color="w", ls="-")
# ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
# ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.grid(True, axis="y", which="major", alpha=0.5, ls="-")
ax.grid(True, axis="y", which="minor", alpha=0.25, ls="--")

ax.legend(fontsize="small", facecolor="white", framealpha=0.5)

ax.plot()

# %%
