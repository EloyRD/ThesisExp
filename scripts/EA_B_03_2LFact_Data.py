# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Python [conda env:thesis] *
#     language: python
#     name: conda-env-thesis-py
# ---

# %% [raw]
# \author{Eloy Ruiz-Donayre}
# \title{TESTCASE B - 2-Level 6-Factor Full Factorial (With 30 replicates) - Data Generation}
# \date{\today}
# \maketitle

# %% [raw]
# \tableofcontents

# %% [markdown]
# # Preliminaries

# %% [markdown]
# Importing python packages and setting display parameters

# %%
import numpy as np
import pandas as pd
import itertools as it
import scipy.stats as stats

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import thesis_EAfunc as EAf
import thesis_visfunc as EAv

# %%
plt.style.use('bmh')
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

pd.set_option('display.latex.repr', True)
pd.set_option('display.latex.longtable', True)

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


def g(x, y):
    mag = np.sqrt(x**2. + y**2.)
    return -(50.*np.sinc(mag/np.pi) - mag)


def f(x, y):
    x_min = -6.01717
    y_min = 9.06022
    f_min = g(x_min+11., y_min+9.) + g(x_min-11.,
                                       y_min-3.) + g(x_min+6., y_min-9.)
    tripsinc = g(x+11., y+9.) + g(x-11., y-3.) + g(x+6., y-9.) - (f_min)
    return tripsinc


# %%
# Testing the minimum
print(f(-6.01717, 9.06022))

# %%
# Testing the function
print(f(-1., -1.), f(-11., -9.), f(11., 3.), f(-6., 9.))

# %% [markdown] {"toc-hr-collapsed": false}
# # Setting up the experiment
# 64 Experiments
# >L-> In each experiment, one set of parameters is used.
# >>L-> 40 Replicates per experiment.
# >>>L-> Each replicate is different due to randomness effects.

# %%
# starting seed
np.random.seed(42)

# %% [markdown]
# ## Initializing data storage

# %%
mult_fit_cols = ['exp'] + ['pop_s'] + ['b'] + ['mut_p'] + ['mut_s'] + ['p_sel'] + ['s_sel'] + \
    ['run', 'generation', 'fitness_min', 'fitness_max', 'fitness_mean', 'fitness_std']
multi_fit = pd.DataFrame(columns=mult_fit_cols)
multi_fit = multi_fit.infer_objects()

# %% [markdown] {"toc-hr-collapsed": false}
# ## Parameter space for the experiment

# %% [markdown]
# ### Initializing

# %%
# Algorithm parameters
# Number of replicates, and generations per experiment
rep_n = 30
gen_f = 200

# Population size
pop_s = [40, 160]

# Parent subpopulation's selection method and size
par_selection = ['uniform', 'tournament_k3']
b = [0.5, 5]
par_s = [z*y for z in pop_s for y in b]

# Progeny subpopulation's size
prog_s = par_s

# Crossover Method
crossover = 'uniform'
# Mutation method, probability and size
mutation = 'random_all_gau_dis'
mut_p = [0.1, 0.5]
mut_s = [2.5, 7.5]

# New population selection method
sur_selection = ['uniform', 'tournament_k3']

# %% [markdown]
# ### 2-Level Factors encoded values

# %%
inputs_labels = {'pop_s': 'Population size',
                 'b': 'Progeny-to-population ratio',
                 'mut_p': 'Mutation Probability',
                 'mut_s': 'Mutation size',
                 'p_sel': 'Parent selection',
                 's_sel': 'Survivor selection method'
                 }

dat = [('pop_s',  40, 160, -1, 1, 'Numerical'),
       ('b', 0.5, 5, -1, 1, 'Numerical'),
       ('mut_p', 0.1, 0.5, -1, 1, 'Numerical (<1)'),
       ('mut_s', 2.5, 7.5, -1, 1, 'Numerical'),
       ('p_sel', 'uniform', 'tournament k3', -1, 1, 'Categorical'),
       ('s_sel', 'uniform', 'tournament k3', -1, 1, 'Categorical')
       ]

inputs_df = pd.DataFrame(dat, columns=[
                         'Factor', 'Value_low', 'Value_high', 'encoded_low', 'encoded_high', 'Variable type'])
inputs_df = inputs_df.set_index(['Factor'])
inputs_df['Label'] = inputs_df.index.map(lambda z: inputs_labels[z])
inputs_df = inputs_df[['Label', 'Variable type',
                       'Value_low', 'Value_high', 'encoded_low', 'encoded_high']]

inputs_df

# %% [markdown]
# ### Combining the 2-level Factors

# %% [markdown]
# We create a list with all the possible combinations of the 2-level factors

# %%
exp_par = list(it.product(pop_s, b, mut_p, mut_s,
                          par_selection, sur_selection))
print('Cantidad de combinaciones de parametros en "exp_par" :'+str(len(exp_par)))
print()
print('Primera y última combinación de parametros en "exp_par":')
print('Secuencia (pop_s, b, mut_p, mut_s, p_sel, s_sel)')
print(exp_par[0])
print(exp_par[63])

# %% [markdown]
# # Experiment execution

# %%
# %%time
exp_n = 1
for (zz, yy, xx, vv, uu, tt) in exp_par:
    sur_selection = tt
    par_selection = uu
    mut_s = vv
    mut_p = xx
    b = yy
    pop_s = zz
    prog_s = int(b * pop_s)
    par_s = prog_s

    fitness_res = EAf.EA_exp_only_fitness(
        rep_n, gen_f, f, domain, pop_s, par_s, prog_s, mut_p, mut_s, par_selection, crossover, mutation, sur_selection)

    fitness_res.insert(0, 's_sel', tt)
    fitness_res.insert(0, 'p_sel', uu)
    fitness_res.insert(0, 'mut_s', vv)
    fitness_res.insert(0, 'mut_p', xx)
    fitness_res.insert(0, 'b', yy)
    fitness_res.insert(0, 'pop_s', zz)
    fitness_res.insert(0, 'exp', exp_n)
    multi_fit = multi_fit.append(fitness_res, ignore_index=True, sort=False)
    multi_fit = multi_fit.infer_objects()

    exp_n += 1

# %% [markdown]
# ## Data storage

# %% [markdown]
# Writing the Data Frame to a pickle file

# %%
multi_fit.to_pickle('./Data/TEST_B_2L_FitData.gz', compression='gzip')

# %% [markdown]
# Reading the Data Frame from a pickle file

# %%
multi_fit = pd.read_pickle('./Data/TEST_B_2L_FitData.gz', compression='gzip')

# %%
multi_fit.tail()

# %% [markdown]
# # Processing data for DOE Analysis

# %% [markdown]
# Storing the latest generation's population of each replicate

# %%
query = (multi_fit['generation'] == gen_f)
multi_final_fitness_res = multi_fit[query]

# %% [markdown]
# Reordering columns

# %%
multi_final_fitness_res = multi_final_fitness_res.drop(
    ['exp', 'generation', 'run', 'seed'], axis=1)
multi_final_fitness_res.columns = [
    'pop_s', 'b', 'mut_p', 'mut_s', 'p_sel', 's_sel', 'f_min', 'f_max', 'f_mean', 'f_std']
multi_final_fitness_res = multi_final_fitness_res[[
    'pop_s', 'b', 'mut_p', 'mut_s', 'p_sel', 's_sel', 'f_min', 'f_max', 'f_mean', 'f_std']]
multi_final_fitness_res = multi_final_fitness_res.reset_index(drop=True)

# %% [markdown]
# Encoding values for DOE's Factor

# %%
multi_final_fitness_res['pop_s'] = multi_final_fitness_res['pop_s'].replace(
    [40, 160], [-1, 1]).infer_objects()
multi_final_fitness_res['b'] = multi_final_fitness_res['b'].replace(
    [.5, 5], [-1, 1]).infer_objects()
multi_final_fitness_res['mut_p'] = multi_final_fitness_res['mut_p'].replace(
    [.1, .5], [-1, 1]).infer_objects()
multi_final_fitness_res['mut_s'] = multi_final_fitness_res['mut_s'].replace(
    [2.5, 7.5], [-1, 1]).infer_objects()
multi_final_fitness_res['p_sel'] = multi_final_fitness_res['p_sel'].replace(
    ['uniform', 'tournament_k3'], [-1, 1]).infer_objects()
multi_final_fitness_res['s_sel'] = multi_final_fitness_res['s_sel'].replace(
    ['uniform', 'tournament_k3'], [-1, 1]).infer_objects()

# %% [markdown]
# Exploring the Data Frame

# %%
multi_final_fitness_res.head()

# %%
multi_final_fitness_res.tail()

# %% [markdown]
# Storing the Factor Coding and DOE results Data Frames

# %%
inputs_df.to_pickle('./Data/TEST_B_DOE_code.gz', compression='gzip')
multi_final_fitness_res.to_pickle(
    './Data/TEST_B_DOE_data.gz', compression='gzip')

# %%
