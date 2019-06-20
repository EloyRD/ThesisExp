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
# \title{TESTCASE A - One Parameter Set}
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

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from thesis_EAfunc import *
from thesis_visfunc import *

# %%
plt.style.use('bmh')
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

pd.set_option('display.latex.repr', True)
pd.set_option('display.latex.longtable', True)

# %% [markdown] {"toc-hr-collapsed": false}
# # Fitness Landscape Definition

# %%
#Problem domain
x_min = -15
x_max = 15
y_min = -15
y_max = 15

#Known minimum
x_point = -1
y_point = -1

domain = (x_min, x_max, y_min, y_max)
point = (x_point, y_point)
img_size = (8.5, 4.25)

#Problem definition
def f(x, y):
    D = 2
    alpha = 1/8
    
    x=(x-5)/6
    y=(y-5)/6
    
    a = np.abs(x ** 2 + y ** 2 - D) ** (alpha * D)
    b = ( 0.5 * (x ** 2 + y ** 2) + (x + y) ) / D
        
    return (a + b + 0.5)


# %%
#Testing the minimum
print(f(-1,-1))

# %%
#Testing the function
print(f(-1.,-1.), f(-11.,-9.), f(11.,3.), f(-6.01717,9.06022))

# %% [markdown]
# ## Visualizing Landscape

# %%
#Visualization parameters
grph_steps = 61
a=15
b=-60
ratio_w=1.3
ln=.75

# %%
EA_plt_land(f, domain, point, grph_steps, a=a, b=b, imgsize=img_size, ratio_w=ratio_w, ln=ln)

# %%
domain_min=(-3,1,-3,1)
EA_plt_land(f, domain_min, point, 21, a=a, b=b, imgsize=img_size, ratio_w=ratio_w, ln=ln)

# %% [markdown] {"toc-hr-collapsed": false}
# # Running the Evolutionary Algorithm

# %% [markdown] {"toc-hr-collapsed": true}
# ## Setting the EA's parameters

# %%
#starting seed
np.random.seed(42)

# %%
#Algorithm parameters
## Number of replicates, and generations per experiment
rep_n = 1
gen_f = 50

## Population size
pop_s = 20

## Parent subpopulation's selection method and size
par_selection = 'fitness_proportional_selection'

b = 3
par_s = b * pop_s

## Progeny subpopulation's and size
prog_s = par_s
### Crossover Method
crossover = 'uniform'
### Mutation method, probability and size
mutation = 'random_all_gau_dis'
mut_p = 0.5
mut_s = 2.5

## Survivors (New population) selection method
sur_selection = 'tournament_k3'

# %% [markdown] {"toc-hr-collapsed": true}
# ## Single Run of the EA Experiments  
# 1 Experiment
# L-> 1 Parameter set for the experiment.
# >L-> 1 Replicate.
# >>L-> The replicate is affected due to the randomness seed.

# %%
# %%time
genera_res, fitness_res = EA_exp(rep_n, gen_f, f, domain, pop_s, par_s, prog_s, mut_p, mut_s, par_selection, crossover, mutation, sur_selection)

# %% [markdown]
# We query the members of the population of the last generation

# %%
z=gen_f
query = (genera_res['generation']==z)
genera_res[query]

# %% [markdown] {"toc-hr-collapsed": false}
# ### Visualization

# %% [markdown] {"toc-hr-collapsed": false}
# #### Fitness development

# %%
EA_fitn_dev(fitness_res, 0)

# %% [markdown] {"toc-hr-collapsed": false}
# #### Population dynamics

# %% [markdown]
# First generation and its progeny

# %%
run_s=0  #First (and only) run
gen_s=0  #First generation
EA_plt_gen(f, domain, grph_steps, genera_res, run_s, gen_s, a=a, b=b, imgsize=img_size, ratio_w=ratio_w, ln=ln)

# %% [markdown]
# Dynamics of the population

# %%
print('Generation 0')
EA_plt_pop(f, domain, grph_steps, genera_res, run_s, 0, a=a, b=b, imgsize=img_size, ratio_w=ratio_w, ln=ln)
print('Generation 5')
EA_plt_pop(f, domain, grph_steps, genera_res, run_s, 5, a=a, b=b, imgsize=img_size, ratio_w=ratio_w, ln=ln)
print('Generation 10')
EA_plt_pop(f, domain, grph_steps, genera_res, run_s, 10, a=a, b=b, imgsize=img_size, ratio_w=ratio_w, ln=ln)
print('Generation 20')
EA_plt_pop(f, domain, grph_steps, genera_res, run_s, 20, a=a, b=b, imgsize=img_size, ratio_w=ratio_w, ln=ln)
print('Generation 30')
EA_plt_pop(f, domain, grph_steps, genera_res, run_s, 30, a=a, b=b, imgsize=img_size, ratio_w=ratio_w, ln=ln)
print('Generation 40')
EA_plt_pop(f, domain, grph_steps, genera_res, run_s, 40, a=a, b=b, imgsize=img_size, ratio_w=ratio_w, ln=ln)
print('Generation 45')
EA_plt_pop(f, domain, grph_steps, genera_res, run_s, 45, a=a, b=b, imgsize=img_size, ratio_w=ratio_w, ln=ln)
print('Generation 50')
EA_plt_pop(f, domain, grph_steps, genera_res, run_s, 50, a=a, b=b, imgsize=img_size, ratio_w=ratio_w, ln=ln)

# %% [markdown]
# Dynamics of the population and its progeny

# %%
print('Generation 0')
EA_plt_gen(f, domain, grph_steps, genera_res, run_s, 0, a=a, b=b, imgsize=img_size, ratio_w=ratio_w, ln=ln)
print('Generation 5')
EA_plt_gen(f, domain, grph_steps, genera_res, run_s, 5, a=a, b=b, imgsize=img_size, ratio_w=ratio_w, ln=ln)
print('Generation 10')
EA_plt_gen(f, domain, grph_steps, genera_res, run_s, 10, a=a, b=b, imgsize=img_size, ratio_w=ratio_w, ln=ln)
print('Generation 20')
EA_plt_gen(f, domain, grph_steps, genera_res, run_s, 20, a=a, b=b, imgsize=img_size, ratio_w=ratio_w, ln=ln)
print('Generation 30')
EA_plt_gen(f, domain, grph_steps, genera_res, run_s, 30, a=a, b=b, imgsize=img_size, ratio_w=ratio_w, ln=ln)
print('Generation 40')
EA_plt_gen(f, domain, grph_steps, genera_res, run_s, 40, a=a, b=b, imgsize=img_size, ratio_w=ratio_w, ln=ln)
print('Generation 45')
EA_plt_gen(f, domain, grph_steps, genera_res, run_s, 45, a=a, b=b, imgsize=img_size, ratio_w=ratio_w, ln=ln)
print('Generation 49')
EA_plt_gen(f, domain, grph_steps, genera_res, run_s, 49, a=a, b=b, imgsize=img_size, ratio_w=ratio_w, ln=ln)

# %% [markdown] {"toc-hr-collapsed": true}
# ## 100 Executions of the EA
# 1 Experiment
# >L-> 1 Parameter set for the experiment.
# >>L-> 100 Replicate.
# >>>L-> Each replicate is different due to randomness effects.

# %% [markdown]
# ### Changing parameters

# %%
# Restarting seed
np.random.seed(42)

# Algorithm parameters
## Number of replicates
rep_n = 100
## Number of generations
gen_f = 200

# %% [markdown]
# ### Execution

# %%
# %%time
fitness_res = EA_exp_only_fitness(rep_n, gen_f, f, domain, pop_s, par_s, prog_s, mut_p, mut_s, par_selection, crossover, mutation, sur_selection)

# %%
fitness_res.head()

# %%
fitness_res.tail()

# %% [markdown] {"toc-hr-collapsed": true}
# ### Data Analysis

# %% [markdown]
# Top 10 fittest

# %%
z=gen_f
query = (fitness_res['generation']==z)
fitness_res[query].sort_values(by=['fitness_min']).head(10)

# %% [markdown]
# Top 10 least fit

# %%
fitness_res[query].sort_values(by=['fitness_min'], ascending=False).head(10)

# %% [markdown]
# ### Visualization

# %% [markdown]
# Aggregated results

# %%
z = gen_f
query = (fitness_res['generation']==z)
type(fitness_res[query]['fitness_mean'])

# %%
sns.distplot(fitness_res[query]['fitness_min'], rug=True)

# %%
sns.distplot(fitness_res[query]['fitness_std'], rug=True)

# %%
sns.lineplot(x='generation', y='fitness_min', data=fitness_res)

# %%
