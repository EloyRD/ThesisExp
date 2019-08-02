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
# \title{TESTCASE B - 2-Level 7-Factor Full Factorial (With 30 replicates) - Data Analysis}
# \date{\today}
# \maketitle

# %% [raw]
# \tableofcontents

# %% [markdown]
# # Preliminaries

# %% [markdown]
# This commands are used in Google Colab:

# %% [markdown] {"colab": {"base_uri": "https://localhost:8080/", "height": 53}, "colab_type": "code", "executionInfo": {"elapsed": 4917, "status": "ok", "timestamp": 1561564955582, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "NyS-yVCxJxfK", "outputId": "aee402f2-7489-4bef-eae1-9318286da1dc"}
# !pip install deap

# %% [markdown] {"colab": {"base_uri": "https://localhost:8080/", "height": 35}, "colab_type": "code", "executionInfo": {"elapsed": 4894, "status": "ok", "timestamp": 1561564955586, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "va1gdtFzJxgF", "outputId": "b6d78934-981e-4c90-a8a1-b451815fac6c"}
# from google.colab import drive
# drive.mount("/content/gdrive")

# %% [markdown]
# Importing python packages and setting display parameters

# %%
import math as mt
import random as rnd
import numpy as np
import itertools as it

import numba
from numba import jit
import joblib

import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import EngFormatter
from matplotlib import lines

import pandas as pd
from collections import OrderedDict
import statistics as stats
import scipy.stats as sstats
import probscale

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

# %% [markdown]
# This sets the work directory for the pickle files

# %% {"colab": {}, "colab_type": "code", "id": "4xI6Ra3OJxhW"}
pickle_dir = "./pickle/"
file_sufix = "C_04"
pickle_dir + file_sufix

# %% [markdown]
# Run this when working in Google Colab:

# %% [markdown] {"colab": {}, "colab_type": "code", "id": "4xI6Ra3OJxhW"}
# pickle_dir = "/content/gdrive/My Drive/Colab Notebooks/thesis/"
# pickle_dir + file_sufix

# %% [markdown]
# # Reading data

# %% [markdown]
# ## Fitness results data

# %% [markdown]
# Reading the Data Frame from a pickle file

# %%
fit_fin_df_file = pickle_dir + file_sufix + "_fit_fin_df_80k.xz"
fit_res_df = pd.read_pickle(fit_fin_df_file)

# %% [markdown]
# Replicates in the sample

# %%
print("Replicates in sample: " + str(len(fit_res_df)))
print("Experiments in sample: " + str(len(fit_res_df) / 40))

# %%
fit_res_df.head()

# %%
fit_res_df.tail()

# %%
fit_res_df["best"].min()

# %%
t_u_v = fit_res_df.copy()
t_u_v = t_u_v[['exp', 'pop', 'b_ratio', 'cx_pb', 'mut_pb', 'mut_sig', 'k_par', 'k_sur','best']]
t_u_v = t_u_v.assign(Successful= lambda x:x["best"]<1e-6)
t_u_v = t_u_v.groupby(["exp", 'pop', 'b_ratio', 'cx_pb', 'mut_pb', 'mut_sig', 'k_par', 'k_sur'])
t_u_v = t_u_v["Successful"].sum().reset_index()
t_u_v["Successful"] = t_u_v["Successful"]/40*100
print(t_u_v.to_latex(index=False))

# %%
bins = [-1e-8, 1e-6, 1, 5, 6, 7, float("inf")]

# %%
d_e_f = fit_res_df.copy()
d_e_f = d_e_f[['exp', 'best']]
d_e_f["bins"] = pd.cut(d_e_f["best"], bins)
d_e_f = d_e_f.groupby(['exp','bins']).size().unstack(level=-1, fill_value=0)
print(d_e_f.to_latex())

# %%
d_e_f["bins"] = pd.cut(d_e_f["best"], bins)
d_e_f = d_e_f.merge(x_y_z[["exp", 'pop', 'b_ratio', 'cx_pb', 'mut_pb', 'mut_sig', 'k_par', 'k_sur', 'Par Conf']], 
                    on=["exp", 'pop', 'b_ratio', 'cx_pb', 'mut_pb', 'mut_sig', 'k_par', 'k_sur'])
d_e_f = d_e_f[["exp", "Par Conf", "rep", "births", "bins"]]
d_e_f = d_e_f.groupby(['exp','Par Conf', "births", 'bins']).size().unstack(level=-1, fill_value=0)
d_e_f

# %% [markdown] {"toc-hr-collapsed": true}
# # Experiment's factors and levels and other parameters

# %% [markdown] {"toc-hr-collapsed": false}
# ## Common parameters

# %%
# Algorithm parameters
# Number of replicates, and generations per experiment
rep_end = 40
births_end = 80e3

# Genes
gen_size = 2
# Population size
pop_size_lvl = [20, 160]
# Progeny and parents size ratio to population size
b_ratio_lvl = [2, 4]

# Progeny parameters
## Crossover probability per gene
cx_pb_lvl = [0.1, 0.5]
## Mutation probability per gene
mut_pb_lvl = [0.1, 0.5]
## Mutation strength
mut_sig_lvl = [0.5, 5]

# Selection by tournament
# Tournament size parent selection
k_par_lvl = [2, 6]
# Tournament size survivor selection
k_sur_lvl = [2, 6]

# %% [markdown]
# ## Factor levels

# %%
factors_levels = [
    ("pop", "Population size", "Integer +", min(pop_size_lvl), max(pop_size_lvl)),
    ("b_ratio", "Progeny-to-pop ratio", "Real +", min(b_ratio_lvl), max(b_ratio_lvl)),
    ("cx_pb", "Crossover prob", "Real [0,1]", min(cx_pb_lvl), max(cx_pb_lvl)),
    ("mut_pb", "Mutation prob", "Real [0,1]", min(mut_pb_lvl), max(mut_pb_lvl)),
    ("mut_sig", "Mutation sigma", "Real +", min(mut_sig_lvl), max(mut_sig_lvl)),
    ("k_par", "Parent tourn size", "Integer +", min(k_par_lvl), max(k_par_lvl)),
    ("k_sur", "Surviv tourn size", "Integer +", min(k_sur_lvl), max(k_sur_lvl)),
]

factors_df = pd.DataFrame(
    factors_levels, columns=["Factor", "Label", "Range", "LowLevel", "HighLevel"]
)
factors_df = factors_df.set_index(["Factor"])

factors_df

# %%
print(factors_df.to_latex())

# %% [markdown] {"toc-hr-collapsed": false}
# # DOE Analisis of Data

# %% [markdown] {"colab_type": "text", "id": "H0JIPY50Jxkr"}
# List with the Factors names

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 35}, "colab_type": "code", "executionInfo": {"elapsed": 8300, "status": "ok", "timestamp": 1561564959237, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "Y6xJR_JrJxkw", "outputId": "368ebca8-ffae-4988-84cf-cb9639f6e2b7"}
factors = list(factors_df.index.array)
print(factors)
factors_DOE = [factor + "_DOE" for factor in factors]
print(factors_DOE)

# %%
fit_success = fit_res_df.assign(success=fit_res_df.best.le(1e-6))
fit_success = fit_success.groupby(factors, as_index=False).agg({"success": np.sum})

# %%
fit_success["success"] = fit_success["success"]/40*100
fit_success

# %%
fit_success_doe = fit_res_df.assign(success=fit_res_df.best.le(1e-6))
fit_success_doe = fit_success_doe.groupby(factors, as_index=False).agg(
    {"success": np.sum}
)

# %%
for factor in factors:
    fac_min = fit_success[factor].min()
    fac_max = fit_success[factor].max()

    fit_success[factor] = fit_success[factor].map(lambda v: -1 if v == fac_min else 1)

# %% [markdown] {"toc-hr-collapsed": false}
# ## Defining variables and variable labels

# %%
labels = {}
labels[1] = list(factors)
for i in [2, 3, 4, 5, 6, 7]:
    labels[i] = list(it.combinations(labels[1], i))

obs_list = ["success"]

# %%
#for k in labels.keys():
#    print(str(k) + " : " + str(labels[k]))
#print()
#print(obs_list)

# %% [markdown] {"toc-hr-collapsed": false}
# ## Computing Main and Interaction Effects

# %% [markdown] {"toc-hr-collapsed": false}
# ### Constant Effect

# %%
effects = {}

# Start with the constant effect: this is $\overline{y}$
effects[0] = {"x0": [fit_success[obs_list[0]].mean()]}
print(effects[0])

# %% [markdown]
# ### Main effect of each variable

# %%
effects[1] = {}
for key in labels[1]:
    effects_result = []
    for obs in obs_list:
        effects_df = fit_success.groupby(key)[obs].mean()
        result = sum([zz * effects_df.loc[zz] for zz in effects_df.index])
        effects_result.append(result)
    effects[1][key] = effects_result

effects[1]

# %% [markdown]
# ### Interaction effects (2-variable to 7-variable interactions)

# %%
for c in range(2, 8):
    effects[c] = {}
    for key in labels[c]:
        effects_result = []
        for obs in obs_list:
            effects_df = fit_success.groupby(key)[obs].mean()
            result = sum(
                [
                    np.prod(zz) * effects_df.loc[zz] / (2 ** (len(zz) - 1))
                    for zz in effects_df.index
                ]
            )
            effects_result.append(result)
        effects[c][key] = effects_result

# %%
len(effects[1])+ len(effects[2]) + len(effects[3]) + len(effects[4]) + len(effects[5]) + len(effects[6]) + len(effects[7])


# %%
def printd(d):
    for k in d.keys():
        print("%25s : %s" % (k, d[k]))

#for i in range(1, 8):
#    printd(effects[i])


# %% [markdown] {"toc-hr-collapsed": false}
# ## Analysis

# %% [markdown] {"toc-hr-collapsed": false}
# ### Analyzing Effects

# %%
master_dict = {}
for nvars in effects.keys():
    effect = effects[nvars]
    for k in effect.keys():
        v = effect[k]
        master_dict[k] = nvars, v[0]

master_df = pd.DataFrame(master_dict).T
master_df.columns = ["interaction"] + obs_list
#master_df.head(15)

# %% [markdown]
# We calculate the Percentage contribution of each main factor to the variance

# %%
n_doe = 1
k_doe = 7

y1 = master_df.copy()
y1 = y1.iloc[y1[obs_list[0]].abs().argsort].iloc[::-1]
y1 = y1.drop("x0")
y1.columns = ["Int Level", "Effect_Estimate"]
y1.index.names = ["Factors"]
y1["Sum_of_Squares"] = y1["Effect_Estimate"] ** 2 * n_doe * (2 ** (k_doe - 2))

# %% [markdown]
# Top Fifteen of Percentage contribution to the variance of Interactions

# %%
SS_tot = (fit_success[obs_list[0]] ** 2).sum() - (
    (fit_success[obs_list[0]].sum() ** 2) / len(fit_success[obs_list[0]])
)
SS_err = SS_tot - (y1["Sum_of_Squares"].sum())
y1["%_Contribution"] = y1["Sum_of_Squares"] / SS_tot * 100
effect_estimate = y1.copy()
effect_estimate.index.names = ["Interaction"]
effect_estimate.drop(["Int Level"], axis=1).head(15)

# %%
print(effect_estimate.drop(["Int Level"], axis=1).head(20).to_latex(float_format=lambda x: '%.1f' % x))

# %% [markdown]
# Top Ten of Percentage contribution to the variance of main interactions

# %%
query = effect_estimate["Int Level"] == 1
effect_estimate[query].drop(["Int Level"], axis=1)

# %% [markdown]
# Top Ten of Percentage contribution to the variance of second-order interactions

# %%
query = effect_estimate["Int Level"] == 2
effect_estimate[query].drop(["Int Level"], axis=1).head(10)

# %% [markdown]
# Top Ten of Percentage contribution to the variance of third-order interactions

# %%
query = effect_estimate["Int Level"] == 3
effect_estimate[query].drop(["Int Level"], axis=1).head(10)

# %% [markdown]
# Percentage contribution of each Interaction level to the variance

# %%
var = y1.groupby(["Int Level"]).agg(
    OrderedDict(
        [
            ("Effect_Estimate", "count"),
            ("Sum_of_Squares", "sum"),
            ("%_Contribution", ["sum", "max", "min"]),
        ]
    )
).rename(columns={"Effect_Estimate": "Count", "sum": "total", "count": "total"}).to_latex()
print(var)

# %% [markdown]
# ### Main effects plot

# %% [markdown]
# Colors represent if factor is in the top 3 (green), top 5 (blue), top 10 (yellow)

# %%
effects_top_10 = effect_estimate.head(9).index.values.tolist()
effects_top_5 = effect_estimate.head(4).index.values.tolist()
effects_top_3 = effect_estimate.head(2).index.values.tolist()
query = effect_estimate["Int Level"] == 1
effect_level_1_ranked = effect_estimate[query].index.values.tolist()
query = effect_estimate["Int Level"] == 2
effect_level_2_ranked = effect_estimate[query].index.values.tolist()

print("Top 10 of effects with biggest impact:")
print(effects_top_10)
print("First level interactions sorted by impact:")
print(effect_level_1_ranked)
print("Secont level interactions sorted by impact:")
print(effect_level_2_ranked)

# %%
# %%time
fig, axs = plt.subplots(
    nrows=2, ncols=4, sharey=True, constrained_layout=True, figsize=(12, 4)
)

for ax, i in zip(axs.flatten(), range(len(effect_level_1_ranked))):
    sns.regplot(
        x=effect_level_1_ranked[i],
        y="success",
        data=fit_success_doe,
        x_estimator=np.mean,
        x_ci=None,
        ci=None,
        truncate=True,
        ax=ax,
    )
    ax.set_ylabel(None)
    ax.set_axisbelow(True)
    x_majors = [
        fit_success_doe[effect_level_1_ranked[i]].min(),
        fit_success_doe[effect_level_1_ranked[i]].max(),
    ]
    ax.xaxis.set_major_locator(ticker.FixedLocator(x_majors))
    ax.grid(True, axis="x", which="major", alpha=1, color="w", ls="-")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(3))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(True, axis="y", which="major", alpha=0.5, ls="-")
    ax.grid(True, axis="y", which="minor", alpha=0.25, ls="--")

for ax, i in zip(axs.flatten(), range(len(effect_level_1_ranked))):
    if effect_level_1_ranked[i] in effects_top_10:
        ax.set_facecolor("xkcd:pale yellow")
    if effect_level_1_ranked[i] in effects_top_5:
        ax.set_facecolor("xkcd:pale blue")
    if effect_level_1_ranked[i] in effects_top_3:
        ax.set_facecolor("xkcd:pale green")

axs[0, 0].set_ylabel("success count")
axs[1, 0].set_ylabel("success count")

axs.flatten()[-1].axis("off")

plt.show()

# %% [markdown]
# ### Interaction effects plot

# %%
# %%time
fig, axs = plt.subplots(
    nrows=3, ncols=4, sharey=True, constrained_layout=True, figsize=(12, 6)
)
# set palette
palette = it.cycle(sns.color_palette("Paired"))

for i, ax in enumerate(axs.flat):
    (a, b) = effect_level_2_ranked[i]
    c = next(palette)
    sns.regplot(
        x=a,
        y="success",
        data=fit_success_doe[fit_success_doe[b] == fit_success_doe[b].min()],
        label=str(fit_success_doe[b].min()),
        x_estimator=np.mean,
        color=c,
        x_ci=None,
        ci=None,
        truncate=True,
        ax=ax,
    )
    c = next(palette)
    sns.regplot(
        x=a,
        y="success",
        data=fit_success_doe[fit_success_doe[b] == fit_success_doe[b].max()],
        label=str(fit_success_doe[b].max()),
        x_estimator=np.mean,
        color=c,
        x_ci=None,
        ci=None,
        truncate=True,
        ax=ax,
    )
    ax.set_ylabel(None)
    ax.set_axisbelow(True)
    x_majors = [fit_success_doe[a].min(), fit_success_doe[a].max()]
    ax.xaxis.set_major_locator(ticker.FixedLocator(x_majors))
    ax.grid(True, axis="x", which="major", alpha=1, color="w", ls="-")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(True, axis="y", which="major", alpha=0.5, ls="-")
    ax.grid(True, axis="y", which="minor", alpha=0.25, ls="--")
    ax.legend(
        labelspacing=0.25,
        fontsize="x-small",
        title=str(b),
        title_fontsize="small",
        facecolor="white",
        framealpha=0.5,
    )
    if effect_level_2_ranked[i] in effects_top_10:
        ax.set_facecolor("xkcd:pale yellow")
    if effect_level_2_ranked[i] in effects_top_5:
        ax.set_facecolor("xkcd:pale blue")
    if effect_level_2_ranked[i] in effects_top_3:
        ax.set_facecolor("xkcd:pale green")

axs[0, 0].set_ylabel("success count")
axs[1, 0].set_ylabel("success count")
axs[2, 0].set_ylabel("success count")

plt.show()

# %% [markdown]
# ### ANOVA Analysis

# %% [markdown]
# From unreplicated 7-level to 4-replicate 5-level

# %%
# %%time
factors_5l = ["mut_sig", "pop", "k_sur", "mut_pb", "k_par"]
fit_success_5l = fit_res_df.assign(success=fit_res_df.best.le(1e-6))
fit_success_5l = fit_success_5l.groupby(factors, as_index=False).agg(
    {"success": np.sum}
)
fit_success_5l["success"] = fit_success_5l["success"]/40*100

# %%
for factor in factors_5l:
    fac_min = fit_success_5l[factor].min()
    fac_max = fit_success_5l[factor].max()

    fit_success_5l[factor] = fit_success_5l[factor].map(
        lambda v: -1 if v == fac_min else 1
    )

labels_5l = {}
labels_5l[1] = list(factors_5l)
for i in [2, 3, 4, 5]:
    labels_5l[i] = list(it.combinations(labels_5l[1], i))

obs_list_5l = ["success"]

effects_5l = {}

effects_5l[0] = {"x0": [fit_success_5l[obs_list_5l[0]].mean()]}

effects_5l[1] = {}
for key in labels_5l[1]:
    effects_result = []
    for obs in obs_list_5l:
        effects_df = fit_success_5l.groupby(key)[obs].mean()
        result = sum([zz * effects_df.loc[zz] for zz in effects_df.index])
        effects_result.append(result)
    effects_5l[1][key] = effects_result

for c in range(2, 6):
    effects_5l[c] = {}
    for key in labels_5l[c]:
        effects_result = []
        for obs in obs_list_5l:
            effects_df = fit_success_5l.groupby(key)[obs].mean()
            result = sum(
                [
                    np.prod(zz) * effects_df.loc[zz] / (2 ** (len(zz) - 1))
                    for zz in effects_df.index
                ]
            )
            effects_result.append(result)
        effects_5l[c][key] = effects_result

# %%
master_dict_5l = {}
for nvars in effects_5l.keys():
    effect = effects_5l[nvars]
    for k in effect.keys():
        v = effect[k]
        master_dict_5l[k] = nvars, v[0]

# %%
master_df_5l = pd.DataFrame(master_dict_5l).T
master_df_5l.columns = ["interaction"] + obs_list_5l
master_df_5l.head(15)

# %%
n_doe_5l = 4
k_doe_5l = 5

y1_5l = master_df_5l.copy()
y1_5l = y1_5l.iloc[y1_5l[obs_list_5l[0]].abs().argsort].iloc[::-1]
y1_5l = y1_5l.drop("x0")
y1_5l.columns = ["Int Level", "Effect_Estimate"]
y1_5l.index.names = ["Factors"]
y1_5l["Sum_of_Squares"] = (
    y1_5l["Effect_Estimate"] ** 2 * n_doe_5l * (2 ** (k_doe_5l - 2))
)

SS_tot_5l = (fit_success_5l[obs_list_5l[0]] ** 2).sum() - (
    (fit_success_5l[obs_list_5l[0]].sum() ** 2) / len(fit_success_5l[obs_list_5l[0]])
)
SS_err_5l = SS_tot_5l - (y1_5l["Sum_of_Squares"].sum())
y1_5l["%_Contribution"] = y1_5l["Sum_of_Squares"] / SS_tot * 100
effect_estimate_5l = y1_5l.copy()
effect_estimate_5l.index.names = ["Interaction"]
effect_estimate_5l.drop(["Int Level"], axis=1).head(15)

# %%
y1_5l.groupby(["Int Level"]).agg(
    OrderedDict(
        [
            ("Effect_Estimate", "count"),
            ("Sum_of_Squares", "sum"),
            ("%_Contribution", ["sum", "max", "min"]),
        ]
    )
).rename(columns={"Effect_Estimate": "Count", "sum": "total", "count": "total"})

# %% [markdown]
# ANOVA Analysis with F statistics (significance level 5%)

# %%
ANOVA_succ_5l = y1_5l.copy()
ANOVA_succ_5l = ANOVA_succ_5l.drop("Effect_Estimate", axis=1)

# %%
ANOVA_succ_5l["Dgrs. Freedom"] = 1
df_tot_5l = n_doe_5l * 2 ** k_doe_5l - 1
df_err_5l = 2 ** k_doe_5l * (n_doe_5l - 1)

# %%
ANOVA_succ_5l["Mean Sqrs"] = (
    ANOVA_succ_5l["Sum_of_Squares"] / ANOVA_succ_5l["Dgrs. Freedom"]
)
ms_err_5l = SS_err_5l / df_err_5l

ANOVA_succ_5l["F 0"] = ANOVA_succ_5l["Mean Sqrs"] / ms_err_5l
sig_level = 0.05
ANOVA_succ_5l["P-Value"] = 1 - sstats.f.cdf(ANOVA_succ_5l["F 0"], dfn=1, dfd=df_err_5l)
# ANOVA_succ_5l["F critical"] = stats.f.ppf(q=1 - sig_level, dfn=1, dfd=df_tot)
# ANOVA_succ_5l["Significant"] = ANOVA_succ_5l["F ratio"] > ANOVA_succ_5l["F critical"]

# %%
ANOVA_succ_5l.loc["Error"] = [
    "Err",
    SS_err_5l,
    SS_err_5l / SS_tot_5l * 100,
    df_err_5l,
    ms_err_5l,
    "",
    "",
]
ANOVA_succ_5l.loc["Total"] = [
    "Tot",
    SS_tot_5l,
    SS_tot_5l / SS_tot_5l * 100,
    df_tot_5l,
    "",
    "",
    "",
]
ANOVA_succ_5l.loc["Model"] = [
    "Mod",
    SS_tot_5l - SS_err_5l,
    (SS_tot_5l - SS_err_5l) / SS_tot_5l * 100,
    "",
    "",
    "",
    "",
]

# %%
ANOVA_succ_5l[['Sum_of_Squares', '%_Contribution', 'Dgrs. Freedom',
       'Mean Sqrs', 'F 0', 'P-Value']]

# %%
ANOVA_succ_5l.groupby(["Int Level"]).agg(
    OrderedDict(
        [("Int Level", "count"), ("Sum_of_Squares", "sum"), ("%_Contribution", "sum")]
    )
).rename(columns={"Int Level": "Count", "sum": "total", "count": "total"})

# %%
df_show = ANOVA_succ_5l.iloc[np.r_[-1, 0:15, -3, -2]]
print(df_show[['Sum_of_Squares', '%_Contribution', 'Dgrs. Freedom',
       'Mean Sqrs', 'F 0', 'P-Value']].to_latex(float_format=lambda x: '%.1f' % x))

# %%
print(df_show[['Sum_of_Squares', '%_Contribution', 'Dgrs. Freedom',
       'Mean Sqrs', 'F 0', 'P-Value']].to_latex())

# %% [markdown] {"toc-hr-collapsed": false}
# ### Normal probability plots of the effects

# %% [markdown]
# Quantify which effects are not normally distributed, to assist in identifying important variables.

# %%
fig, ax = plt.subplots(figsize=(5, 4))
scatter_options_inv = dict(
    marker="^",
    markerfacecolor="none",
    markeredgewidth=1.75,
    linestyle="none",
    alpha=0,
    zorder=5,
)
scatter_options = dict(
    marker="^",
    markerfacecolor="none",
    markeredgewidth=1.75,
    linestyle="none",
    alpha=0.5,
    zorder=5,
    label="Effect Estimate",
)
line_options = dict(
    dashes=(10, 2, 5, 2), color="r", linewidth=1, zorder=10, label="Best-fit line (Full set)"
)
line_options_5 = dict(
    dashes=(10, 2, 5, 2), color="b", linewidth=1, zorder=10, label="Best-fit line (w/o Top-5)"
)
line_options_10 = dict(
    dashes=(10, 2, 5, 2), color="g", linewidth=1, zorder=10, label="Best-fit line (w/o Top-10)"
)
line_options_15 = dict(
    dashes=(10, 2, 5, 2), color="b", linewidth=.5, zorder=10, label="Best-fit line(w/o Top-15)"
)
fig = probscale.probplot(
    effect_estimate["Effect_Estimate"],
    ax=ax,
    plottype="prob",
    probax="y",
    problabel="Standard Normal Probabilities",
    datalabel="Effect Estimate",
    scatter_kws=scatter_options,
)
fig = probscale.probplot(
    effect_estimate["Effect_Estimate"],
    ax=ax,
    plottype="prob",
    bestfit=True,
    probax="y",
    problabel="Standard Normal Probabilities",
    datalabel="Effect Estimate",
    scatter_kws=scatter_options_inv,
    line_kws=line_options,
)
fig = probscale.probplot(
    effect_estimate["Effect_Estimate"].iloc[5:],
    ax=ax,
    plottype="prob",
    bestfit=True,
    probax="y",
    problabel="Standard Normal Probabilities",
    datalabel="Effect Estimate",
    scatter_kws=scatter_options_inv,
    line_kws=line_options_5,
)
fig = probscale.probplot(
    effect_estimate["Effect_Estimate"].iloc[10:],
    ax=ax,
    plottype="prob",
    bestfit=True,
    probax="y",
    problabel="Standard Normal Probabilities",
    datalabel="Effect Estimate",
    scatter_kws=scatter_options_inv,
    line_kws=line_options_10,
)
fig = probscale.probplot(
    effect_estimate["Effect_Estimate"].iloc[15:],
    ax=ax,
    plottype="prob",
    bestfit=True,
    probax="y",
    problabel="Standard Normal Probabilities",
    datalabel="Effect Estimate",
    scatter_kws=scatter_options_inv,
    line_kws=line_options_15,
)
y_majors = [0.5, 1, 5, 10, 20, 40, 50, 60, 80, 90, 95, 99, 99.5]
y_minors = [0.1, 0.2, 2, 30, 70, 98, 99.8, 99.9]
ax.legend(facecolor="white", fontsize="small", framealpha=0.5)
ax.yaxis.set_major_locator(ticker.FixedLocator(y_majors))
ax.grid(True, axis="y", which="major", alpha=0.5, ls="-")
ax.yaxis.set_minor_locator(ticker.FixedLocator(y_minors))
ax.grid(True, axis="y", which="minor", alpha=0.25, ls="--")
ax.set_ylabel("Standard Normal Probabilities", fontsize="medium")
ax.grid(True, axis="x", which="major", alpha=1, color="w", ls="-")
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.grid(True, axis="x", which="minor", alpha=0.75, color="w", ls="--")
ax.set_xlabel("Effect Estimate", fontsize="medium")
ax.set_axisbelow(True)
plt.show()

# %% [markdown]
# ### Normal probability plot of residuals

# %%
fit_success.head()

# %%
pred_c_0 = effects[0]["x0"][0]

pred_c_mut_sig = effects[1]["mut_sig"][0] / 2
pred_c_pop = effects[1]["pop"][0] / 2
pred_c_k_sur = effects[1]["k_sur"][0] / 2
pred_c_mut_pb = effects[1]["mut_pb"][0] / 2
pred_c_b_ratio = effects[1]["b_ratio"][0] / 2

pred_c_pop_mut_sig = effects[2][("pop", "mut_sig")][0] / 2
pred_c_mut_sig_k_sur = effects[2][("mut_sig", "k_sur")][0] / 2
pred_c_mut_pb_k_sur = effects[2][("mut_pb", "k_sur")][0] / 2

print(pred_c_0)
print(pred_c_mut_sig, pred_c_pop, pred_c_k_sur, pred_c_mut_pb, pred_c_b_ratio)
print(pred_c_pop_mut_sig, pred_c_mut_sig_k_sur,pred_c_mut_pb_k_sur)


# %%
def pred_1(b_ratio, cx_pb, mut_pb, mut_sig, k_par, k_sur):
    pred_1 = (
        pred_c_0
        + pred_c_mut_sig * mut_sig
        + pred_c_k_sur * k_sur
        + pred_c_mut_pb * mut_pb
        + pred_c_b_ratio * b_ratio
    )
    # Error con pop
    return pred_1


def pred_2(b_ratio, cx_pb, mut_pb, mut_sig, k_par, k_sur):
    pred_2 = (
        pred_c_mut_sig_k_sur * mut_sig * k_sur
        + pred_c_mut_pb_k_sur * mut_pb * k_sur
    )
    # Error con pop
    return pred_2


fit_prediction = fit_success.copy().infer_objects()
fit_prediction["pred_1"] = fit_success.apply(
    lambda x: pred_1(x.b_ratio, x.cx_pb, x.mut_pb, x.mut_sig, x.k_par, x.k_sur), axis=1
)
fit_prediction["pred_1"] = fit_prediction["pred_1"] + fit_prediction["pop"] * pred_c_pop
fit_prediction = fit_prediction.assign(res_1=lambda x: x.success - x.pred_1)
fit_prediction["pred_2"] = fit_success.apply(
    lambda x: pred_2(x.b_ratio, x.cx_pb, x.mut_pb, x.mut_sig, x.k_par, x.k_sur), axis=1
)
fit_prediction["pred_2"] = fit_prediction["pred_2"] + fit_prediction["pred_1"]
fit_prediction["pred_2"] = (
    fit_prediction["pred_2"]
    + fit_prediction["pop"] * fit_prediction["mut_sig"] * pred_c_pop_mut_sig
)
fit_prediction = fit_prediction.assign(res_2=lambda x: x.success - x.pred_2)

fit_prediction.head()

# %%
fig, axs = plt.subplots(figsize=(10, 4), ncols=2, sharex=True, constrained_layout=True)
# fig, ax = plt.subplots(figsize=(6,4.5))

scatter_options = dict(
    marker="o",
    markeredgewidth=1.75,
    linestyle="none",
    alpha=0.5,
    zorder=5,
    label="Residual",
)
line_options = dict(
    dashes=(10, 2, 5, 2), color="g", linewidth=1, zorder=10, label="Best-fit line"
)

fig = probscale.probplot(
    fit_prediction["res_1"],
    ax=axs[0],
    plottype="prob",
    probax="y",
    datalabel="Residual",
    bestfit=True,
    scatter_kws=scatter_options,
    line_kws=line_options,
)
fig = probscale.probplot(
    fit_prediction["res_2"],
    ax=axs[1],
    plottype="prob",
    probax="y",
    datalabel="Residual",
    bestfit=True,
    scatter_kws=scatter_options,
    line_kws=line_options,
)

y_majors = [0.5, 1, 5, 10, 20, 40, 50, 60, 80, 90, 95, 99, 99.5]
y_minors = [0.1, 0.2, 2, 30, 70, 98, 99.8, 99.9]

axs[0].set_title("(a) residuals from only-factors regression equation", fontsize="medium")
axs[1].set_title("(b) residuals from second-order regression equation", fontsize="medium")
axs[0].set_ylabel("Standard Normal Probabilities", fontsize="medium")
axs[1].yaxis.set_ticklabels([])

for ax in axs.flat:
    ax.legend(facecolor="white", fontsize="small", framealpha=0.5)
    ax.yaxis.set_major_locator(ticker.FixedLocator(y_majors))
    ax.yaxis.set_minor_locator(ticker.FixedLocator(y_minors))
    ax.grid(True, axis="y", which="major", alpha=0.5, ls="-")
    ax.grid(True, axis="y", which="minor", alpha=0.25, ls="--")
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, axis="x", which="major", alpha=1, color="w", ls="-")
    ax.grid(True, axis="x", which="minor", alpha=0.75, color="w", ls="--")
    ax.set_xlabel("Residual", fontsize="medium")
    ax.set_xlim((-40, 40))
plt.show()

# %% [markdown]
# ### Contour plots of predicted values

# %%
pred_c_0 = effects[0]["x0"][0]

pred_c_mut_sig = effects[1]["mut_sig"][0] / 2
pred_c_pop = effects[1]["pop"][0] / 2
pred_c_k_sur = effects[1]["k_sur"][0] / 2
pred_c_mut_pb = effects[1]["mut_pb"][0] / 2
pred_c_b_ratio = effects[1]["b_ratio"][0] / 2


pred_c_pop_mut_sig = effects[2][("pop", "mut_sig")][0] / 2
pred_c_mut_sig_k_sur = effects[2][("mut_sig", "k_sur")][0] / 2
pred_c_mut_pb_k_sur = effects[2][("mut_pb", "k_sur")][0] / 2

def pred_1_2(pop, b_ratio, cx_pb, mut_pb, mut_sig, k_par, k_sur):
    pred_1_2 = (
        pred_c_0
        
        + pred_c_mut_sig * mut_sig
        + pred_c_pop * pop
        + pred_c_k_sur * k_sur
        + pred_c_mut_pb * mut_pb
        + pred_c_b_ratio * b_ratio
        
        + pred_c_pop_mut_sig * pop * mut_sig
        + pred_c_mut_sig_k_sur * mut_sig * k_sur
        + pred_c_mut_pb_k_sur * mut_pb * k_sur
    )
    return pred_1_2


# %%
# %%time
fig, axs = plt.subplots(
    figsize=(12, 12),
    nrows=3,
    ncols=3,
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

#Variable values
delta = 0.025
mut_sig_x = np.arange(-1.0, 1.0, delta)
pop_y = np.arange(-1.0, 1.0, delta)

k_sur_plot_list = [-1, 0, 1]
mut_pb_plot_list = [-1, 0, 1]

b_ratio_plot = 0
k_par_plot = 0
cx_pb_plot = 0

X, Y = np.meshgrid(mut_sig_x, pop_y)

k_sur_title = [2, 4, 6]
mut_pb_title = [.1, .3, .5]

for i, row in enumerate(axs):
    for j, cell in enumerate(row):
        Z = pred_1_2(
            Y, #pop_plot_list[j],
            b_ratio_plot, #Y,
            cx_pb_plot,
            mut_pb_plot_list[j],
            X,
            k_par_plot,
            k_sur_plot_list[i],
        )
        ax = axs[i, j]
        CS = ax.contour(X, Y, Z, colors="black")
        ax.clabel(CS, inline=1, fontsize=10, fmt='%1.1f')
        if i == 0:
            cell.set_title("at mut_pb=" + str(mut_pb_title[j]), fontsize="medium")
        if i == len(axs) - 1:
            cell.set_xlabel("mut_sig", fontsize="medium")
            mut_sig_map = np.linspace(-0.5, 5, 9)
            mut_sig_map = [round(elem, 1) for elem in mut_sig_map]
            ax.set_xticklabels(mut_sig_map)
        if j == 0:
            cell.set_ylabel(
                "at k_sur=" + str(k_sur_title[i]) + "\n" + "pop", fontsize="medium"
            )
            pop_map = np.linspace(20, 160, 9)
            pop_map = [round(elem, 0) for elem in pop_map]
            ax.set_yticklabels(pop_map)

axs[0, 0].set_ylim((-1, 1))
axs[0, 0].set_xlim((-1, 1))

# %%
# %%time
fig, axs = plt.subplots(
    figsize=(12, 12),
    nrows=3,
    ncols=3,
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

#Variable values
delta = 0.025
mut_sig_x = np.arange(-1.0, 1.0, delta)
pop_y = np.arange(-1.0, 1.0, delta)

k_sur_plot_list = [-1, 0, 1]
mut_pb_plot_list = [-1, 0, 1]

b_ratio_plot = -1
k_par_plot = 0
cx_pb_plot = 0

X, Y = np.meshgrid(mut_sig_x, pop_y)

k_sur_title = [2, 4, 6]
mut_pb_title = [.1, .3, .5]

for i, row in enumerate(axs):
    for j, cell in enumerate(row):
        Z = pred_1_2(
            Y, #pop_plot_list[j],
            b_ratio_plot, #Y,
            cx_pb_plot,
            mut_pb_plot_list[j],
            X,
            k_par_plot,
            k_sur_plot_list[i],
        )
        ax = axs[i, j]
        CS = ax.contour(X, Y, Z, colors="black")
        ax.clabel(CS, inline=1, fontsize=10, fmt='%1.1f')
        if i == 0:
            cell.set_title("at mut_pb=" + str(mut_pb_title[j]), fontsize="medium")
        if i == len(axs) - 1:
            cell.set_xlabel("mut_sig", fontsize="medium")
            mut_sig_map = np.linspace(-0.5, 5, 9)
            mut_sig_map = [round(elem, 1) for elem in mut_sig_map]
            ax.set_xticklabels(mut_sig_map)
        if j == 0:
            cell.set_ylabel(
                "at k_sur=" + str(k_sur_title[i]) + "\n" + "pop", fontsize="medium"
            )
            pop_map = np.linspace(20, 160, 9)
            pop_map = [round(elem, 0) for elem in pop_map]
            ax.set_yticklabels(pop_map)

axs[0, 0].set_ylim((-1, 1))
axs[0, 0].set_xlim((-1, 1))

# %% [markdown]
# # Visualization of data

# %% [markdown]
# Average value of minimum fitness for each generation

# %%
fit_log_file = pickle_dir + file_sufix + "_fit_log_df_80k.xz"
fit_log_df = pd.read_pickle(fit_log_file)

# %%
fit_log_df.head()

# %%
# %%time
g = sns.relplot(
    x="births",
    y="best",
    col="pop",
    row="k_sur",
    hue="b_ratio",
    kind="line",
    data=fit_log_df[fit_log_df["mut_sig"] == 0.5],
)

leg = g._legend
leg.set_bbox_to_anchor([0.65, 0.95])
leg._loc = 1

# %%
