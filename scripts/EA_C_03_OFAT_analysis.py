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

# %%

# %% [raw] {"colab_type": "raw", "id": "jzbYate3JxeJ"}
# \author{Eloy Ruiz-Donayre}
# \title{TESTCASE B - One-Factor-at-a-Time Analysis}
# \date{\today}
# \maketitle

# %% [raw] {"colab_type": "raw", "id": "cOYS7JxXJxeq"}
# \tableofcontents

# %% [markdown] {"colab_type": "text", "id": "0BmCMwHXJxe9"}
# # Preliminaries

# %% [markdown]
# This commands are used in Google Colab:

# %% [markdown] {"colab": {"base_uri": "https://localhost:8080/", "height": 53}, "colab_type": "code", "executionInfo": {"elapsed": 4917, "status": "ok", "timestamp": 1561564955582, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "NyS-yVCxJxfK", "outputId": "aee402f2-7489-4bef-eae1-9318286da1dc"}
# !pip install deap

# %% [markdown] {"colab": {"base_uri": "https://localhost:8080/", "height": 35}, "colab_type": "code", "executionInfo": {"elapsed": 4894, "status": "ok", "timestamp": 1561564955586, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "va1gdtFzJxgF", "outputId": "b6d78934-981e-4c90-a8a1-b451815fac6c"}
# from google.colab import drive
# drive.mount("/content/gdrive")

# %% [markdown] {"colab_type": "text", "id": "xJQmgTMjJxgn"}
# Importing python packages and setting display parameters

# %% {"colab": {}, "colab_type": "code", "id": "p89ld-m_Jxgu"}
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

import pandas as pd
import statistics as stats


# %% {"colab": {}, "colab_type": "code", "id": "Y0nwi14cJxhA"}
# %matplotlib inline
# #%config InlineBackend.figure_format = "retina"

plt.style.use("default")
plt.style.use("bmh")
# plt.rcParams.update({"figure.autolayout": True})
plt.rcParams["figure.figsize"] = (12, 9)
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.dpi"] = 100

# %% {"colab": {}, "colab_type": "code", "id": "Y0nwi14cJxhA"}
pd.set_option("display.latex.repr", True)

# %% {"colab": {}, "colab_type": "code", "id": "Y0nwi14cJxhA"}
pd.set_option("display.latex.longtable", True)

# %% [markdown]
# This sets the work directory for the pickle files

# %% {"colab": {}, "colab_type": "code", "id": "4xI6Ra3OJxhW"}
pickle_dir = "./pickle/"
file_sufix = "C_02"
pickle_dir + file_sufix

# %% [markdown]
# Run this when working in Google Colab:

# %% [markdown] {"colab": {}, "colab_type": "code", "id": "4xI6Ra3OJxhW"}
# pickle_dir = "/content/gdrive/My Drive/Colab Notebooks/thesis/"
# pickle_dir + file_sufix

# %% [markdown] {"colab_type": "text", "id": "BoVikt-fJxhu", "toc-hr-collapsed": false}
# # Experiment parameters

# %% [markdown] {"colab_type": "text", "id": "qUJEBU1rJxh2", "toc-hr-collapsed": false}
# ## Common parameters

# %% {"colab": {}, "colab_type": "code", "id": "fmS6tuLkJxh-"}
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

# %% [markdown] {"colab_type": "text", "id": "nPk0bL4BJxiR"}
# ### Factor levels

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 300}, "colab_type": "code", "executionInfo": {"elapsed": 5671, "status": "ok", "timestamp": 1561564956473, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "4Ynwjdw7JxiU", "outputId": "8e4124bd-8f3f-4956-ea5c-20a90740f56f"}
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

# %% [markdown] {"colab_type": "text", "id": "gN150LMcJxiq"}
# # Data Analysis

# %% [markdown] {"colab_type": "text", "id": "yRdHZ176Jxiv"}
# Reading the dataframes of the values to plot and the final results from the files

# %% {"colab": {}, "colab_type": "code", "id": "ULXBS8rpJxi1"}
fit_df_file = pickle_dir + file_sufix + "_fit_df.xz"
fit_fin_df_file = pickle_dir + file_sufix + "_fit_fin_df.xz"
fit_30k_df_file = pickle_dir + file_sufix + "_fit_30k_df.xz"
fit_60k_df_file = pickle_dir + file_sufix + "_fit_60k_df.xz"

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 53}, "colab_type": "code", "executionInfo": {"elapsed": 8368, "status": "ok", "timestamp": 1561564959226, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "Fc-BMbbEJxjF", "outputId": "1fd309fe-5c15-404d-9b34-7e8d65f87acb"}
# %time
fit_plot = pd.read_pickle(fit_df_file)

fit_30k = pd.read_pickle(fit_30k_df_file)
fit_60k = pd.read_pickle(fit_60k_df_file)
fit_120k = pd.read_pickle(fit_fin_df_file)

query_exact = fit_30k["best"] < 1e-6
fit_30k_exact = fit_30k[query_exact]
query_exact = fit_60k["best"] < 1e-6
fit_60k_exact = fit_60k[query_exact]
query_exact = fit_120k["best"] < 1e-6
fit_120k_exact = fit_120k[query_exact]

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 53}, "colab_type": "code", "executionInfo": {"elapsed": 8345, "status": "ok", "timestamp": 1561564959229, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "ZQd5XhJ9Jxjb", "outputId": "6e46b068-eb1c-4923-f01e-198a41e3188b"}
print(len(fit_plot))
print(len(fit_30k) / 40)
print(len(fit_60k) / 40)
print(len(fit_120k) / 40)

# %%
fit_30k = fit_30k[['exp', 'pop', 'b_ratio', 'cx_pb', 'mut_pb', 'mut_sig', 'k_par', 'k_sur',
       'rep', 'births','best']]
fit_30k['births'] = 30000
fit_30k.head().append(fit_30k.tail())

# %%
fit_60k = fit_60k[['exp', 'pop', 'b_ratio', 'cx_pb', 'mut_pb', 'mut_sig', 'k_par', 'k_sur',
       'rep', 'births','best']].reset_index(drop=True)
fit_60k['births'] = 60000
fit_60k.head().append(fit_60k.tail())

# %%
fit_120k = fit_120k[['exp', 'pop', 'b_ratio', 'cx_pb', 'mut_pb', 'mut_sig', 'k_par', 'k_sur',
       'rep', 'births','best']].reset_index(drop=True)
fit_120k['births'] = 120000
fit_120k = fit_120k
fit_120k.head().append(fit_120k.tail())

# %%
fit_results = fit_30k.copy()
fit_results = fit_results.append(fit_60k, sort=False)
fit_results = fit_results.append(fit_120k, sort=False)
fit_results = fit_results.reset_index(drop=True)

# %%
fit_results.head()

# %%
x_y_z = fit_120k.groupby(['exp', 'pop', 'b_ratio', 'cx_pb', 'mut_pb', 'mut_sig', 'k_par', 'k_sur'])[["best"]].size()
x_y_z = x_y_z.reset_index()
x_y_z["Par Conf"] = x_y_z.index
#x_y_z.index.names=["Par Conf"]
x_y_z = x_y_z.rename(columns={0:"Replicates"})
x_y_z = x_y_z[['exp', 'Par Conf', 'pop', 'b_ratio', 'cx_pb', 'mut_pb', 'mut_sig', 'k_par', 'k_sur',
       'Replicates']]
x_y_z

# %%
bins = [-1e-8, 1e-6, 1, 5, 6, 7, float("inf")]

# %%
a_b_c = fit_results.copy()
a_b_c["bins"] = pd.cut(a_b_c["best"], bins)
a_b_c = a_b_c.merge(x_y_z[["exp", 'pop', 'b_ratio', 'cx_pb', 'mut_pb', 'mut_sig', 'k_par', 'k_sur', 'Par Conf']], 
                    on=["exp", 'pop', 'b_ratio', 'cx_pb', 'mut_pb', 'mut_sig', 'k_par', 'k_sur'])
a_b_c = a_b_c[["exp", "Par Conf", "rep", "births", "bins"]]
a_b_c = a_b_c.groupby(['exp','Par Conf', 'bins', "births"]).size().unstack(level=-1, fill_value=0)
a_b_c = a_b_c.reset_index()
a_b_c.sort_values(by=[120000], ascending=False)

# %%
d_e_f = fit_results.copy()
d_e_f["bins"] = pd.cut(d_e_f["best"], bins)
d_e_f = d_e_f.merge(x_y_z[["exp", 'pop', 'b_ratio', 'cx_pb', 'mut_pb', 'mut_sig', 'k_par', 'k_sur', 'Par Conf']], 
                    on=["exp", 'pop', 'b_ratio', 'cx_pb', 'mut_pb', 'mut_sig', 'k_par', 'k_sur'])
d_e_f = d_e_f[["exp", "Par Conf", "rep", "births", "bins"]]
d_e_f = d_e_f.groupby(['exp','Par Conf', "births", 'bins']).size().unstack(level=-1, fill_value=0)
d_e_f

# %%
kkk = fit_results.copy()
query = kkk["best"] < 1e-6
kkk = kkk[query]
kkk["best"].min()

# %%
t_u_v = fit_results.copy()
t_u_v = t_u_v.merge(x_y_z[["exp", 'pop', 'b_ratio', 'cx_pb', 'mut_pb', 'mut_sig', 'k_par', 'k_sur', 'Par Conf']], 
                    on=["exp", 'pop', 'b_ratio', 'cx_pb', 'mut_pb', 'mut_sig', 'k_par', 'k_sur'])
t_u_v = t_u_v[['exp', 'Par Conf', 'pop', 'b_ratio', 'cx_pb', 'mut_pb', 'mut_sig', 'k_par', 'k_sur',
       'rep', 'births', 'best']]
t_u_v = t_u_v.assign(Successful= lambda x:x["best"]<1e-6)
t_u_v = t_u_v.groupby(["exp", "Par Conf", 'pop', 'b_ratio', 'cx_pb', 'mut_pb', 'mut_sig', 'k_par', 'k_sur', "births"])
t_u_v = t_u_v["Successful"].sum().reset_index()
t_u_v["Successful"] = t_u_v["Successful"]/40*100
t_u_v

# %% [markdown] {"colab_type": "text", "id": "AeQO1xmeJxkm", "toc-hr-collapsed": false}
# # Visualization

# %% [markdown] {"colab_type": "text", "id": "H0JIPY50Jxkr"}
# Factors to iterate in the visualization

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 35}, "colab_type": "code", "executionInfo": {"elapsed": 8300, "status": "ok", "timestamp": 1561564959237, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "Y6xJR_JrJxkw", "outputId": "368ebca8-ffae-4988-84cf-cb9639f6e2b7"}
factors = list(factors_df.index.array)
print(factors)
fact = list(fit_plot.columns)
fact = fact[1:8]

# %%
for i in range(1,8):
    query= t_u_v["exp"] == i
    plot_data = t_u_v[query]
    fig, ax = plt.subplots(figsize=(6,4))

    palette = sns.color_palette("tab10", 3)
    sns.lineplot(x=factors[i-1], y="Successful", hue="births", style="births", markers=True, dashes=False, palette=palette, data=plot_data, ax=ax)
    #ax.legend(labelspacing=0.25, fontsize="x-small", ncol=2,           title="Experiment "+ str(exp_n + 1) + "\nFactor '" + exp_factor + "'",            title_fontsize="small", facecolor="white",framealpha=0.5,loc=1,)
    ax.legend(labelspacing=0.25, ncol=2, facecolor="white", framealpha=0.5, fontsize="x-small",  
              title="Experiment "+ str(i) + "\nFactor '" + factors[i-1] + "'", title_fontsize="small", 
              )
    ax.set_ylim(0, 100)
    ax.set_xlabel(factors[i-1], fontsize="medium")
    ax.set_ylabel("Success(%)", fontsize="medium")

    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(True, axis="x", which="major", alpha=1, color="w", ls="-")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.grid(True, axis="y", which="major", alpha=0.5, ls="-")
    ax.grid(True, axis="y", which="minor", alpha=0.25, ls="--")

# %% {"colab": {}, "colab_type": "code", "id": "mS5oVCJDJxlN"}
formatter1 = EngFormatter(places=0, sep=u"\N{THIN SPACE}")  # U+2009

# %% [markdown] {"colab_type": "text", "id": "TQkJhjTpJxl3", "toc-hr-collapsed": true}
# ## Histograms

# %%
list_df = [
    "fit_30k",
    "fit_60k",
    "fit_120k",
    "fit_30k_exact",
    "fit_60k_exact",
    "fit_120k_exact",
]
list_texts = [
    "(a) at 30 K",
    "(b) at 60 K",
    "(c) at 120 K",
    "(d) at 30 K",
    "(e) at 60 K",
    "(f) at 120 K",
]

# %%
# %%time
for exp_n in range(7):
    fig, axs = plt.subplots(
        nrows=2,
        ncols=3,
        sharey="row",
        sharex="row",
        constrained_layout=True,
        figsize=(12, 4),
    )

    for i, ax in enumerate(axs.flat):
        query1 = eval(list_df[i])["exp"] == (exp_n + 1)
        exp_data = eval(list_df[i])[query1]
        exp_factor = factors[exp_n]
        exp_levels = factors_df["Levels"][exp_n]
        exp_levels.sort()

        hist_label = []
        hist_data = []
        for k in range(len(exp_levels)):
            query2 = exp_data[exp_factor] == exp_levels[k]
            hist_data.append(exp_data[query2]["best"].tolist())
            hist_label.append(exp_levels[k])

        if i <= 2:
            bins = np.linspace(-0.01, 6.99, 7 + 1)
        else:
            bins = np.linspace(-1e-8, 1e-6 - 1e-8, num=6)
        ax.hist(hist_data, bins, label=hist_label)
        ax.vlines(
            bins, 0, 40, colors="xkcd:teal", linestyles=(0, (5, 10)), linewidths=0.75
        )
        ax.set_title(list_texts[i], fontsize="medium")
        ax.set_axisbelow(True)
        ax.minorticks_on()
        ax.set_yticks([5, 15, 25, 35], minor=True)
        ax.grid(True, axis="y", which="minor", alpha=0.5)

    for i, row in enumerate(axs):
        for j, cell in enumerate(row):
            if i == 0:
                cell.set_ylim(None, 40.5)
                if j == 2:
                    cell.legend(
                        labelspacing=0.25,
                        fontsize="x-small",
                        ncol=2,
                        title="Experiment "
                        + str(exp_n + 1)
                        + "\nFactor '"
                        + exp_factor
                        + "'",
                        title_fontsize="small",
                        facecolor="white",
                        framealpha=0.5,
                        loc=1,
                    )
            if i == 1:
                cell.set_ylim(None, 25.5)
                cell.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
            if i == len(axs) - 1:
                cell.set_xlabel("fitness", fontsize="medium")
            if j == 0:
                cell.set_ylabel("count", fontsize="medium")

    # fig.suptitle("Histograms of replicate's best fitness per experiment factor's levels at different birth counts", fontsize='large')
    plt.show()

# %% [markdown] {"colab_type": "text", "id": "57UaGy0FJxph", "toc-hr-collapsed": false}
# ## Average fitness development

# %% [markdown] {"colab_type": "text", "id": "aLWTA9usJxq9"}
# ### Agregated per experiment and level

# %% [markdown] {"colab_type": "text", "id": "Ivx8qx2ZJxrB"}
# Development of minimum (best) fitness for each level of each experiment

# %%
# %%time
for exp_n in range(7):
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 2))
    if exp_n in [4]:
        palette = sns.color_palette("tab10", 6)
    else:
        palette = sns.color_palette("tab10", 5)
    query = fit_plot["exp"] == (exp_n + 1)
    data_exp = fit_plot[query]
    sns.lineplot(
        x="births",
        y="best",
        style=factors[exp_n],
        hue=factors[exp_n],
        palette=palette,
        data=data_exp,
        ci=None,
        ax=ax,
        estimator=np.min,
    )
    ax.set_axisbelow(True)
    ax.legend(
        labelspacing=0.25,
        fontsize="x-small",
        ncol=2,
#        title="Experiment " + str(exp_n + 1),
#        title_fontsize="small",
        facecolor="white",
        framealpha=0.5,
        loc=0,
    )
    ax.xaxis.set_major_formatter(formatter1)
    ax.set_xlim((0, 120e3))
    ax.set_xlabel("births", fontsize="medium")
    ax.set_ylim((0, .000001))
    ax.set_ylabel("fitness", fontsize="medium")
    ax.minorticks_on()
#    ax.set_yticks([1, 3, 5], minor=True)
    ax.grid(True, axis="y", which="minor", alpha=0.5)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    
    plt.show()

# %%
# %%time
for exp_n in range(7):
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 2))
    if exp_n in [4]:
        palette = sns.color_palette("tab10", 6)
    else:
        palette = sns.color_palette("tab10", 5)
    query = fit_plot["exp"] == (exp_n + 1)
    data_exp = fit_plot[query]
    sns.lineplot(
        x="births",
        y="best",
        style=factors[exp_n],
        hue=factors[exp_n],
        palette=palette,
        data=data_exp,
        ci=None,
        #        ci=68,
        ax=ax,
        estimator=np.mean,
    )
    ax.set_axisbelow(True)
    ax.legend(
        labelspacing=0.25,
        fontsize="x-small",
        ncol=2,
#        title="Experiment " + str(exp_n + 1),
#        title_fontsize="small",
        facecolor="white",
        framealpha=0.5,
        loc=0,
    )
    ax.xaxis.set_major_formatter(formatter1)
    ax.set_xlim((0, 120e3))
    ax.set_xlabel("births", fontsize="medium")
    ax.set_ylim((0, 4.1))
    ax.set_ylabel("fitness", fontsize="medium")
    ax.minorticks_on()
    ax.set_yticks([1, 3], minor=True)
    ax.grid(True, axis="y", which="minor", alpha=0.5)

    plt.show()

# %%
# %%time
for exp_n in range(6, 7):
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 2))
    if exp_n in [4]:
        palette = sns.color_palette("tab10", 6)
    else:
        palette = sns.color_palette("tab10", 5)
    query = fit_plot["exp"] == (exp_n + 1)
    data_exp = fit_plot[query]
    sns.lineplot(
        x="births",
        y="best",
        style=factors[exp_n],
        hue=factors[exp_n],
        palette=palette,
        data=data_exp,
        ci=None,
        ax=ax,
        estimator=np.mean,
    )
    ax.set_axisbelow(True)
    ax.legend(
        labelspacing=0.25,
        fontsize="x-small",
        ncol=2,
        title="Experiment " + str(exp_n + 1),
        title_fontsize="small",
        facecolor="white",
        framealpha=0.5,
        loc=9,
    )
    ax.xaxis.set_major_formatter(formatter1)
    ax.set_xlim((0, 120e3))
    ax.set_xlabel("births", fontsize="medium")
    ax.set_ylabel("fitness", fontsize="medium")
    ax.minorticks_on()
    ax.set_yticks([1, 3, 5], minor=True)
    ax.grid(True, axis="y", which="minor", alpha=0.5)

    plt.show()

# %% [markdown] {"colab_type": "text", "id": "LtQjlE5zJxpn"}
# ### Agregated per experiment

# %% [markdown] {"colab_type": "text", "id": "HsdyZrgeJxps"}
# Development of average minimum (best) fitness for each experiment (Each experiment has one factor at different levels)

# %%
# %%time
palette = it.cycle(sns.color_palette("tab10"))
for exp_n in range(7):
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12, 2))
    c = next(palette)
    query = fit_plot["exp"] == (exp_n + 1)
    data_exp = fit_plot[query]
    bin_min = data_exp["births"].min()
    bin_max = data_exp["births"].max()
    bins = np.linspace(bin_min, bin_max, 361)
    plot_centers = (bins[:-1] + bins[1:]) / 2
    plot_centers = plot_centers.astype(int)
    data_exp["range"] = pd.cut(
        data_exp.births, bins, labels=plot_centers, include_lowest=True
    )
    sns.lineplot(
        x="range",
        y="best",
        label=("Experiment " + str(exp_n + 1) + ": varying of " + factors[exp_n]),
        color=c,
        data=data_exp,
        ci=68,
        ax=ax,
    )
    ax.set_axisbelow(True)
    ax.legend(facecolor="white", fontsize="small", framealpha=0.5)
    ax.xaxis.set_major_formatter(formatter1)
    ax.set_xlim((0, 120e3))
    ax.set_xlabel("births", fontsize="medium")
    ax.set_ylim((0, 5.1))
    ax.set_ylabel("fitness", fontsize="medium")
    ax.minorticks_on()
    ax.set_yticks([1, 3, 5], minor=True)
    ax.grid(True, axis="y", which="minor", alpha=0.5)

    plt.show()

# %%
