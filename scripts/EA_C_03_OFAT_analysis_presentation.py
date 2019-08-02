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

# %%
factors_df.to_latex()

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

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 206}, "colab_type": "code", "executionInfo": {"elapsed": 8322, "status": "ok", "timestamp": 1561564959234, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "2qhtoq1nJxkH", "outputId": "32af5e4f-7546-45ce-f474-7b467cb0fa57"}
fit_plot.head()

# %% [markdown] {"colab_type": "text", "id": "AeQO1xmeJxkm", "toc-hr-collapsed": false}
# # Visualization

# %% [markdown] {"colab_type": "text", "id": "H0JIPY50Jxkr"}
# Factors to iterate in the visualization

# %% {"colab": {"base_uri": "https://localhost:8080/", "height": 35}, "colab_type": "code", "executionInfo": {"elapsed": 8300, "status": "ok", "timestamp": 1561564959237, "user": {"displayName": "Eloy Ruiz Donayre", "photoUrl": "https://lh5.googleusercontent.com/-eCyR1x8lRb0/AAAAAAAAAAI/AAAAAAAAABI/hEiDwzQwWs0/s64/photo.jpg", "userId": "04104170300566764648"}, "user_tz": -120}, "id": "Y6xJR_JrJxkw", "outputId": "368ebca8-ffae-4988-84cf-cb9639f6e2b7"}
factors = list(factors_df.index.array)
print(factors)
fact = list(fit_plot.columns)
fact = fact[1:8]

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
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12, 2))
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
        ci=68,
        ax=ax,
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
    ax.set_ylim((0, 5.1))
    ax.set_ylabel("fitness", fontsize="medium")
    ax.minorticks_on()
    ax.set_yticks([1, 3, 5], minor=True)
    ax.grid(True, axis="y", which="minor", alpha=0.5)

    plt.show()

# %%
# %%time
for exp_n in range(6, 7):
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12, 2))
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
        ci=68,
        ax=ax,
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
