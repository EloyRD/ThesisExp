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
# \title{TESTCASE B - 2-Level 6-Factor Full Factorial (With 30 replicates) - Data Analysis}
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
plt.style.use("bmh")
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

pd.set_option("display.latex.repr", True)
pd.set_option("display.latex.longtable", True)

# %% [markdown]
# # Reading data

# %% [markdown]
# ## Fitness results data

# %% [markdown]
# Reading the Data Frame from a pickle file

# %%
multi_fit = pd.read_pickle("./Data/TEST_B_2L_FitData.gz", compression="gzip")

# %% [markdown]
# Replicates in the sample

# %%
len(multi_fit) / (201)

# %% [markdown]
# ## DOE data and encoded values

# %%
doe = pd.read_pickle("./Data/TEST_B_DOE_data.gz", compression="gzip")
coded_values = pd.read_pickle("./Data/TEST_B_DOE_code.gz", compression="gzip")
coded_values

# %%
doe.head()

# %% [markdown]
# # Visualization of data

# %% [markdown]
# ## Development of minimum (best) fitness

# %% [markdown]
# Fitness after each generation for each of the 1920 replicates

# %%
fig, ax = plt.subplots()
h = ax.hist2d(
    x="generation", y="fitness_min", bins=(100, 160), cmap="gist_heat_r", data=multi_fit
)
ax.set_xlabel("generation")
ax.set_ylabel("fitness_min")
ax.set_xlim(0, 200)
ax.set_ylim(0, 15)
cb = fig.colorbar(h[3], ax=ax)
cb.set_label("count")
plt.tight_layout()

# %%
fig, ax = plt.subplots()
h = ax.hist2d(
    x="generation", y="fitness_std", bins=(100, 60), cmap="gist_heat_r", data=multi_fit
)
ax.set_xlabel("generation")
ax.set_ylabel("fitness_std")
ax.set_xlim(0, 200)
ax.set_ylim(0, 15)
cb = fig.colorbar(h[3], ax=ax)
cb.set_label("count")
plt.tight_layout()

# %% [markdown]
# Average value of minimum fitness for each generation

# %%
sns.lineplot(x="generation", y="fitness_min", data=multi_fit)

# %%
# %%time
hue = "s_sel"
g = sns.relplot(
    x="generation",
    y="fitness_min",
    col="b",
    row="p_sel",
    hue=hue,
    kind="line",
    data=multi_fit[multi_fit["pop_s"] == 160],
)

leg = g._legend
leg.set_bbox_to_anchor([0.65, 0.95])
leg._loc = 1

# %% [markdown]
# ## Final minimum (best) fitness distribution

# %% [markdown]
# Histogram of minimum (best) fitness of final population

# %%
sns.distplot(doe["f_min"], rug=False, kde=False)

# %% [markdown]
# Minimum fitness vs standard deviation (final population)

# %%
hexplot = sns.jointplot(x="f_min", y="f_std", kind="hex", data=doe)
# shrink fig so cbar is visible
plt.subplots_adjust(left=0.1, right=0.8, top=0.8, bottom=0.1)
# make new ax object for the cbar
cbar_ax = hexplot.fig.add_axes([0.85, 0.1, 0.02, 0.6])  # x, y, width, height
cbar = plt.colorbar(cax=cbar_ax)
cbar.set_label("count")
plt.show()

# %%
query = (doe["f_min"] < 10) & (doe["f_std"] < 15)
hexplot = sns.jointplot(
    x="f_min", y="f_std", kind="hex", joint_kws=dict(gridsize=20), data=doe[query]
)
# shrink fig so cbar is visible
plt.subplots_adjust(left=0.1, right=0.8, top=0.8, bottom=0.1)
# make new ax object for the cbar
cbar_ax = hexplot.fig.add_axes([0.85, 0.1, 0.02, 0.6])  # x, y, width, height
cbar = plt.colorbar(cax=cbar_ax)
cbar.set_label("count")
plt.show()

# %% [markdown]
# Minimum fitness vs mean fitness (final population)

# %%
hexplot = sns.jointplot(x="f_min", y="f_mean", kind="hex", data=doe)
# shrink fig so cbar is visible
plt.subplots_adjust(left=0.1, right=0.8, top=0.8, bottom=0.1)
# make new ax object for the cbar
cbar_ax = hexplot.fig.add_axes([0.85, 0.1, 0.02, 0.6])  # x, y, width, height
cbar = plt.colorbar(cax=cbar_ax)
cbar.set_label("count")
plt.show()

# %%
query = (doe["f_min"] < 10) & (doe["f_mean"] < 10)
hexplot = sns.jointplot(
    x="f_min", y="f_mean", kind="hex", joint_kws=dict(gridsize=20), data=doe[query]
)
# shrink fig so cbar is visible
plt.subplots_adjust(left=0.1, right=0.8, top=0.8, bottom=0.1)
# make new ax object for the cbar
cbar_ax = hexplot.fig.add_axes([0.85, 0.1, 0.02, 0.6])  # x, y, width, height
cbar = plt.colorbar(cax=cbar_ax)
cbar.set_label("count")
plt.show()

# %% [markdown] {"toc-hr-collapsed": false}
# # DOE Analisis of Data

# %%
list(doe.columns[0:6])

# %% [markdown] {"toc-hr-collapsed": false}
# ## Defining variables and variable labels

# %%
labels = {}
labels[1] = list(doe.columns[0:6])
for i in [2, 3, 4, 5, 6]:
    labels[i] = list(it.combinations(labels[1], i))

obs_list = list(doe.columns[-4:-1])

for k in labels.keys():
    print(str(k) + " : " + str(labels[k]))
print()
print(obs_list)

# %% [markdown] {"toc-hr-collapsed": false}
# ## Computing Main and Interaction Effects

# %% [markdown] {"toc-hr-collapsed": false}
# ### Constant Effect

# %%
effects = {}

# Start with the constant effect: this is $\overline{y}$
effects[0] = {"x0": [doe["f_min"].mean(), doe["f_max"].mean(), doe["f_mean"].mean()]}
print(effects[0])

# %% [markdown]
# ### Main effect of each variable

# %%
effects[1] = {}
for key in labels[1]:
    effects_result = []
    for obs in obs_list:
        effects_df = doe.groupby(key)[obs].mean()
        result = sum([zz * effects_df.loc[zz] for zz in effects_df.index])
        effects_result.append(result)
    effects[1][key] = effects_result

effects[1]

# %% [markdown]
# ### Interaction effects (2-variable to 6-variable interactions)

# %%
for c in [2, 3, 4, 5, 6]:
    effects[c] = {}
    for key in labels[c]:
        effects_result = []
        for obs in obs_list:
            effects_df = doe.groupby(key)[obs].mean()
            result = sum(
                [
                    np.prod(zz) * effects_df.loc[zz] / (2 ** (len(zz) - 1))
                    for zz in effects_df.index
                ]
            )
            effects_result.append(result)
        effects[c][key] = effects_result


# %%
def printd(d):
    for k in d.keys():
        print("%25s : %s" % (k, d[k]))


for i in range(1, 7):
    printd(effects[i])

# %% [markdown] {"toc-hr-collapsed": false}
# ## Analysis

# %% [markdown] {"toc-hr-collapsed": false}
# ### Analyzing Effects

# %%
print(len(effects))

# %%
master_dict = {}
for nvars in effects.keys():

    effect = effects[nvars]
    for k in effect.keys():
        v = effect[k]
        master_dict[k] = v

master_df = pd.DataFrame(master_dict).T
master_df.columns = obs_list
master_df.head()

# %%
n = 30
k = 6

y1 = master_df[["f_min"]].copy()
y1 = y1.iloc[y1["f_min"].abs().argsort].iloc[::-1]
y1 = y1.drop("x0")
y1.columns = ["Effects_Estimate"]
y1.index.names = ["Factors"]
y1["Sum_of_Squares"] = y1["Effects_Estimate"] ** 2 * n * (2 ** (k - 2))

SS_tot = (doe["f_min"] ** 2).sum() - ((doe["f_min"].sum() ** 2) / len(doe["f_min"]))
SS_err = SS_tot - (y1["Sum_of_Squares"].sum())
y1["%_Contribution"] = y1["Sum_of_Squares"] / SS_tot * 100

# %%
y1.loc["Error"] = [None, SS_err, SS_err / SS_tot * 100]
y1.loc["Total"] = [None, SS_tot, SS_tot / SS_tot * 100]
y1.loc["Model"] = [None, SS_tot - SS_err, (SS_tot - SS_err) / SS_tot * 100]

# %% [markdown]
# Top 10 effects for observable 'minimum fitness (final population)':

# %%
y1.iloc[np.r_[-1, 0:9, -3, -2]]

# %% [markdown]
# ### ANOVA Analysis

# %% [markdown]
# ANOVA Analysis with F statistics (significance level 5%)

# %%
ANOVA_y1 = y1.copy()
ANOVA_y1 = ANOVA_y1.drop("Effects_Estimate", axis=1)
ANOVA_y1["Dgrs. Freedom"] = 1
df_tot = len(doe["f_min"]) - 1
df_err = df_tot - len(master_df)

ANOVA_y1["Mean Sqrs"] = ANOVA_y1["Sum_of_Squares"] / 1
ms_err = SS_err / df_err

ANOVA_y1["F ratio"] = ANOVA_y1["Mean Sqrs"] / ms_err
sig_level = 0.05
ANOVA_y1["F critical"] = stats.f.ppf(q=1 - sig_level, dfn=1, dfd=df_tot)
ANOVA_y1["Significant"] = ANOVA_y1["F ratio"] > ANOVA_y1["F critical"]

# %%
df_show = ANOVA_y1.iloc[np.r_[-1, 0:10, -3, -2]]
df_show

# %% [markdown]
# ### Main effects plot

# %% [markdown]
# Colors represent if factor is in the top 3 (green), top 5 (blue), top 10 (yellow)

# %%
variable = ["pop_s", "b", "mut_p", "mut_s", "p_sel", "s_sel"]
f, axs = plt.subplots(1, 6, figsize=(18, 3), sharey=True)
x_ci = None
for i in range(len(variable)):
    sns.regplot(
        x=variable[i],
        y="f_min",
        data=doe,
        x_estimator=np.mean,
        x_ci=x_ci,
        ci=None,
        truncate=True,
        ax=axs[i],
    )
for ax in axs.flat:
    ax.set_ylabel(None)
axs[0].set_ylabel("min_fitness")

# Top 3
axs[5].set_facecolor("xkcd:pale green")
axs[4].set_facecolor("xkcd:pale green")

# Top 5
axs[0].set_facecolor("xkcd:pale blue")

# Top 10
axs[1].set_facecolor("xkcd:pale yellow")
axs[2].set_facecolor("xkcd:pale yellow")

plt.tight_layout()

# %% [markdown]
# ### Interaction effects plot

# %%
# %%time
factors = ["pop_s", "b", "mut_p", "mut_s", "p_sel", "s_sel"]
f, axs = plt.subplots(6, 6, figsize=(12, 12), sharey=True, sharex=True)
x_ci = None

# set palette
palette = it.cycle(sns.color_palette("Paired"))

for i in range(len(factors)):
    for j in range(len(factors)):
        yy = factors[j]

        c = next(palette)
        sns.regplot(
            x=factors[i],
            y="f_min",
            data=doe[doe[yy] == -1],
            label="-1",
            x_estimator=np.mean,
            color=c,
            x_ci=x_ci,
            ci=None,
            truncate=True,
            ax=axs[j, i],
        )
        c = next(palette)
        sns.regplot(
            x=factors[i],
            y="f_min",
            data=doe[doe[yy] == 1],
            label="1",
            x_estimator=np.mean,
            color=c,
            x_ci=x_ci,
            ci=None,
            truncate=True,
            ax=axs[j, i],
        )

        # axs[j,i].legend(title=yy,facecolor='white')

        if i == j:
            axs[j, i].clear()

for ax in axs.flat:
    ax.set_ylabel(None)
    ax.set_xlabel(None)

axs[0, 0].set_xlim((-1.1, 1.1))
axs[0, 0].set_ylim((0, 20))


for i in range(len(factors)):
    axs[i, 0].set_ylabel("min_fitness")
    axs[-1, i].set_xlabel(factors[i])
    legend_elements = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=-1,
            markerfacecolor=next(palette),
            markersize=10,
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=1,
            markerfacecolor=next(palette),
            markersize=10,
        ),
    ]
    axs[i, i].legend(
        handles=legend_elements, loc="center", title=factors[i], facecolor="white"
    )

# Top 3
axs[4, 5].set_facecolor("xkcd:pale green")
axs[5, 4].set_facecolor("xkcd:pale green")

# Top 5
axs[0, 5].set_facecolor("xkcd:pale blue")
axs[5, 0].set_facecolor("xkcd:pale blue")

# Top 10
axs[0, 4].set_facecolor("xkcd:pale yellow")
axs[4, 0].set_facecolor("xkcd:pale yellow")
axs[2, 4].set_facecolor("xkcd:pale yellow")
axs[4, 2].set_facecolor("xkcd:pale yellow")

plt.tight_layout()
plt.show()

# %% [markdown] {"toc-hr-collapsed": false}
# ### Normal probability plots of the effects

# %% [markdown]
# Quantify which effects are not normally distributed, to assist in identifying important variables.

# %%
fig, ax = plt.subplots(figsize=(14, 4))

stats.probplot(y1.iloc[0:-3]["Effects_Estimate"], dist="norm", plot=ax)
plt.show()

# %%
