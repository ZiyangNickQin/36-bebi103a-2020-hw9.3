# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import os, sys, subprocess
data_path = r"c://users/qrlin/hw9"
# ------------------------------

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import numba

import iqplot

import colorcet
import bokeh.io
import holoviews as hv
bokeh.io.output_notebook()
import scipy
from scipy import stats
import math
# -

df = pd.read_csv(os.path.join(data_path, "gardner_time_to_catastrophe_dic_tidy.csv"), comment="#")


@numba.njit
def draw_bs_sample(data):
    """Draw a bootstrap sample from a 1D data set."""
    return np.random.choice(data, size=len(data))


# +
def draw_bs_reps(data, stat_fun, size=1):
    """Draw boostrap replicates computed with stat_fun from 1D data set."""
    return np.array([stat_fun(draw_bs_sample(data)) for _ in range(size)])


@numba.njit
def draw_bs_reps_mean(data, size=1):
    """Draw boostrap replicates of the mean from 1D data set."""
    out = np.empty(size)
    for i in range(size):
        out[i] = np.mean(draw_bs_sample(data))
    return out


# -

df["Fluorescent group"] = df["labeled"].apply(lambda x: "Fluorescently labeled" if x else "Control")

# Extract the values of columns in dataframe
sample_ctrl = df.loc[df["Fluorescent group"] == "Control", "time to catastrophe (s)"].values
sample_label = df.loc[df["Fluorescent group"] == "Fluorescently labeled", "time to catastrophe (s)"].values
# Calculate the bootstrap mean of different labeled groups
bs_reps_mean_ctrl = draw_bs_reps_mean(sample_ctrl, size=10000)
bs_reps_mean_label = draw_bs_reps_mean(sample_label, size=10000)


# +
@numba.njit
def draw_perm_sample(x, y):
    """Generate a permutation sample."""
    concat_data = np.concatenate((x, y))
    np.random.shuffle(concat_data)

    return concat_data[:len(x)], concat_data[len(x):]

def draw_perm_reps(x, y, stat_fun, size=1):
    """Generate array of permuation replicates."""
    return np.array([stat_fun(*draw_perm_sample(x, y)) for _ in range(size)])

@numba.njit
def draw_perm_reps_diff_mean(x, y, size=1):
    """Generate array of permuation replicates."""
    out = np.empty(size)
    for i in range(size):
        x_perm, y_perm = draw_perm_sample(x, y)
        out[i] = np.mean(x_perm) - np.mean(y_perm)

    return out


# +
@numba.njit
def draw_bs_sample(data):
    """Draw a bootstrap sample from a 1D data set."""
    return np.random.choice(data, size=len(data))

# Set up Numpy arrays for convenience (also much better performance)
sample_ctrl = df.loc[df["Fluorescent group"] == "Control", "time to catastrophe (s)"].values
sample_label = df.loc[df["Fluorescent group"] == "Fluorescently labeled", "time to catastrophe (s)"].values

# ECDF values for plotting
ctrl_ecdf = np.arange(1, len(sample_ctrl)+1) / len(sample_ctrl)
label_ecdf = np.arange(1, len(sample_label)+1) / len(sample_label)

p = iqplot.ecdf(
    data=df,
    q='time to catastrophe (s)',
    cats=['labeled'],
    style='staircase',
    width=500,
    height=500,
    title = "time to catastrophe for labeled and unlabeled MT",
    conf_int = True        #default 2.5,97.5
)

p.legend.location = 'bottom_right'
p.legend.title = "The number of CTDr"

# Make 100 bootstrap samples and plot them
for _ in range(100):
    bs_ctrl = draw_bs_sample(sample_ctrl)
    bs_label = draw_bs_sample(sample_label)

    # Add semitransparent ECDFs to the plot
    p.circle(np.sort(bs_ctrl), ctrl_ecdf, color=colorcet.b_glasbey_category10[0], alpha=0.02)
    p.circle(np.sort(bs_label), label_ecdf, color=colorcet.b_glasbey_category10[1], alpha=0.02)


# -

def show_bs_con_int():
    bokeh.io.show(p)


show_bs_con_int()
