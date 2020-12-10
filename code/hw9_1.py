# -*- coding: utf-8 -*-
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
data_path = "../data/"
# ------------------------------

import warnings
import os, sys, subprocess
import numpy as np
import pandas as pd
from random import randint
import scipy.optimize
import scipy.stats as st
import tqdm
import numba
import bebi103
import bokeh.io
import bokeh.plotting
from bokeh.layouts import gridplot
bokeh.io.output_notebook()
import iqplot
import holoviews as hv
hv.extension('bokeh')

bebi103.hv.set_defaults()

# +
df = pd.read_csv(os.path.join(data_path, "gardner_mt_catastrophe_only_tubulin.csv"), comment="#")
col_names = {"12 uM": 12, "7 uM": 7, "9 uM": 9, "10 uM": 10, "14 uM": 14}
df_renamed = df.rename(columns=col_names)
df_renamed = df_renamed.melt()

col_names = {"variable": "tubulin concentration", "value" : "time to catastrophe (s)"}
df_renamed = df_renamed.rename(columns = col_names)
df_tidy = df_renamed.sort_values(by=['tubulin concentration'])
df_tidy = df_tidy.dropna()


# -

def show_ecdfs_dif_con():
    p = iqplot.ecdf(
        data=df_tidy,
        q='time to catastrophe (s)',
        cats=['tubulin concentration'],
        style='staircase'
    )

    p.legend.location = 'bottom_right'
    p.legend.title = "tubulin concentration (uM)"

    bokeh.io.show(p)


# +
rg = np.random.default_rng()

def get_gamma_paras(ar):
    """Get parameter maximum likelihood 
    estimates for gamma distribution"""
    m = np.mean(ar)
    v = np.std(ar)**2
    alpha_mle = m**2/v
    beta_mle = m/v
    return (alpha_mle,beta_mle)


# -

tubulin_12microM = df["12 uM"].values
alpha_beta = get_gamma_paras(tubulin_12microM)
alpha_mle, beta_mle = alpha_beta

gamma_samples = np.array(
    [rg.gamma(alpha_mle, 1/beta_mle, size=len(tubulin_12microM)) for _ in range(100000)]
)


def show_qq_gamma_12():
    p = bebi103.viz.qqplot(
        data=tubulin_12microM,
        samples=gamma_samples,
        x_axis_label="time to catastrophe (s)",
        y_axis_label="time to catastrophe (s)",
        title="QQ plot for gamma distribution model of 12 uM tubulin concentration"
    )

    bokeh.io.show(p)


def log_like_poisson_exp(params, t):
    
    beta1, db = params
    
    if beta1 < 0 or db < 0:
        return -np.inf
    
    log_like = len(t)*np.log(beta1) + len(t)*np.log(beta1 + db) - len(t)*np.log(db) - beta1*np.sum(t) + np.sum(np.log(1 - np.exp(-db*t)))
    
    return log_like


def double_exp_mle(ell):
    """Compute MLE for parameters in double expenontial model."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, ell: -log_like_poisson_exp(params, ell),
            x0=np.array([0.005, 0.001]),
            args=(ell),
            method='Powell'
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)


beta1_db = double_exp_mle(tubulin_12microM)
beta1_mle, db_mle = beta1_db

double_exp_samples = np.array([rg.exponential(1/beta1_mle, size = len(tubulin_12microM)) + rg.exponential(1/(beta1_mle+db_mle), size = len(tubulin_12microM)) for _ in range(100000)])


def show_qq_dp_12():
    p = bebi103.viz.qqplot(
        data=tubulin_12microM,
        samples=double_exp_samples,
        x_axis_label="time to catastrophe (s)",
        y_axis_label="time to catastrophe (s)",
        title="QQ plot for double Poisson model of 12 uM tubulin concentration"
    )

    bokeh.io.show(p)


def show_pe_gammar_12():
    p = bebi103.viz.predictive_ecdf(
        samples=gamma_samples, data=tubulin_12microM, discrete=True, x_axis_label="time to catastrophe (s)",
        title = "Predictive ECDFs for gamma distribution model of 12 uM tubulin concentration"
    )

    bokeh.io.show(p)


def show_pe_dp_12():
    p = bebi103.viz.predictive_ecdf(
        samples=double_exp_samples, data=tubulin_12microM, discrete=True, x_axis_label="time to catastrophe (s)",
        title = "Predictive ECDFs for double Poisson distribution model of 12 uM tubulin concentration"
    )

    bokeh.io.show(p)


def draw_parametric_bs_reps(data, size=1):
    """Parametric bootstrap replicates of parameters of
    Gamma distribution."""
    bs_reps_alpha = np.empty(size)
    bs_reps_beta = np.empty(size)
    alpha, beta = get_gamma_paras(data)

    for i in range(size):
        bs_sample = rg.gamma(alpha, 1/beta,size=len(data))
        bs_reps_alpha[i], bs_reps_beta[i] = get_gamma_paras(bs_sample)

    return bs_reps_alpha, bs_reps_beta


# +
concentration_list = [7, 9, 10, 12, 14]
bs_reps_list = []
conf_int_list = []

for concentration in concentration_list:

    time_vals = df_tidy.loc[df_tidy["tubulin concentration"] == concentration, "time to catastrophe (s)"]
    
    bs_reps = draw_parametric_bs_reps(
    time_vals, size=10000
    )

    
    bs_reps_list.append(np.array(bs_reps).transpose())
    
    conf_int_list.append(np.percentile(bs_reps, [2.5, 97.5], axis=1))


# -

def generate_colors(length):
    colors = []

    for i in range(length):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    return colors


def plot_conf_regions_alpha_beta_plane(bs_reps_list):
    
    c_list = ["7 uM", "9 uM", "10 uM", "12 uM", "14 uM"]
    colors = generate_colors(len(bs_reps_list))
    
    reps = { i : bs_reps_list[i] for i in range(0, len(bs_reps_list) ) }

    p = bokeh.plotting.figure(
        x_axis_label = "α",
        y_axis_label = "β",
        frame_height = 400,
        frame_width = 700,
    )

    for trial, bs_reps in reps.items():
        #Extract contour lines in D-k_off plane.

        x_line, y_line = bebi103.viz.contour_lines_from_samples(
            x = bs_reps[:, -2], y=bs_reps[:, -1], levels = [0.95]
        )

        #Plot the contour lines with fill
        for x, y in zip(x_line, y_line):
            p.line(x, y, line_width=2, color=colors[trial], legend_label=f'concentration {c_list[trial]}')
            p.patch(x, y, fill_color = colors[trial], alpha=0.3)

    p.legend.location = 'bottom_right'

    bokeh.io.show(p)


def show_alpha_beta_region():
    plot_conf_regions_alpha_beta_plane(bs_reps_list)




