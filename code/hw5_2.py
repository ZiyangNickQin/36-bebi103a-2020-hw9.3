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
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import iqplot
import pandas as pd
import bokeh.io
bokeh.io.output_notebook()

import scipy.stats
import math


# -

def get_total_time_at_beta(beta1, beta2):
    rg = np.random.default_rng(seed=3252)
    
    # since the time between each Poisson process is exponentially distributed we first generate two random datasets
    event1 = rg.exponential(1/beta1, size =150)
    event2 = rg.exponential(1/beta2, size =150)
    
    di= {'Process 1':event1,'Process 2':event2}
    df = pd.DataFrame(di)
    sum_column = df["Process 1"] + df["Process 2"]
    df["Total time"] = sum_column
    df["beta2/beta1"] = beta2/beta1
    return df
def get_total_time_at_beta_2000(beta1, beta2):
    rg = np.random.default_rng(seed=3252)
    event1 = rg.exponential(1/beta1, size =2000)
    event2 = rg.exponential(1/beta2, size =2000)
    di= {'Process 1':event1,'Process 2':event2}
    df = pd.DataFrame(di)
    sum_column = df["Process 1"] + df["Process 2"]
    df["Total time"] = sum_column
    df["beta2/beta1"] = beta2/beta1
    return df


# +
def plot_double_exp_simulation():
    beta1 = 10
    beta2 = 15
    p = iqplot.ecdf(
    data = get_total_time_at_beta(beta1, beta2),
    q="Total time",
    )
# If fit the function, it should be:
    x_theor = np.linspace(0, 0.7, 400)
    y_theor = beta1*beta2/(beta2-beta1)*(1/beta1*(1-np.exp(beta1*x_theor*(-1))) - 1/beta2*(1-np.exp(beta2*x_theor*(-1))))
    p.line(
        x=x_theor, y=y_theor, line_width=2, line_color="orange",
    )

# plot
    p.legend.location = 'top_left'
    bokeh.io.show(p)


# +
def plot_double_exp_simulation_2000():
# Stimilation:
    beta3 = 10
    beta4 = 15
    p = iqplot.ecdf(
        data = get_total_time_at_beta_2000(beta3, beta4),
        q="Total time",
    )

# If fit the function, it should be:
    x_theor = np.linspace(0, 0.7, 400)
    y_theor = beta3*beta4/(beta4-beta3)*(1/beta3*(1-np.exp(beta3*x_theor*(-1))) - 1/beta4*(1-np.exp(beta4*x_theor*(-1))))
    p.line(
        x=x_theor, y=y_theor, line_width=2, line_color="orange",
    )

#plot
    p.legend.location = 'top_left'
    bokeh.io.show(p)
# -


