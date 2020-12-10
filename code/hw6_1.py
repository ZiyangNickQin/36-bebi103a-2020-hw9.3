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
data_path = "../data/"
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
df.head()


def show_ecdfs_with_interval():
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

    bokeh.io.show(p)
    



