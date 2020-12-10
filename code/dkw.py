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

# + endofcell="--"
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

# # +
import os, sys, subprocess
data_path = r"C:\Users\qrlin\hw9"
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
df["Fluorescent group"] = df["labeled"].apply(lambda x: "Fluorescently labeled" if x else "Control")


sample_ctrl = df.loc[df["Fluorescent group"] == "Control", "time to catastrophe (s)"].values
sample_label = df.loc[df["Fluorescent group"] == "Fluorescently labeled", "time to catastrophe (s)"].values

# # +
af = 0.05
n_ctrl = len(sample_ctrl)
eu_ctrl = (1/2/n_ctrl*(math.log10(2/af)))**0.5
n_label = len(sample_label)
eu_label = (1/2/n_label*(math.log10(2/af)))**0.5

# -

def ecdf(x_array, start_potion=0, end_potion=1):
    
    """This function takes a one-dimensional Numpy array (or Pandas Series) of data 
    and returns the x and y values for plotting the ECDF in the “dots” style,"""
    
    #if x_array is Panda Series, we need to convert it to Numpy array
    if isinstance(x_array, pd.DataFrame):
        x_array = x_array.to_numpy()
        
    y_array = np.zeros_like(x_array)
    for i, element in enumerate(x_array):
        y_array[i] = (x_array <= element).sum()
    y_array = y_array/len(y_array)
    
    
    
    dic={"data" : x_array, "ecdf" :y_array}
    df1=DataFrame(dic)
    df2=df1.loc[(df1["ecdf"] >= start_potion) & (df1["ecdf"] <= end_potion)]
    
    return df2

# # +
df_ecdf_label = ecdf(sample_label)
df_ecdf_ctrl = ecdf(sample_ctrl)

# extract the data from the dataframe
ecdf_label = df_ecdf_label["ecdf"].values
ecdf_ctrl = df_ecdf_ctrl["ecdf"].values

# # +

# we compute the upper limit and lower limit of two kinds of samples respectively 
down_ctrl=[]
for i in range(len(sample_ctrl)):
    down_ctrl.append(max(0,ecdf_ctrl[i]-eu_ctrl))
down_ctrl = np.array(down_ctrl)

down_label=[]
for i in range(len(sample_label)):
    down_label.append(max(0,ecdf_label[i]-eu_label))
down_label = np.array(down_label)

up_ctrl=[]
for i in range(len(sample_ctrl)):
    up_ctrl.append(min(1,ecdf_ctrl[i]+eu_ctrl))
up_ctrl = np.array(up_ctrl)

up_label=[]
for i in range(len(sample_label)):
    up_label.append(min(1,ecdf_label[i]+eu_label))
up_label = np.array(up_label)
# --

# +
Type = ["down ctrl"] * len(down_ctrl) + ["down label"] * len(down_label) +["up ctrl"] * len(up_ctrl) +["up label"] * len(up_label)
x=np.concatenate((sample_ctrl, sample_label, sample_ctrl, sample_label))
y=np.concatenate((down_ctrl, down_label, up_ctrl, up_label))
dic={"Type" : Type,
    "x" : x,
    "y" :y}
df1=DataFrame(dic)


hv.extension("bokeh")
hv_fig = hv.Points(
    data=df1,
    kdims=["x", "y"],
    vdims=["Type"],
).groupby(
    'Type'
).opts(
    width=500,
    legend_position = "bottom_right"
).overlay()

# -

def show_dkw():
    p = hv.render(hv_fig)
    bokeh.io.show(p)
