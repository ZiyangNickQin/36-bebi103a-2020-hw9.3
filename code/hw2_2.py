# setup
import os, sys, subprocess
data_path = r"c://users/qrlin/hw9"
import numpy as np
import scipy.special
import pandas as pd
import bokeh.io
import bokeh.plotting
bokeh.io.output_notebook()
import holoviews as hv
import bebi103
import matplotlib.pyplot as plt
# %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np


def ecdf_vals(x_array):
    
    """This function takes a one-dimensional Numpy array (or Pandas Series) of data 
    and returns the x and y values for plotting the ECDF in the “dots” style,"""
    
    #if x_array is Panda Series, we need to convert it to Numpy array
    if isinstance(x_array, pd.DataFrame):
        x_array = x_array.to_numpy()
         
    y_array = np.zeros_like(x_array)
    for i, element in enumerate(x_array):
        y_array[i] = (x_array <= element).sum()
    y_array = y_array/len(y_array)
    return x_array, y_array

fname = os.path.join(data_path, "gardner_time_to_catastrophe_dic_tidy.csv")
df = pd.read_csv(fname)

df["Fluorescent group"] = df["labeled"].apply(lambda x: "Fluorescently labeled" if x else "Control")

df_labeled = df.loc[df["Fluorescent group"] == "Fluorescently labeled"][["time to catastrophe (s)"]]
df_control = df.loc[df["Fluorescent group"] == "Control"][["time to catastrophe (s)"]]

(ecdf_x_labeled, ecdf_y_labeled) = ecdf_vals(df_labeled)
(ecdf_x_control, ecdf_y_control) = ecdf_vals(df_control)

#Plot ECDFs of fluorescently labeled and control microtubules together
def plot_ecdf():
    labeled_ecdf = plt.scatter(ecdf_x_labeled, ecdf_y_labeled, s=9, color="red")
    control_ecdf = plt.scatter(ecdf_x_control, ecdf_y_control, s=9, color="blue")
    plt.title('ECDF of time to catastrophe (s) for fluorescently labeled and control cells')
    plt.xlabel('Time to catastrophe (s)', fontsize=17)
    plt.ylabel('ECDF', fontsize=17)
    plt.legend((labeled_ecdf, control_ecdf),('Labeled', 'Control'),numpoints=1, loc='right', fontsize=14)
    plt.rcParams["figure.figsize"] = (6,6)
    plt.show()