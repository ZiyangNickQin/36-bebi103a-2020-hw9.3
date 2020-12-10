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

# ### HW 8.2

# +
data_path = r"C:\Users\qzy91\Desktop\BEBi103a"
# ------------------------------

import warnings
import os, sys, subprocess
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats as st
import tqdm
import numba
import bebi103
import bokeh.io
import bokeh.plotting
bokeh.io.output_notebook()
import iqplot
import holoviews as hv
hv.extension('bokeh')

bebi103.hv.set_defaults()
# -

# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; We first open the file. And according to the question, we only need to analyze the labeled MT molecules, so we extract it from the file as a new dataframe:

df = pd.read_csv(os.path.join(data_path, "gardner_time_to_catastrophe_dic_tidy.csv"), comment="#")
dfl = df.loc[df["labeled"]]
dfl

# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; We then draw the ECDF of the 'catastrophe time (s)' to review the general information about our dataset.

# +
p = iqplot.ecdf(
    data=dfl,
    q='time to catastrophe (s)',
    x_axis_label="time to catastrophe (s)",
    title='ECDF of labeled ones',
    frame_height=250,
    frame_width=300,
)

bokeh.io.show(p)
# -

# #### 8.2 a)

# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Because in gamma distribution:
# \begin{align}
# f(y;\alpha, \beta) = \frac{1}{\Gamma(\alpha)}\,\frac{(\beta y)^\alpha}{y} \,\mathrm{e}^{-\beta y}\\
# \end{align}
#
# \begin{align}
# mean &= \frac{\alpha}{\beta} \\
# varianc&e = \frac{\alpha}{\beta^2}
# \end{align}

# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; We' have:

# \begin{align}
# \alpha &= \frac{mean^2}{variance}\\
# \beta &= \frac{mean}{variance}\\
# \end{align}

# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; So, we can calculate these 2 parameters directly and $\textbf{analytically}$ from the dataset instead of numerically. We first define the functions for computing analytical MLE and performing parametric bootstrap to get confidence intervals:

# +
rg = np.random.default_rng()

def get_gamma_paras(ar):
    m = np.mean(ar)
    v = np.std(ar)**2
    alpha_mle = m**2/v
    beta_mle = m/v
    return (alpha_mle,beta_mle)

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


# -

# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; And finally, let's calculate the parameters and their confidence intervals:

# +
test = dfl["time to catastrophe (s)"]

bs_reps_alpha, bs_reps_beta = draw_parametric_bs_reps(
    test, size=10000
)

print('MLE values for α and β:', get_gamma_paras(test))
print('α conf int:', np.percentile(bs_reps_alpha, [2.5, 97.5]))
print('β conf int:', np.percentile(bs_reps_beta, [2.5, 97.5]))
# -

# In this case, our calculated mean value will be:
# \begin{align}
# \mu = \frac{\alpha}{\beta} = \frac{2.2239376038081664}{0.005046250504393194} = 440.711 (s)
# \end{align}

# #### 8.2 b)

# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; We first need to extract the catastrophe's time of the MT molecules as a new array for future optimization:

t_vals = dfl["time to catastrophe (s)"].values


# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Then, unlike the gamma distribution, who already has an existing function for PDF, we need to define the PDF function for the "double exponential distribution". We've finished the analytical part in hw5.2, and the deriving process is as following:

# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; First of all, let's get down to how we reach the PDF of two exponentially distributed Poisson processes. For two sequential Poisson process, it's quicker to use convolution to calculate the overall PDF:
#
# We know that total time of two Poisson process, $t = T_1 + T_2$, so $T_2 = t - T_1$, in this case, $T_1 >0$, $T_2 > 0$ and $t = T_1 + T_2 > 0$
#
# \begin{align}
#      f(t; \beta_1, \beta_2) = \int_{-\infty}^{t} f_{\beta_1}(T_1) \ f_{\beta_2}(t - T_1) \ dT_1
# \end{align}

# And so each Poisson process, it obeys exponential distribution which gives,
#
# \begin{align}
#      f_{\beta_1}(t) = \beta_1 \cdot e^{-\beta_1 t} \\
#      f_{\beta_2}(t) = \beta_2 \cdot e^{-\beta_2 t}
# \end{align}
#
# so take $f_{\beta_1}(t)$ and $f_{\beta_2}(t)$ into our PDF then we get:
#
# \begin{align}
#      PDF(\beta_1, \beta_2) &= \int_{-\infty}^{t} f_{\beta_1}(T_1) \ f_{\beta_2}(t - T_1) \ dT_1 \\
#      &=\int_{-\infty}^{t} \beta_1 \beta_2 \cdot e^{-\beta_1 T_1} e^{-\beta_2 (t - T_1)} \ dT_1
# \end{align}

# And since the exponentials, $f_{\beta}(t)$, have density $0$ to the left of $0$. So we look only at the case $t \ge 0$ and in our case, the actual expression is:
#
# \begin{align}
#      PDF(\beta_1, \beta_2) &= \int_{0}^{t} f_{\beta_1}(T_1) \ f_{\beta_2}(t - T_1) \ dT_1 \\
#      &=\int_{0}^{t} \beta_1 \beta_2 \cdot e^{-\beta_1 T_1} e^{-\beta_2 (t - T_1)} \ dT_1 \\
#      &=\beta_1 \beta_2 \cdot e^{-\beta_2 t} \int_{0}^{t} e^{-(\beta_1 - \beta_2)T_1} \ dT_1 \\
#      &=\frac{\beta_1\beta_2}{\beta_1-\beta_2} \cdot e^{-\beta_2 t} \cdot (-e^{-(\beta_1 - \beta_2)t})|^t_0 \\
#      &=\frac{\beta_1\beta_2}{\beta_1-\beta_2} \cdot (e^{-\beta_2 t} - e^{-\beta_1 t}) = f(t; \beta_1, \beta_2)
# \end{align}
#

# Next, we stipulate that $\beta_1 \leq \beta_2$ and $\Delta \beta = \beta_2 - \beta_1 \geq 0$. Assuming that all the observations are independent and identically distributed, we have the Likelihood function:
#
# \begin{align}
# L(\beta_1, \Delta\beta; t) = \prod_{i} \frac{\beta_1(\beta_1 + \Delta\beta)}{\Delta\beta}e^{-\beta_1 t_i} (1 - e^{-\Delta\beta t_i}),
# \end{align}

# when we take the logarithm of Likelihood function above, we will get:
#
#
# \begin{align}
# \ell(\beta_1^*, \Delta\beta^*; t) = n ln \beta_1 - n ln \Delta\beta + n ln (\beta_1 + \Delta\beta) - \beta_1 \sum_{i} t_i + \sum_{i} ln \left(1 - e^{-\Delta\beta t_i} \right)
# \end{align}
#
#
# Now, we define the function **log_like_poisson_exp(params, t)** that computes the log likelihood using the above mathematical model.

def log_like_poisson_exp(params, t):
    
    beta1, db = params
    
    if beta1 < 0 or db < 0:
        return -np.inf
    
    log_like = len(t)*np.log(beta1) + len(t)*np.log(beta1 + db) - len(t)*np.log(db) - beta1*np.sum(t) + np.sum(np.log(1 - np.exp(-db*t)))
    
    return log_like
    


# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Now we define the function **double_exp_mle(ell)** to compute MLE for parameters in double exponential model:

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


beta1_db = double_exp_mle(t_vals)
beta1_mle, db_mle = beta1_db
print('MLE values for β1 and Δβ: ', beta1_db)


# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; We draw bootstrap replicates to compute the confidence interval:

def draw_parametric_bs_reps(data, beta1, db, size=1):
    """Parametric bootstrap replicates of parameters of
    Student-t distribution."""
    bs_reps_beta1 = np.empty(size)
    bs_reps_db = np.empty(size)

    for i in range(size):
        bs_sample = rg.exponential(1/beta1, size = len(data)) + rg.exponential(1/(beta1+db), size = len(data))
        bs_reps_beta1[i] = double_exp_mle(bs_sample)[0]
        bs_reps_db[i] = double_exp_mle(bs_sample)[1]

    return bs_reps_beta1, bs_reps_db


# +
bs_reps_beta1, bs_reps_db = draw_parametric_bs_reps(
    t_vals, beta1_mle, db_mle, size=1000
)

print('β1 conf int of the distribution:', np.percentile(bs_reps_beta1, [2.5, 97.5]))
print('Δβ conf int of the distribution:', np.percentile(bs_reps_db, [2.5, 97.5]))
# -

# $\textbf{Further discussion}$: theoretically, the expectation of the 'mixed' exponential distribution could be derived through the following calculation:
#
# \begin{align}
# E(t) &= \int_{-\infty}^{+\infty} |t| \ f(t) dt \\
# &=\int_{0}^{+\infty} t \ \frac{\beta_1 \beta_2}{\beta_2 - \beta_1} \left(e^{-\beta_1 t}- e^{-\beta_2 t} \right) dt \\
# &=\frac{\beta_1 \beta_2}{\beta_2 - \beta_1} \left(\int_{0}^{+\infty} te^{-\beta_1 t} dt - \int_{0}^{+\infty} te^{-\beta_2 t} dt \right) \\
# &=\frac{\beta_1 \beta_2}{\beta_2 - \beta_1} \left(\frac{1}{\beta_1^2} - \frac{1}{\beta_2^2} \right) \\
# &=\frac{\beta_1 + \beta_2}{\beta_1 \beta_2}
# \end{align}

# In order to test the precision of our MLE method, we then try to calculate the mean value of the catastrophe time in the dataframe. In our case, we use the calculated MLE of $\beta_1 = 0.0041127$ and $\beta_2 = 0.0041127+0.00090066 =0.005013$ to get the mean value.
#
# \begin{align}
# \mu &= \frac{\beta1 + \beta_2}{\beta_1\beta_2} \\
# &=\frac{0.0041127 + 0.005013}{0.0041127 × 0.005013} \\
# &=442.63 (s)
# \end{align}

mean = np.mean(t_vals)
print('Real mean value of catastrophe time from our dataset:', mean, '(s)')

# Comparing the calculated mean $(442.63 \ s)$ and the real mean $(440.711 \ s)$, we could say that our strategy is really precise and the 'biexponential' distribution could be a good model for catastrophe time.

# ### Computing Environment

# %load_ext watermark
# %watermark -v -p numpy,scipy,pandas,bokeh,holoviews,jupyterlab

# ### Attributions
# **HW8.2 a)** Ruilin wrote the solution. Ziyang helped with polishing and annotations.
#
# **HW8.2 b)** We attempted to solve it together during the lab session. Then, we continued working on it together iteratively by writing the functions, improving code and adding mathematical formulas, derivations and annotations.
