import arviz as az
import numpy as np
import plotnine as gg
import pandas as pd
from polars import DataFrame, col, Int64, Float64, Series, when, from_pandas
from cmdstanpy import CmdStanModel, install_cmdstan, cmdstan_path, set_cmdstan_path

SEED = 123

az.style.use("arviz-darkgrid")

def simulate_ar_data(intercept, coef1, coef2, noise=0.3, *, warmup=10, steps=200):
    # We sample some extra warmup steps, to let the AR process stabilize
    draws = np.zeros(warmup + steps)
    # Initialize first draws at intercept
    draws[:2] = intercept
    for step in range(2, warmup + steps):
        draws[step] = (
            intercept
            + coef1 * draws[step - 1]
            + coef2 * draws[step - 2]
            + np.random.normal(0, noise)
        )
    # Discard the warmup draws
    return draws[warmup:]

# True parameters of the AR process
ar1_data = simulate_ar_data(10, -0.9, 0)

ar1_df = pd.DataFrame({'index': range(len(ar1_data)), 'value': ar1_data})

# Plot using plotnine
plot_simulated = (gg.ggplot(ar1_df, gg.aes(x='index', y='value')) +
        gg.geom_line() +
        gg.ggtitle('Generated Autoregressive Timeseries') +
        gg.theme(plot_title=gg.element_text(size=15, ha='center')) +
        gg.theme(figure_size=(10, 3))
       )

plot_simulated.show()


# Define priors
priors = {
    "coefs": {"mu": [10, 0.2], "sigma": [0.1, 0.1], "size": 2},
    "sigma": 8,
    "init": {"mu": 9, "sigma": 0.1, "size": 1},
}

# Compile model
model = CmdStanModel(stan_file='models/ar1.stan')

# Data for the model
data = {
    'N': len(ar1_data),
    'y': ar1_data,
    'coefs_mu': priors['coefs']['mu'],
    'coefs_sigma': priors['coefs']['sigma'],
    'sigma': priors['sigma'],
    'init_mu': priors['init']['mu'],
    'init_sigma': priors['init']['sigma']
}

# Sample from the model
fit = model.sample(data=data, seed=100, iter_sampling=2000, iter_warmup=1000)

# Convert the CmdStanMCMC object to an ArviZ InferenceData object
idata_ar = az.from_cmdstanpy(posterior=fit)

# Plotting the results using plotnine
ar1_df = pd.DataFrame({'index': range(len(ar1_data)), 'value': ar1_data})
plot_simulated_model = (gg.ggplot(ar1_df, gg.aes(x='index', y='value')) +
        gg.geom_line() +
        gg.ggtitle('Generated Autoregressive Timeseries') +
        gg.theme(plot_title=gg.element_text(size=15, ha='center')) +
        gg.theme(figure_size=(10, 3))
       )

plot_simulated_model.show()

print(idata_ar)