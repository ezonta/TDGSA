#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import chaospy as cp
import multiprocessing
from joblib import Parallel, delayed


class distribution:
    """A class that acts as a wrapper for a chaospy distribution"""

    def __init__(self, dist_dict):
        """Constructor method

        Args:
            dist_dict (dict): Dict that defines the distribution of each parameter (e.g. {'alpha':['uniform', [0, 1]], 'beta':['normal': [0, 1]]}),
            where the numbers define upper and lower bounds for the uniform distribution and mean and standard deviation for the normal distribution.
        """
        dist_list = []
        param_names = []
        param_ranges = []

        for key, value in dist_dict.items():
            dist_type = value[0]
            dist_params = value[1]
            param_names.append(key)
            param_ranges.append(dist_params)
            if dist_type == "normal":
                dist_list.append(cp.Normal(*dist_params))
            elif dist_type == "uniform":
                dist_list.append(cp.Uniform(*dist_params))
            elif dist_type == "loguniform":
                dist_list.append(cp.LogUniform(*dist_params))
            elif dist_type == "lognormal":
                dist_list.append(cp.LogNormal(*dist_params))
            else:
                raise ValueError(
                    f"Unknown distribution type: {dist_type}. Please choose from normal, uniform, lognormal, or loguniform.\n"
                )

        self.dist = cp.J(*dist_list)
        self.dim = len(dist_list)
        self.param_names = param_names
        self.param_ranges = param_ranges
        self.num_samples = None

    def sample(self, num_samples=1):
        """A method that samples from the distribution"""
        samples = np.array(self.dist.sample(num_samples)).T
        return samples


class simulator:
    """A class that acts as a wrapper for the model of interest.

    Args:
        model (callable): The model for sensitivity analysis, which should return the time series of interest.
        dist (distribution): The distribution from which parameters are sampled.
        data (array, optional): An array of parameters and time-dependent output, if pre-existing.
    """

    def __init__(self, model, dist, timesteps_solver, data=None, parallel=True):
        self.model = model
        self.dist = dist
        self.time = timesteps_solver
        self.parallel = parallel
        
        self.num_samples = None
        self.data = data  # params and output as np.array
        self.params = None  # params as pd.dataframe (columns are param names)
        self.output = None  # output as pd.dataframe (columns are time steps)

    def generate_params(self, num_samples=1):
        """A method that samples from the given distribution.

        Args:
            size (int, optional): The number of samples to generate. Defaults to 1.

        Returns:
            array: An array of parameters.
        """
        self.num_samples = num_samples
        params = self.dist.sample(num_samples)
        self.params = pd.DataFrame(params, columns=self.dist.param_names)
        return params

    def generate_output(self, params):
        """A method that calls the model with a set of parameters.

        Args:
            params (array): An array of parameters.

        Returns:
            output: An array of time-dependent output.
        """
        num_cores = multiprocessing.cpu_count()
        if self.parallel:
            output = Parallel(n_jobs=num_cores)(
                delayed(self.model)(param) for param in params
            )
        else:
            output = [self.model(param) for param in params]
        output = np.array(output).reshape(params.shape[0], -1)
        self.output = pd.DataFrame(output, columns=self.time)
        self.data = [params, output]
        return output

    def plot_output(self):
        """A method that plots the generated output of the simulator"""
        if self.output is None:
            raise ValueError("No output available. Please run the simulator first.\n")
        else:
            fig, ax = plt.subplots(2, 1, sharex=True)
            for i in range(self.num_samples):
                ax[0].plot(self.time, self.output.iloc[i])
            ax[0].set_ylabel("Output")

            ax[1].plot(self.time, self.output.mean())
            ax[1].fill_between(
                self.time,
                self.output.mean() - self.output.std(),
                self.output.mean() + self.output.std(),
                alpha=0.3,
            )
            ax[1].set_xlabel("Time")
            ax[1].set_ylabel("Output")
            plt.tight_layout()
            plt.show()
