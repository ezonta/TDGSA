#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import chaospy as cp
import multiprocessing
from joblib import Parallel, delayed
from typing import Callable, Optional, Union, Mapping
from numpy.typing import ArrayLike
from tqdm.autonotebook import tqdm

class distribution:
    """docstring
    example of dist_dict:
    dist_dict = {
        "param1": ("normal", [0, 1]),
        "param2": ("uniform", [0, 1]),
        "param3": ("lognormal", [0, 1]),
        "param4": ("loguniform", [0, 1]),
    }
    """
    dist: cp.J
    dim: int
    param_names: list[str]
    param_ranges: list[list[float]]
    num_samples: int
    
    def __init__(self, dist_dict: Mapping[str, tuple[str, list[float]]]) -> None:
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
                    f"Unknown distribution type: {dist_type}. Please choose from 'normal', 'uniform', 'lognormal', or 'loguniform'.\n"
                )

        self.dist = cp.J(*dist_list)
        self.dim = len(dist_list)
        self.param_names = param_names
        self.param_ranges = param_ranges

    def sample(self, num_samples: int, rule: str) -> ArrayLike:
        """docstring
        """
        samples = np.array(self.dist.sample(num_samples, rule=rule)).T
        return samples


class simulator:
    """docstring
    """
    model: Callable[[ArrayLike], ArrayLike]
    dist: distribution
    time: ArrayLike
    data: Optional[ArrayLike]
    num_samples: Optional[int]
    params: Optional[pd.DataFrame]
    output: Optional[pd.DataFrame]

    def __init__(self, model: Callable[[ArrayLike], ArrayLike], timesteps_solver: ArrayLike, data: Optional[ArrayLike]=None) -> None:
        self.model = model
        self.time = timesteps_solver
        
        #TODO: move all of this to TDGSA class
        self.num_samples = 0
        self.data = data  
        self.params = None  
        self.output = None
        
    def run(self, params) -> ArrayLike:
        """docstring
        """
        num_cores = multiprocessing.cpu_count()
        outputs = Parallel(n_jobs=num_cores)(
            delayed(self.model)(param) for param in tqdm(params)
        )
        outputs = np.array(outputs).reshape(params.shape[0], -1)
        return outputs
    
    # TODO: move to TDGSA class
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
