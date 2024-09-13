#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import chaospy as cp
import multiprocessing
from joblib import Parallel, delayed
from typing import Callable, Optional, Union, Dict
from numpy.typing import NDArray
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
    dist_dict: Dict[str, tuple[str, list[float]]]

    def __init__(self, dist_dict: Dict[str, tuple[str, list[float]]]) -> None:
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
        self.dist_dict = dist_dict

    def sample(self, num_samples: int, rule: str = "random") -> NDArray:
        """docstring"""
        samples = np.array(self.dist.sample(num_samples, rule=rule)).T
        return samples


class simulator:
    """docstring"""

    model: Callable[[NDArray], NDArray]
    time: NDArray

    def __init__(
        self,
        model: Callable[[NDArray], NDArray],
        timesteps_solver: NDArray
    ) -> None:
        self.model = model
        self.time = timesteps_solver

    def run(self, params: NDArray) -> NDArray:
        """docstring"""
        num_cores = multiprocessing.cpu_count()
        outputs = Parallel(n_jobs=num_cores)(
            delayed(self.model)(param) for param in tqdm(params)
        )
        outputs = np.array(outputs).reshape(params.shape[0], -1)
        return outputs
