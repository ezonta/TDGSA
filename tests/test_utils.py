#!/usr/bin/env python3

from TDGSA import simulator, distribution
import numpy as np


def test_distribution():
    dist_dict = {
        "param1": ("normal", [0, 1]),
        "param2": ("uniform", [0, 1]),
        "param3": ("lognormal", [0, 1]),
        "param4": ("loguniform", [0, 1]),
    }
    dist = distribution(dist_dict)
    assert dist.dim == 4
    assert dist.param_names == ["param1", "param2", "param3", "param4"]
    assert dist.param_ranges == [[0, 1], [0, 1], [0, 1], [0, 1]]
    assert dist.dist_dict == dist_dict
    assert dist.dist.sample(10).shape == (4, 10)
    assert dist.dist.sample(10)[1, :].min() >= 0
    assert dist.dist.sample(10)[1, :].max() <= 1


def test_simulator():
    def model(params):
        return np.array(params)

    dist_dict = {
        "param1": ("normal", [0, 1]),
        "param2": ("uniform", [0, 1]),
        "param3": ("lognormal", [0, 1]),
        "param4": ("loguniform", [0, 1]),
    }
    timesteps = np.linspace(0, 10, 100)
    sim = simulator(model, timesteps)
    dist = distribution(dist_dict)
    output = sim.run(dist.sample(10))
    assert sim.model([1, 2, 3, 4]).shape == (4,)
    assert sim.model == model
    assert sim.time.shape == (100,)
    assert output.shape == (10, 4)
