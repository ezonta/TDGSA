#!/usr/bin/env python3

from TDGSA import simulator, distribution, time_dependent_sensitivity_analysis
import numpy as np
import pandas as pd
import pytest

def test_time_dependent_sensitivity_analysis():
    num_timesteps_solver = 100
    num_timesteps_quadrature = 100
    num_samples = 10
    
    def model(params):
        return np.array([params[0] * params[1] * params[2] * params[3]] * num_timesteps_solver)
    dist_dict = {
        "param1": ("normal", [0, 1]),
        "param2": ("uniform", [0, 1]),
        "param3": ("lognormal", [0, 1]),
        "param4": ("loguniform", [0, 1]),
    }
    timesteps = np.linspace(0, 10, num_timesteps_solver)
    sim = simulator(model, timesteps)
    dist = distribution(dist_dict)
    tdsa = time_dependent_sensitivity_analysis(sim, dist)
    assert tdsa.simulator == sim
    assert tdsa.distribution == dist
    assert tdsa.timesteps_solver.shape == (num_timesteps_solver,)

    # Test sample_params_and_run_simulator
    tdsa.sample_params_and_run_simulator(num_samples)
    assert tdsa.params.shape == (num_samples, 4)
    assert tdsa.outputs.shape == (num_samples, num_timesteps_solver)
    assert tdsa.num_samples == num_samples

    # Test instantiation with data
    data = (tdsa.params, tdsa.outputs)
    tdsa_with_data = time_dependent_sensitivity_analysis(sim, dist, data=data)
    assert tdsa_with_data.params.equals(tdsa.params)
    assert tdsa_with_data.outputs.equals(tdsa.outputs)

    # Test compute_sobol_indices with PCE method
    tdsa.compute_sobol_indices(method="PCE", PCE_order=3, num_timesteps_quadrature=num_timesteps_quadrature)
    assert "PCE" in tdsa.sobol_indices

    # Test compute_sobol_indices with KL method
    tdsa.compute_sobol_indices(method="KL", KL_truncation_level=5, num_timesteps_quadrature=num_timesteps_quadrature)
    assert "KL" in tdsa.sobol_indices

    # Test evaluate_surrogate_model
    sample_param = np.array([0.5, 0.5, 0.5, 0.5])
    surrogate_output = tdsa.evaluate_surrogate_model(sample_param, method="PCE")
    assert surrogate_output.shape == (num_timesteps_quadrature,)
    
    # Test ValueError for invalid sampling method
    with pytest.raises(ValueError):
        tdsa.sample_params_and_run_simulator(num_samples, sampling_method="invalid_method")

    # Test ValueError for compute_sobol_indices without outputs
    tdsa_no_data = time_dependent_sensitivity_analysis(sim, dist, num_timesteps_quadrature=num_timesteps_quadrature)
    with pytest.raises(ValueError):
        tdsa_no_data.compute_sobol_indices(method="PCE")

    # Test ValueError for invalid method in compute_sobol_indices
    with pytest.raises(ValueError):
        tdsa.compute_sobol_indices(method="invalid_method")

    # Test ValueError for compute_second_order_sobol_indices without PCE coefficients and quadrature timesteps
    with pytest.raises(ValueError):
        tdsa_no_data.compute_second_order_sobol_indices(method="PCE")

    # Test ValueError for compute_third_order_sobol_indices without PCE coefficients and quadrature timesteps
    with pytest.raises(ValueError):
        tdsa_no_data.compute_third_order_sobol_indices(method="PCE")

        