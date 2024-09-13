#!/usr/bin/env python3

from TDGSA import simulator, distribution, time_dependent_sensitivity_analysis
import numpy as np
from scipy.integrate import odeint
import pytest
import matplotlib.pyplot as plt

def test_oscillator():

    NUM_TIMESTEPS_SOLVER = 500
    num_timesteps_quadrature = 100
    num_samples = 1000

    def my_model_numeric(param):
        def derivatives(x, t):
            return [x[1],-2*param[0]*x[1]-(param[0]**2 + param[1]**2)*x[0]]
        time = np.linspace(0,10,NUM_TIMESTEPS_SOLVER)
        displacement, velocity = odeint(derivatives, [param[2],0], time).T
        return displacement

    my_dist_dict = {
        r"$\alpha$": ("uniform", [3/8, 5/8]),
        r"$\beta$": ("uniform", [10/4, 15/4]),
        r"$\ell$": ("uniform", [-5/4, -3/4])
    }

    my_distribution = distribution(my_dist_dict)
    my_simulator = simulator(my_model_numeric, timesteps_solver=np.linspace(0,10,NUM_TIMESTEPS_SOLVER))
    my_tdsa = time_dependent_sensitivity_analysis(my_simulator, my_distribution, num_timesteps_quadrature=num_timesteps_quadrature)

    my_params, my_output = my_tdsa.sample_params_and_run_simulator(num_samples)

    sobol_indices_KL = my_tdsa.compute_sobol_indices("KL")

    sobol_indices_PCE = my_tdsa.compute_sobol_indices("PCE")

    second_order, params = my_tdsa.compute_second_order_sobol_indices("PCE")
    third_order, params = my_tdsa.compute_second_order_sobol_indices("KL")
    third_order, params = my_tdsa.compute_third_order_sobol_indices("PCE")
    second_order, params = my_tdsa.compute_third_order_sobol_indices("KL")
    
    relative_error = np.abs(sobol_indices_KL - sobol_indices_PCE) / np.abs(sobol_indices_PCE)
    assert np.all(relative_error <= 0.20), "Relative error between KL and PCE Sobol' indices is higher than 20%"
    assert np.all(sobol_indices_KL >= 0), "Negative Sobol' indices computed with KL method"
    assert np.all(sobol_indices_PCE >= 0), "Negative Sobol' indices computed with PCE method"
    assert sobol_indices_KL["total"][1] > sobol_indices_KL["total"][2] and sobol_indices_KL["total"][2] > sobol_indices_KL["total"][0], "Total-order Sobol' indices computed with KL method are not correctly ordered"
    assert sobol_indices_PCE["total"][1] > sobol_indices_PCE["total"][2] and sobol_indices_PCE["total"][2] > sobol_indices_PCE["total"][0], "Total-order Sobol' indices computed with PCE method are not correctly ordered"

