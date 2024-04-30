#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import chaospy as cp
import multiprocessing
from joblib import Parallel, delayed
from sklearn import linear_model
import utils
import xarray as xr
from typing import Callable, Optional, Union, Mapping
from numpy.typing import ArrayLike
from tqdm.autonotebook import tqdm

# TODO: Use typing for all methods
# TODO: Implement second and third order calculation for KL
# TODO: Implement different sampling (QMC, Quadrature)  in simulator and 
# automatically use quadrature or regression in PCE method based on chosen sampling
# TODO: Drop simulator.data in favor of simulator.output and simulator.params (and possibly drop all NaN rows)
# TODO: put "method" in compute_sobol_indices as an argument
# TODO: Implement pointwise-in-time GPR surrogate models
# TODO: Implement MC method

class time_dependent_sensitivity_analysis:
    """A class that performs time-dependent global sensitivity analysis."""
    # TODO: sort class members
    simulator: utils.simulator
    distribution: utils.distribution
    param_names: list[str]
    
    params: Optional[pd.DataFrame]
    outputs: Optional[pd.DataFrame]
    
    num_samples: Optional[int]
    
    PCE_quad_weights: Optional[ArrayLike]
    
    def __init__(self, simulator: utils.simulator, distribution: utils.distribution, **kwargs):

        self.simulator = simulator
        self.distribution = distribution
        
        self.params = None
        self.outputs = None

        self.param_names = self.distribution.param_names
        self.num_samples = self.simulator.num_samples
        self.num_params = self.distribution.dim
        self.timesteps_solver = self.simulator.time

        self.num_timesteps_quadrature = kwargs.get("num_timesteps_quadrature", 100)
        # quadrature or regression
        self.PCE_option = kwargs.get("PCE_option", "regression")
        # TODO: change these and get them from sample_params_and_run_simulator
        self.PCE_quad_weights = kwargs.get("PCE_quad_weights", None)
        # TODO: use as kwargs in compute_sobol_indices instead
        self.PCE_order = kwargs.get("PCE_order", 4)
        self.KL_truncation_level = kwargs.get("KL_truncation_level", 8)
        # TODO: remove together with MC method
        # random, halton, latin_hypercube, sobol, etc.
        self.MC_option = kwargs.get("MC_option", "random")

        self.sobol_indices = None
        self.td_sobol_indices = None
        
        self._coeff_pointwise = None
        self._polynomial_pointwise = None
        
    def sample_params_and_run_simulator(self, num_samples: int, sampling_method: str="random", **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        """A method that generates parameter samples and runs the simulator.
        --> List options for kwargs for quasirandom_method (Halton, Sobol, Latin Hypercube) and quadrature_method (Clenshaw-Curtis, etc.) """
        
        ## sampling of parameters
        print("Sampling parameters ...\n")
        
        if sampling_method == "random":
            samples = self.distribution.sample(num_samples=num_samples, rule="random")
        
        elif sampling_method == "quasirandom":
            quasirandom_rule = kwargs.get("quasirandom_method", "halton")
            if quasirandom_rule != "halton" and quasirandom_rule != "sobol" and quasirandom_rule != "latin_hypercube":
                raise ValueError(
                    f"Unknown quasirandom method: {quasirandom_rule}. Please choose from 'halton', 'sobol', or 'latin_hypercube'.\n"
                )
            samples = self.distribution.sample(num_samples=num_samples, rule=quasirandom_rule)
        
        elif sampling_method == "quadrature":
            quadrature_rule = kwargs.get("quadrature_method", "clenshaw_curtis")
            quadrature_order = kwargs.get("quadrature_order", 4)
            if quadrature_rule != "clenshaw_curtis" and quadrature_rule != "gaussian" and quadrature_rule != "legendre":
                raise ValueError(
                    f"Unknown quadrature method: {quadrature_rule}. Please choose from 'clenshaw_curtis', 'gaussian', or 'legendre'.\n"
                )
            samples, weights = cp.generate_quadrature(quadrature_order, self.distribution.dist, rule=quadrature_rule)
            
        else:
            raise ValueError(
                f"Unknown sampling method: {sampling_method}. Please choose from 'random', 'quasirandom', or 'quadrature'.\n"
            )
        
        ## run simulator and get outputs
        print("Running simulator ...\n")
        
        outputs = self.simulator.run(samples)
        
        params = pd.DataFrame(samples, columns=self.param_names)
        outputs = pd.DataFrame(outputs, columns=self.timesteps_solver)
        
        self.params = params
        self.outputs = outputs
        
        return params, outputs

    def compute_sobol_indices(self, method):
        """A method that runs the time-dependent sensitivity analysis and returns the generalized sobol indices."""
        if self.simulator.data is None:
            raise ValueError("No data available. Please run the simulator first.\n")
        else:
            params = self.simulator.data[0]
            output = self.simulator.data[1]

        if method == "KL":
            self._KL_analysis(params, output)
        elif method == "PCE":
            self._PCE_analysis(params, output)
        elif method == "MC":
            self._MC_analysis(params, output)
        else:
            raise ValueError(
                f"Unknown method: {method}. Please choose from Karhunen-Loève ('KL'), Polynomial Chaos Expansion ('PCE'), or Monte Carlo ('MC').\n"
            )

        return self.sobol_indices

    def _KL_analysis(self, params, output):
        """A method that performs TD-GSA using a Karhunen-Loeve expansion."""
        # Center the output and interpolate to quadrature nodes
        mean = np.mean(output, axis=0)
        output_centered = np.array([out - mean for out in output])
        timesteps_solver = self.timesteps_solver
        timesteps_quadrature = np.linspace(
            timesteps_solver[0], timesteps_solver[-1], self.num_timesteps_quadrature
        )
        centered_outputs_quadrature = np.array(
            [
                np.interp(timesteps_quadrature, timesteps_solver, output)
                for output in output_centered
            ]
        )

        # Form covariance matrix
        covariance_matrix = np.zeros(
            (len(timesteps_quadrature), len(timesteps_quadrature))
        )

        for l in range(len(timesteps_quadrature)):
            for m in range(len(timesteps_quadrature)):
                covariance_matrix[l, m] = (1 / (self.num_samples + 1)) * np.sum(
                    centered_outputs_quadrature[:, l]
                    * centered_outputs_quadrature[:, m]
                )
        # save covaraicne matrix for later plotting
        self._covariance_matrix = covariance_matrix

        # Let W = diag(w_1, ..., w_n) and solve the eigenvalue problem
        h = timesteps_quadrature[1] - timesteps_quadrature[0]
        weights = np.ones(len(timesteps_quadrature)) * h
        weights[0] = 0.5 * h
        weights[-1] = 0.5 * h

        W = np.diag(weights)

        W_sqrt = np.sqrt(W)

        discretized_covariance_matrix = W_sqrt @ covariance_matrix @ W_sqrt
        eigenvalues, eigenvectors = np.linalg.eigh(discretized_covariance_matrix)
        sorted_eigenvalues_args = eigenvalues.argsort()[::-1]
        # descending order
        sorted_eigenvalues = eigenvalues[sorted_eigenvalues_args]
        # descending order
        sorted_eigenvectors = eigenvectors[:, sorted_eigenvalues_args]
        W_inverse_sqrt = np.linalg.inv(W_sqrt)
        sorted_eigenvectors = W_inverse_sqrt @ sorted_eigenvectors

        # save eigenvalue spectrum and the variance ratio for later plotting and sanity checks
        self._sorted_eigenvalues = sorted_eigenvalues
        self._sorted_eigenvalues_normed = sorted_eigenvalues / sorted_eigenvalues[0]
        self._r_Nkl = [
            sum(sorted_eigenvalues[:i]) / sum(sorted_eigenvalues)
            for i in range(len(sorted_eigenvalues) // 3)
        ]
        if self._r_Nkl[-1] < 0.90:
            raise Warning(
                "The variance ratio is less than 90%. Consider increasing the truncation level. \n You can view the eigenvalue spectrum by calling the plot_eigenvalue_spectrum method.\n"
            )

        # Choose a truncation level N_kl and compute the discretized KL modes
        N_kl = self.KL_truncation_level
        KL_modes = np.zeros((self.num_samples, N_kl))

        for i in range(N_kl):
            for k in range(self.num_samples):
                for m in range(len(timesteps_quadrature)):
                    KL_modes[k, i] += (
                        weights[m]
                        * sorted_eigenvectors[m, i]
                        * centered_outputs_quadrature[k, m]
                    )

        # Compute a surrogate model for each KL mode
        PCE_order = self.PCE_order
        PCE_option = self.PCE_option
        joint_dist = self.simulator.dist.dist

        expansion, norms = cp.generate_expansion(
            PCE_order, joint_dist, normed=True, graded=False, retall=True
        )
        num_cores = multiprocessing.cpu_count()

        if PCE_option == "regression":
            surrogate_models = Parallel(n_jobs=num_cores)(
                delayed(cp.fit_regression)(
                    expansion,
                    params.T,
                    KL_modes[:, i],
                    retall=1,
                    model=linear_model.LinearRegression(fit_intercept=False),
                )
                for i in range(N_kl)
            )
        elif PCE_option == "quadrature":
            surrogate_models = Parallel(n_jobs=num_cores)(
                delayed(cp.fit_quadrature)(
                    expansion, params.T, self.PCE_quad_weights, KL_modes[:, i], retall=1
                )
                for i in range(N_kl)
            )
        else:
            raise ValueError(
                "Unknown PCE option. Please choose from regression or quadrature.\n"
            )

        surrogate_model_coeffs = [surrogate_models[i][1] for i in range(N_kl)]
        surrogate_model_poly = surrogate_models[0][0]

        # Compute the generalized Sobol indices
        surrogate_model_poly_dict = surrogate_model_poly.todict()
        sum_coeff_per_param_total = np.zeros((N_kl, self.num_params))
        sum_coeff_per_param_first = np.zeros((N_kl, self.num_params))

        masks_total = []
        masks_first = []
        for i in range(self.num_params):
            mask_total = [1 if key[i] == 0 else 0 for key in surrogate_model_poly_dict.keys()]
            masks_total.append(mask_total)
            mask_first = [1 if key[i] != 0 and key.count(0) == (len(key)-1) else 0 for key in surrogate_model_poly_dict.keys()]
            masks_first.append(mask_first)
        masks_total = np.array(masks_total)
        masks_first = np.array(masks_first)

        # assuming that ||q_k||^2 = 1 for all k when using normalized polynomials
        for i in range(self.num_params):
            for j in range(N_kl):
                # sum all squared coefficients where the term does not contain q_j
                sum_coeff_per_param_total[j, i] = np.sum(
                    surrogate_model_coeffs[j] ** 2 * masks_total[i]
                )
                # sum all squared coefficients where the term contains only q_j
                sum_coeff_per_param_first[j, i] = np.sum(
                    surrogate_model_coeffs[j] ** 2 * masks_first[i]
                )

        sum_eigenvalues = sum(sorted_eigenvalues)
        sobol_indices_total = np.ones(self.num_params)
        sobol_indices_first = np.zeros(self.num_params)

        for i in range(self.num_params):
            sobol_indices_total[i] -= sum(sum_coeff_per_param_total[:, i]) / sum_eigenvalues
            sobol_indices_first[i] = sum(sum_coeff_per_param_first[:, i]) / sum_eigenvalues
        sobol_indices = np.zeros((self.num_params, 2))
        sobol_indices[:, 0] = sobol_indices_first
        sobol_indices[:, 1] = sobol_indices_total
        sobol_indices = pd.DataFrame(sobol_indices, columns=["first", "total"], index=self.param_names)
        self.sobol_indices = sobol_indices

    def _PCE_analysis(self, params, output):
        """A method that performs TD-GSA using Polynomial Chaos Expansion surrogate models."""
        # Construct pointwise-in-time PCEs
        timesteps_solver = self.timesteps_solver
        timesteps_quadrature = np.linspace(
            timesteps_solver[0], timesteps_solver[-1], self.num_timesteps_quadrature
        )
        outputs_quadrature = np.array(
            [np.interp(timesteps_quadrature, timesteps_solver, out) for out in output]
        )

        PCE_order = self.PCE_order
        PCE_option = self.PCE_option
        joint_dist = self.simulator.dist.dist

        expansion, norms = cp.generate_expansion(
            PCE_order, joint_dist, normed=True, graded=False, retall=True
        )
        num_cores = multiprocessing.cpu_count()

        if PCE_option == "regression":
            surrogate_models_pointwise = Parallel(n_jobs=num_cores)(
                delayed(cp.fit_regression)(
                    expansion,
                    params.T,
                    outputs_quadrature[:, m],
                    retall=1,
                    model=linear_model.LinearRegression(fit_intercept=False),
                )
                for m in range(len(timesteps_quadrature))
            )
        elif PCE_option == "quadrature":
            surrogate_models_pointwise = Parallel(n_jobs=num_cores)(
                delayed(cp.fit_quadrature)(
                    expansion,
                    params.T,
                    self.PCE_quad_weights,
                    outputs_quadrature[:, m],
                    retall=1,
                )
                for m in range(len(timesteps_quadrature))
            )
        else:
            raise ValueError(
                "Unknown PCE option. Please choose from regression or quadrature.\n"
            )

        polynomial_pointwise_dict = surrogate_models_pointwise[0][0].todict()
        coeff_pointwise = [
            surrogate_models_pointwise[m][1] for m in range(len(timesteps_quadrature))
        ]
        polynomial_pointwise = [surrogate_models_pointwise[m][0] for m in range(len(timesteps_quadrature))]
        # save for later computation of second and third order sobol indices
        self._polynomial_pointwise_dict = polynomial_pointwise_dict
        self._coeff_pointwise = coeff_pointwise
        self._polynomial_pointwise = polynomial_pointwise

        # Generate masks to select coefficients for each parameter depending on occurence in expansion
        masks_total = []
        masks_first = []
        for i in range(self.num_params):
            mask_total = [1 if key[i] != 0 else 0 for key in polynomial_pointwise_dict.keys()]
            mask_first = [1 if key[i] != 0 and key.count(0) == (len(key)-1) else 0 for key in polynomial_pointwise_dict.keys()]
            masks_total.append(mask_total)
            masks_first.append(mask_first)
        masks_total = np.array(masks_total)
        masks_first = np.array(masks_first)

        # Compute variances for each parameter at each time step

        variance_over_time_total = np.zeros(
            (len(timesteps_quadrature), self.num_params), dtype=np.double
        )
        variance_over_time_first = np.zeros(
            (len(timesteps_quadrature), self.num_params), dtype=np.double
        )
        total_variance_over_time = np.zeros(len(timesteps_quadrature), dtype=np.double)

        for m in range(len(timesteps_quadrature)):

            total_variance_over_time[m] = np.sum(coeff_pointwise[m][1:] ** 2)

            for i in range(self.num_params):

                variance_over_time_total[m, i] = np.sum(coeff_pointwise[m] ** 2 * masks_total[i])
                variance_over_time_first[m, i] = np.sum(coeff_pointwise[m] ** 2 * masks_first[i])

        # Compute the generalized Sobol indices
        td_sobol_indices_total = np.ones((len(timesteps_quadrature), self.num_params))
        td_sobol_indices_first = np.zeros((len(timesteps_quadrature), self.num_params))

        for m in range(len(timesteps_quadrature)):
            for i in range(self.num_params):

                if m == 0:

                    td_sobol_indices_total[m, i] = np.NaN
                    td_sobol_indices_first[m, i] = np.NaN

                else:

                    h = timesteps_quadrature[1] - timesteps_quadrature[0]
                    weights = np.ones(m) * h
                    weights[0] = 0.5 * h
                    weights[-1] = 0.5 * h

                    denum = np.dot(np.asfarray(total_variance_over_time[:m]), weights)
                    td_sobol_indices_total[m, i] = (
                        np.dot(np.asfarray(variance_over_time_total[:m, i]), weights) / denum
                    )
                    td_sobol_indices_first[m, i] = ( 
                        np.dot(np.asfarray(variance_over_time_first[:m, i]), weights) / denum
                    )
        sobol_indices = np.zeros((self.num_params, 2))
        sobol_indices[:, 0] = td_sobol_indices_first[-1,:]
        sobol_indices[:, 1] = td_sobol_indices_total[-1,:]
        sobol_indices = pd.DataFrame(sobol_indices, columns=["first", "total"], index=self.param_names)
        self.sobol_indices = sobol_indices
        
        td_sobol_indices_total = pd.DataFrame(td_sobol_indices_total, columns=self.param_names, index=timesteps_quadrature)
        td_sobol_indices_first = pd.DataFrame(td_sobol_indices_first, columns=self.param_names, index=timesteps_quadrature)
        td_sobol_indices = pd.concat([td_sobol_indices_first, td_sobol_indices_total], keys=["first", "total"], axis=1)
        self.td_sobol_indices = td_sobol_indices

    def _MC_analysis(self, params, output):
        """A method that performs TD-GSA using Monte Carlo estimators."""
        raise NotImplementedError("MC method not yet implemented.")
    
    def compute_second_order_sobol_indices(self):
        """A method that computes second order Sobol' indices."""
        if self._coeff_pointwise is None:
            raise ValueError(
                "No polynomial coefficients available. Please run a PCE method first.\n"
            )
        timesteps_solver = self.timesteps_solver
        timesteps_quadrature = np.linspace(
            timesteps_solver[0], timesteps_solver[-1], self.num_timesteps_quadrature)
        
        masks_second = []
        param_combinations = []
        for i in range(self.num_params):
            for j in range(i+1, self.num_params):
                mask_second = [1 if key[i] != 0 and key[j] != 0 and key.count(0)==(len(key)-2) else 0 for key in self._polynomial_pointwise_dict.keys()]
                masks_second.append(mask_second)
                param_combinations.append(self.param_names[i] + " " + self.param_names[j])
        masks_second = np.array(masks_second)
        
        # Compute variances for each parameter at each time step

        variance_over_time_second = np.zeros(
            (len(timesteps_quadrature), len(param_combinations)), dtype=np.double
        )
        total_variance_over_time = np.zeros(len(timesteps_quadrature), dtype=np.double)

        for m in range(len(timesteps_quadrature)):

            total_variance_over_time[m] = np.sum(self._coeff_pointwise[m][1:] ** 2)

            for i in range(len(param_combinations)):

                variance_over_time_second[m, i] = np.sum(self._coeff_pointwise[m] ** 2 * masks_second[i])

        # Compute the generalized Sobol indices
        td_sobol_indices_second = np.zeros((len(timesteps_quadrature), len(param_combinations)))

        for m in range(len(timesteps_quadrature)):
            for i in range(len(param_combinations)):

                if m == 0:

                    td_sobol_indices_second[m, i] = np.NaN

                else:

                    h = timesteps_quadrature[1] - timesteps_quadrature[0]
                    weights = np.ones(m) * h
                    weights[0] = 0.5 * h
                    weights[-1] = 0.5 * h

                    denum = np.dot(np.asfarray(total_variance_over_time[:m]), weights)
                    td_sobol_indices_second[m, i] = (
                        np.dot(np.asfarray(variance_over_time_second[:m, i]), weights) / denum
                    )
        second_order_sobol_indices = td_sobol_indices_second[-1,:]
        
        return second_order_sobol_indices, td_sobol_indices_second, param_combinations        
    
    def compute_third_order_sobol_indices(self):
        """A method that computes third order Sobol' indices."""
        if self._coeff_pointwise is None:
            raise ValueError(
                "No polynomial coefficients available. Please run a PCE method first.\n"
            )
        timesteps_solver = self.timesteps_solver
        timesteps_quadrature = np.linspace(
            timesteps_solver[0], timesteps_solver[-1], self.num_timesteps_quadrature)
        
        masks_third = []
        param_combinations = []
        for i in range(self.num_params):
            for j in range(i+1, self.num_params):
                for k in range(j+1, self.num_params):
                    mask_third = [1 if key[i] != 0 and key[j] != 0 and key[k] != 0 and key.count(0)==(len(key)-3) else 0 for key in self._polynomial_pointwise_dict.keys()]
                    masks_third.append(mask_third)
                    param_combinations.append(self.param_names[i] + " " + self.param_names[j] + " " + self.param_names[k])
        masks_third = np.array(masks_third)
        
        # Compute variances for each parameter at each time step

        variance_over_time_third = np.zeros(
            (len(timesteps_quadrature), len(param_combinations)), dtype=np.double
        )
        total_variance_over_time = np.zeros(len(timesteps_quadrature), dtype=np.double)

        for m in range(len(timesteps_quadrature)):

            total_variance_over_time[m] = np.sum(self._coeff_pointwise[m][1:] ** 2)

            for i in range(len(param_combinations)):

                variance_over_time_third[m, i] = np.sum(self._coeff_pointwise[m] ** 2 * masks_third[i])

        # Compute the generalized Sobol indices
        td_sobol_indices_third = np.zeros((len(timesteps_quadrature), len(param_combinations)))

        for m in range(len(timesteps_quadrature)):
            for i in range(len(param_combinations)):

                if m == 0:

                    td_sobol_indices_third[m, i] = np.NaN

                else:

                    h = timesteps_quadrature[1] - timesteps_quadrature[0]
                    weights = np.ones(m) * h
                    weights[0] = 0.5 * h
                    weights[-1] = 0.5 * h

                    denum = np.dot(np.asfarray(total_variance_over_time[:m]), weights)
                    td_sobol_indices_third[m, i] = (
                        np.dot(np.asfarray(variance_over_time_third[:m, i]), weights) / denum
                    )
        third_order_sobol_indices = td_sobol_indices_third[-1,:]
        
        return third_order_sobol_indices, td_sobol_indices_third, param_combinations
    
    def PCE_surrogate(self, param):
        if self._polynomial_pointwise is None:
            raise ValueError(
                "No PCE surrogate models available. Please run the PCE method first.\n"
            )
        result = [cp.call(self._polynomial_pointwise[m],param) for m in range(self.num_timesteps_quadrature)]
        return np.array(result)
    
    def plot(self, plot_option: str, return_fig: bool=False) -> Optional[plt.Figure]:
        """A method that plots the results of the time-dependent sensitivity analysis."""
        if plot_option == "sobol_indices":
            fig = self._plot_sobol_indices()
        elif plot_option == "time_dependent_sobol_indices":
            fig = self._plot_time_dependent_sobol_indices()
        elif plot_option == "simulator_output":
            fig = self._plot_simulator_output()
        elif plot_option == "covariance_matrix":
            fig = self._plot_covariance_matrix()
        elif plot_option == "eigenvalue_spectrum":
            fig = self._plot_eigenvalue_spectrum()
        else:
            raise ValueError(
                f"Unknown plot option: {plot_option}. Please choose from 'sobol_indices', 'time_dependent_sobol_indices', 'covariance_matrix', or 'eigenvalue_spectrum'.\n"
            )
        if return_fig:
            return fig
        else:
            plt.show()

    def _plot_sobol_indices(self):
        """A method that plots the generalized Sobol indices."""
        if self.sobol_indices is None:
            raise ValueError(
                "No Sobol indices available. Please run a TD-GSA method first.\n"
            )
        else:
            fig, ax = plt.subplots()
            x = np.arange(self.num_params)
            ax.bar(x-0.055, self.sobol_indices["first"], label="first order", width=0.1)
            ax.bar(x+0.055, self.sobol_indices["total"], label="total order", width=0.1)
            ax.set_xticks(x, self.param_names)
            ax.set_ylabel("Generalized Sobol' index")
            ax.legend()

    def _plot_time_dependent_sobol_indices(self):
        """A method that plots the time-evolution of the generalized Sobol indices."""
        timesteps_quadrature = np.linspace(
            self.timesteps_solver[0],
            self.timesteps_solver[-1],
            self.num_timesteps_quadrature,
        )

        if self.td_sobol_indices is None:
            raise ValueError(
                "No time-dependent Sobol indices available. Please run a TD-GSA method (other than 'KL') first.\n"
            )
        else:

            fig, ax = plt.subplots(2, 1, sharex=True)
            for i in range(self.num_params):
                ax[0].plot(
                    timesteps_quadrature,
                    self.td_sobol_indices["first"][self.param_names[i]],
                    label=self.param_names[i],
                )
                ax[1].plot(
                    timesteps_quadrature,
                    self.td_sobol_indices["total"][self.param_names[i]],
                    label=self.param_names[i],
                )
            ax[1].set_xlabel("Time / s")
            ax[0].set_ylabel("Generalized \n first order Sobol' index")
            ax[1].set_ylabel("Generalized \n total order Sobol' index")
            ax[0].legend()
            ax[1].legend()
            plt.tight_layout()
            plt.show()
            
    def _plot_simulator_output(self):
        pass

    def _plot_covariance_matrix(self):
        """A method that plots the covariance matrix."""
        if self._covariance_matrix is None:
            raise ValueError(
                "No covariance matrix available. Please run the KL method first.\n"
            )
        else:
            plt.figure()
            sns.heatmap(self._covariance_matrix)
            plt.show()

    def _plot_eigenvalue_spectrum(self):
        """A method that plots the eigenvalue spectrum to check spectral decay."""
        if self._sorted_eigenvalues_normed is None:
            raise ValueError(
                "No eigenvalues available. Please run the KL method first.\n"
            )
        else:
            fig, axes = plt.subplots(1, 3)
            axes[0].plot(self._sorted_eigenvalues_normed, "x")
            axes[0].set_xlabel("Eigenvalue index $i$")
            axes[0].set_ylabel("Normalized eigenvalue $\lambda_i / \lambda_0$")
            axes[0].set_yscale("log")
            axes[1].plot(self._sorted_eigenvalues, "x")
            axes[1].set_xlabel("Eigenvalue index $i$")
            axes[1].set_ylabel("Eigenvalue $\lambda_i$")
            axes[2].plot(self._r_Nkl, "x")
            axes[2].set_xlabel("Truncation level $N_{kl}$")
            axes[2].set_ylabel("Variance ratio $r_{N_{kl}}$")
            plt.tight_layout()
            plt.show()
