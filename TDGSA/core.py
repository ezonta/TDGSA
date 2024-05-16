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
from typing import Callable, Optional, Union, Dict
from numpy.typing import NDArray
from tqdm.autonotebook import tqdm


# TODO: Better docstrings
# TODO: Make KL surrogate model a callable method like PCE
# TODO: Implement pointwise-in-time GPR surrogate models
# TODO: Implement MC method


class time_dependent_sensitivity_analysis:
    """A class that performs time-dependent global sensitivity analysis."""

    # public
    simulator: utils.simulator
    distribution: utils.distribution
    param_names: list[str]

    params: Optional[pd.DataFrame]
    outputs: Optional[pd.DataFrame]

    num_samples: Optional[int]
    timesteps_solver: NDArray

    sobol_indices: Dict[str, pd.DataFrame]
    td_sobol_indices: Dict[str, pd.DataFrame]
    higher_order_sobol_indices: Dict[str, pd.DataFrame]
    td_higher_order_sobol_indices: Dict[str, pd.DataFrame]

    # private
    _num_timesteps_quadrature: Optional[int]

    _PCE_option: Optional[str]
    _PCE_quad_weights: Optional[NDArray]

    _KL_truncation_level: Optional[int]
    _covariance_matrix: Optional[NDArray]
    _sorted_eigenvalues: Optional[NDArray]
    _sorted_eigenvectors: Optional[NDArray]
    _sorted_eigenvalues_normed: Optional[NDArray]
    _r_Nkl: Optional[NDArray]
    _KL_mean: Optional[NDArray]

    _polynomial_dict: Optional[Dict]
    _PCE_coeffs: Dict[str, list[NDArray]]
    _polynomial_pointwise: Dict[str, Optional[list[cp.ndpoly]]]

    _param_combinations_second_order: Optional[list[str]]
    _param_combinations_third_order: Optional[list[str]]

    def __init__(
        self,
        simulator: utils.simulator,
        distribution: utils.distribution,
        data: Optional[tuple[pd.DataFrame, pd.DataFrame]] = None,
        **kwargs: Union[int, str],
    ) -> None:

        self.simulator = simulator
        self.distribution = distribution

        if data is not None:
            self.params = data[0]
            self.outputs = data[1]
        else:
            self.params = None
            self.outputs = None

        self.param_names = self.distribution.param_names
        self.num_samples = None
        self.num_params = self.distribution.dim
        self.timesteps_solver = self.simulator.time

        self.sobol_indices = {}
        self.td_sobol_indices = {}
        self.higher_order_sobol_indices = {}
        self.td_higher_order_sobol_indices = {}

        self._num_timesteps_quadrature = None

        self._PCE_option = None
        self._PCE_quad_weights = None

        self._KL_truncation_level = None
        self._covariance_matrix = None
        self._sorted_eigenvalues = None
        self._sorted_eigenvectors = None
        self._sorted_eigenvalues_normed = None
        self._r_Nkl = None
        self._KL_mean = None

        self._polynomial_dict = None
        self._PCE_coeffs = {}
        self._polynomial_pointwise = {"PCE": None, "KL": None}

        self._param_combinations_second_order = None
        self._param_combinations_third_order = None

        # TODO: remove together with MC method
        # random, halton, latin_hypercube, sobol, etc.
        self.MC_option = kwargs.get("MC_option", "random")

    def sample_params_and_run_simulator(
        self,
        num_samples: int,
        sampling_method: str = "random",
        **kwargs: Union[int, str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """A method that generates parameter samples and runs the simulator.

        Options:

        sampling_method
            - 'random': random sampling
            - 'quasirandom': quasi-random sampling (quasirandom_rule: 'halton', 'sobol', 'latin_hypercube')
            - 'quadrature': quadrature sampling with nodes and weights (quadrature_rule: 'clenshaw_curtis', 'gaussian', 'legendre' + quadrature_order)
        """

        ## sampling of parameters
        print("Sampling parameters ...\n")

        if sampling_method == "random":
            samples = self.distribution.sample(num_samples=num_samples, rule="random")
            self._PCE_option = "regression"

        elif sampling_method == "quasirandom":
            quasirandom_rule = kwargs.get("quasirandom_method", "halton")
            if (
                quasirandom_rule != "halton"
                and quasirandom_rule != "sobol"
                and quasirandom_rule != "latin_hypercube"
            ):
                raise ValueError(
                    f"Unknown quasirandom method: {quasirandom_rule}. Please choose from 'halton', 'sobol', or 'latin_hypercube'.\n"
                )
            samples = self.distribution.sample(
                num_samples=num_samples, rule=quasirandom_rule
            )
            self._PCE_option = "regression"

        elif sampling_method == "quadrature":
            quadrature_rule = kwargs.get("quadrature_method", "clenshaw_curtis")
            quadrature_order = kwargs.get("quadrature_order", 4)
            if (
                quadrature_rule != "clenshaw_curtis"
                and quadrature_rule != "gaussian"
                and quadrature_rule != "legendre"
            ):
                raise ValueError(
                    f"Unknown quadrature method: {quadrature_rule}. Please choose from 'clenshaw_curtis', 'gaussian', or 'legendre'.\n"
                )
            samples, weights = cp.generate_quadrature(
                quadrature_order, self.distribution.dist, rule=quadrature_rule
            )
            self._PCE_option = "quadrature"
            self._PCE_quad_weights = weights

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
        self.num_samples = num_samples

        return params, outputs

    def compute_sobol_indices(self, method: str, **kwargs) -> pd.DataFrame:
        """A method that runs the time-dependent sensitivity analysis and returns the generalized sobol indices.

        Options:
            - method: 'KL', 'PCE'

        kwargs:
            - num_timesteps_quadrature: number of quadrature nodes in time (default is 100)
            - KL_truncation_level: truncation level for the Karhunen-Loève expansion (default is 8)
            - PCE_order: order of the Polynomial Chaos Expansion (default is 4)"""
        if self.outputs is None:
            raise ValueError(
                "No data available. Please run sample_params_and_run_simulator() first.\n"
            )

        # TODO: read out kwargs that are currently in constructor method

        if method == "KL":
            self._KL_analysis(self.params.to_numpy(), self.outputs.to_numpy(), **kwargs)
        elif method == "PCE":
            self._PCE_analysis(
                self.params.to_numpy(), self.outputs.to_numpy(), **kwargs
            )
        elif method == "MC":
            self._MC_analysis()
        else:
            raise ValueError(
                f"Unknown method: {method}. Please choose from Karhunen-Loève ('KL'), Polynomial Chaos Expansion ('PCE'), or Monte Carlo ('MC').\n"
            )

        return self.sobol_indices[method]

    def _KL_analysis(
        self, params: NDArray, output: NDArray, **kwargs: Union[int, str]
    ) -> None:
        """A method that performs TD-GSA using a Karhunen-Loève expansion."""
        # Center the output and interpolate to quadrature nodes
        mean = np.mean(output, axis=0)
        output_centered = np.array([out - mean for out in output])
        timesteps_solver = self.timesteps_solver
        self._num_timesteps_quadrature = kwargs.get("num_timesteps_quadrature", 100)
        timesteps_quadrature = np.linspace(
            timesteps_solver[0], timesteps_solver[-1], self._num_timesteps_quadrature
        )
        centered_outputs_quadrature = np.array(
            [
                np.interp(timesteps_quadrature, timesteps_solver, output)
                for output in output_centered
            ]
        )
        self._KL_mean = np.interp(timesteps_quadrature, timesteps_solver, mean)

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
        # save covariance matrix for later plotting
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
        self._sorted_eigenvectors = sorted_eigenvectors
        self._sorted_eigenvalues_normed = sorted_eigenvalues / sorted_eigenvalues[0]
        self._r_Nkl = [
            sum(sorted_eigenvalues[:i]) / sum(sorted_eigenvalues)
            for i in range(len(sorted_eigenvalues))
        ]
        if self._r_Nkl[-1] < 0.90:
            raise Warning(
                "The variance ratio is less than 90%. Consider increasing the truncation level. \n You can view the eigenvalue spectrum by calling the plot() method.\n"
            )

        # Choose a truncation level N_kl and compute the discretized KL modes
        N_kl = kwargs.get("KL_truncation_level", 8)
        self._KL_truncation_level = N_kl
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
        PCE_order = kwargs.get("PCE_order", 4)
        PCE_option = self._PCE_option
        joint_dist = self.distribution.dist

        print("Generating PCE expansion ...\n")

        expansion, norms = cp.generate_expansion(
            PCE_order, joint_dist, normed=True, graded=False, retall=True
        )

        print("Fitting surrogate models ...\n")

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
                for i in tqdm(range(N_kl))
            )
        elif PCE_option == "quadrature":
            if self._PCE_quad_weights is None:
                raise ValueError(
                    "No quadrature weights available. Please run sample_params_and_run_simulator() with the quadrature option first.\n"
                )
            surrogate_models = Parallel(n_jobs=num_cores)(
                delayed(cp.fit_quadrature)(
                    expansion,
                    params.T,
                    self._PCE_quad_weights,
                    KL_modes[:, i],
                    retall=1,
                )
                for i in tqdm(range(N_kl))
            )

        surrogate_model_coeffs = [surrogate_models[i][1] for i in range(N_kl)]
        polynomial_pointwise = [surrogate_models[i][0] for i in range(N_kl)]
        surrogate_model_poly = surrogate_models[0][0]

        # Compute the generalized Sobol indices
        surrogate_model_poly_dict = surrogate_model_poly.todict()
        self._polynomial_dict = surrogate_model_poly_dict
        self._PCE_coeffs["KL"] = surrogate_model_coeffs
        self._polynomial_pointwise["KL"] = polynomial_pointwise
        sum_coeff_per_param_total = np.zeros((N_kl, self.num_params))
        sum_coeff_per_param_first = np.zeros((N_kl, self.num_params))

        masks_total = []
        masks_first = []
        for i in range(self.num_params):
            mask_total = [
                1 if key[i] == 0 else 0 for key in surrogate_model_poly_dict.keys()
            ]
            masks_total.append(mask_total)
            mask_first = [
                1 if key[i] != 0 and key.count(0) == (len(key) - 1) else 0
                for key in surrogate_model_poly_dict.keys()
            ]
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
            sobol_indices_total[i] -= (
                sum(sum_coeff_per_param_total[:, i]) / sum_eigenvalues
            )
            sobol_indices_first[i] = (
                sum(sum_coeff_per_param_first[:, i]) / sum_eigenvalues
            )
        sobol_indices = np.zeros((self.num_params, 2))
        sobol_indices[:, 0] = sobol_indices_first
        sobol_indices[:, 1] = sobol_indices_total
        sobol_indices = pd.DataFrame(
            sobol_indices, columns=["first", "total"], index=self.param_names
        )
        self.sobol_indices["KL"] = sobol_indices

    def _PCE_analysis(
        self, params: NDArray, output: NDArray, **kwargs: Union[int, str]
    ) -> None:
        """A method that performs TD-GSA using Polynomial Chaos Expansion surrogate models."""
        # Construct pointwise-in-time PCEs
        timesteps_solver = self.timesteps_solver
        self._num_timesteps_quadrature = kwargs.get("num_timesteps_quadrature", 100)
        timesteps_quadrature = np.linspace(
            timesteps_solver[0], timesteps_solver[-1], self._num_timesteps_quadrature
        )
        outputs_quadrature = np.array(
            [np.interp(timesteps_quadrature, timesteps_solver, out) for out in output]
        )

        PCE_order = kwargs.get("PCE_order", 4)
        PCE_option = self._PCE_option
        joint_dist = self.distribution.dist

        print("Generating PCE expansion ...\n")

        expansion, norms = cp.generate_expansion(
            PCE_order, joint_dist, normed=True, graded=False, retall=True
        )

        print("Fitting surrogate models ...\n")

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
                for m in tqdm(range(len(timesteps_quadrature)))
            )
        elif PCE_option == "quadrature":
            surrogate_models_pointwise = Parallel(n_jobs=num_cores)(
                delayed(cp.fit_quadrature)(
                    expansion,
                    params.T,
                    self._PCE_quad_weights,
                    outputs_quadrature[:, m],
                    retall=1,
                )
                for m in tqdm(range(len(timesteps_quadrature)))
            )

        polynomial_pointwise_dict = surrogate_models_pointwise[0][0].todict()
        coeff_pointwise = [
            surrogate_models_pointwise[m][1] for m in range(len(timesteps_quadrature))
        ]
        polynomial_pointwise = [
            surrogate_models_pointwise[m][0] for m in range(len(timesteps_quadrature))
        ]
        # save for later computation of second and third order sobol indices and PCE surrogate evaluation
        self._polynomial_dict = polynomial_pointwise_dict
        self._PCE_coeffs["PCE"] = coeff_pointwise
        self._polynomial_pointwise["PCE"] = polynomial_pointwise

        # Generate masks to select coefficients for each parameter depending on occurence in expansion
        masks_total = []
        masks_first = []
        for i in range(self.num_params):
            mask_total = [
                1 if key[i] != 0 else 0 for key in polynomial_pointwise_dict.keys()
            ]
            mask_first = [
                1 if key[i] != 0 and key.count(0) == (len(key) - 1) else 0
                for key in polynomial_pointwise_dict.keys()
            ]
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

                variance_over_time_total[m, i] = np.sum(
                    coeff_pointwise[m] ** 2 * masks_total[i]
                )
                variance_over_time_first[m, i] = np.sum(
                    coeff_pointwise[m] ** 2 * masks_first[i]
                )

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
                        np.dot(np.asfarray(variance_over_time_total[:m, i]), weights)
                        / denum
                    )
                    td_sobol_indices_first[m, i] = (
                        np.dot(np.asfarray(variance_over_time_first[:m, i]), weights)
                        / denum
                    )
        sobol_indices = np.zeros((self.num_params, 2))
        sobol_indices[:, 0] = td_sobol_indices_first[-1, :]
        sobol_indices[:, 1] = td_sobol_indices_total[-1, :]
        sobol_indices = pd.DataFrame(
            sobol_indices, columns=["first", "total"], index=self.param_names
        )
        self.sobol_indices["PCE"] = sobol_indices

        td_sobol_indices_total = pd.DataFrame(
            td_sobol_indices_total, columns=self.param_names, index=timesteps_quadrature
        )
        td_sobol_indices_first = pd.DataFrame(
            td_sobol_indices_first, columns=self.param_names, index=timesteps_quadrature
        )
        self.td_sobol_indices["first"] = td_sobol_indices_first
        self.td_sobol_indices["total"] = td_sobol_indices_total

    def _MC_analysis(self) -> None:
        """A method that performs TD-GSA using Monte Carlo estimators."""
        raise NotImplementedError("MC method not yet implemented.")

    # TODO: make possible to compute with KL as well
    def compute_second_order_sobol_indices(
        self, method: str
    ) -> tuple[pd.DataFrame, list[str]]:
        """A method that computes second order Sobol' indices."""
        if self._PCE_coeffs is {}:
            raise ValueError(
                "No polynomial coefficients available. Please run compute_sobol_indices() first.\n"
            )

        timesteps_solver = self.timesteps_solver
        timesteps_quadrature = np.linspace(
            timesteps_solver[0], timesteps_solver[-1], self._num_timesteps_quadrature
        )

        masks_second = []
        param_combinations = []
        for i in range(self.num_params):
            for j in range(i + 1, self.num_params):
                mask_second = [
                    (
                        1
                        if key[i] != 0
                        and key[j] != 0
                        and key.count(0) == (len(key) - 2)
                        else 0
                    )
                    for key in self._polynomial_dict.keys()
                ]
                masks_second.append(mask_second)
                param_combinations.append(
                    self.param_names[i] + " " + self.param_names[j]
                )
        masks_second = np.array(masks_second)

        # Compute variances for each parameter at each time step

        coeffs = self._PCE_coeffs[method]

        if method == "PCE":

            variance_over_time_second = np.zeros(
                (len(timesteps_quadrature), len(param_combinations)), dtype=np.double
            )
            total_variance_over_time = np.zeros(
                len(timesteps_quadrature), dtype=np.double
            )

            for m in range(len(timesteps_quadrature)):

                total_variance_over_time[m] = np.sum(coeffs[m][1:] ** 2)

                for i in range(len(param_combinations)):

                    variance_over_time_second[m, i] = np.sum(
                        coeffs[m] ** 2 * masks_second[i]
                    )

            # Compute the generalized Sobol indices
            td_sobol_indices_second = np.zeros(
                (len(timesteps_quadrature), len(param_combinations))
            )

            for m in range(len(timesteps_quadrature)):
                for i in range(len(param_combinations)):

                    if m == 0:

                        td_sobol_indices_second[m, i] = np.NaN

                    else:

                        h = timesteps_quadrature[1] - timesteps_quadrature[0]
                        weights = np.ones(m) * h
                        weights[0] = 0.5 * h
                        weights[-1] = 0.5 * h

                        denum = np.dot(
                            np.asfarray(total_variance_over_time[:m]), weights
                        )
                        td_sobol_indices_second[m, i] = (
                            np.dot(
                                np.asfarray(variance_over_time_second[:m, i]), weights
                            )
                            / denum
                        )
            sobol_indices_second = pd.DataFrame(
                td_sobol_indices_second[-1, :],
                columns=["second"],
                index=param_combinations,
            )
            td_sobol_indices_second = pd.DataFrame(
                td_sobol_indices_second,
                columns=param_combinations,
                index=timesteps_quadrature,
            )
            self.higher_order_sobol_indices["second_PCE"] = sobol_indices_second
            self.td_higher_order_sobol_indices["second_PCE"] = td_sobol_indices_second
            self._param_combinations_second_order = param_combinations

        elif method == "KL":

            sum_coeff_per_param_combination = np.zeros(
                (self._KL_truncation_level, len(param_combinations))
            )

            for i in range(len(param_combinations)):
                for j in range(self._KL_truncation_level):
                    sum_coeff_per_param_combination[j, i] = np.sum(
                        coeffs[j] ** 2 * masks_second[i]
                    )

            sum_eigenvalues = sum(self._sorted_eigenvalues)
            sobol_indices_second = np.zeros(len(param_combinations))

            for i in range(len(param_combinations)):
                sobol_indices_second[i] = (
                    sum(sum_coeff_per_param_combination[:, i]) / sum_eigenvalues
                )

            sobol_indices_second = pd.DataFrame(
                sobol_indices_second, columns=["second"], index=param_combinations
            )
            self.higher_order_sobol_indices["second_KL"] = sobol_indices_second
            self._param_combinations_second_order = param_combinations

        else:
            raise ValueError(
                f"Unknown method: {method}. Please choose from Karhunen-Loève ('KL') or Polynomial Chaos Expansion ('PCE').\n"
            )

        key = "second_" + method
        return (self.higher_order_sobol_indices[key], param_combinations)

    # TODO: same changes as in second order computation
    def compute_third_order_sobol_indices(
        self, method: str
    ) -> tuple[pd.DataFrame, list[str]]:
        """A method that computes third order Sobol' indices."""
        if self._PCE_coeffs is {}:
            raise ValueError(
                "No polynomial coefficients available. Please run compute_sobol_indices() first.\n"
            )

        timesteps_solver = self.timesteps_solver
        timesteps_quadrature = np.linspace(
            timesteps_solver[0], timesteps_solver[-1], self._num_timesteps_quadrature
        )

        masks_third = []
        param_combinations = []
        for i in range(self.num_params):
            for j in range(i + 1, self.num_params):
                for k in range(j + 1, self.num_params):
                    mask_third = [
                        (
                            1
                            if key[i] != 0
                            and key[j] != 0
                            and key[k] != 0
                            and key.count(0) == (len(key) - 3)
                            else 0
                        )
                        for key in self._polynomial_dict.keys()
                    ]
                    masks_third.append(mask_third)
                    param_combinations.append(
                        self.param_names[i]
                        + " "
                        + self.param_names[j]
                        + " "
                        + self.param_names[k]
                    )
        masks_third = np.array(masks_third)

        # Compute variances for each parameter at each time step

        coeffs = self._PCE_coeffs[method]

        if method == "PCE":

            variance_over_time_third = np.zeros(
                (len(timesteps_quadrature), len(param_combinations)), dtype=np.double
            )
            total_variance_over_time = np.zeros(
                len(timesteps_quadrature), dtype=np.double
            )

            for m in range(len(timesteps_quadrature)):

                total_variance_over_time[m] = np.sum(coeffs[m][1:] ** 2)

                for i in range(len(param_combinations)):

                    variance_over_time_third[m, i] = np.sum(
                        coeffs[m] ** 2 * masks_third[i]
                    )

            # Compute the generalized Sobol indices
            td_sobol_indices_third = np.zeros(
                (len(timesteps_quadrature), len(param_combinations))
            )

            for m in range(len(timesteps_quadrature)):
                for i in range(len(param_combinations)):

                    if m == 0:

                        td_sobol_indices_third[m, i] = np.NaN

                    else:

                        h = timesteps_quadrature[1] - timesteps_quadrature[0]
                        weights = np.ones(m) * h
                        weights[0] = 0.5 * h
                        weights[-1] = 0.5 * h

                        denum = np.dot(
                            np.asfarray(total_variance_over_time[:m]), weights
                        )
                        td_sobol_indices_third[m, i] = (
                            np.dot(
                                np.asfarray(variance_over_time_third[:m, i]), weights
                            )
                            / denum
                        )
            sobol_indices_third = pd.DataFrame(
                td_sobol_indices_third[-1, :],
                columns=["third"],
                index=param_combinations,
            )
            td_sobol_indices_third = pd.DataFrame(
                td_sobol_indices_third,
                columns=param_combinations,
                index=timesteps_quadrature,
            )
            self.higher_order_sobol_indices["third_PCE"] = sobol_indices_third
            self.td_higher_order_sobol_indices["third_PCE"] = td_sobol_indices_third
            self._param_combinations_third_order = param_combinations

        elif method == "KL":

            sum_coeff_per_param_combination = np.zeros(
                (self._KL_truncation_level, len(param_combinations))
            )

            for i in range(len(param_combinations)):
                for j in range(self._KL_truncation_level):
                    sum_coeff_per_param_combination[j, i] = np.sum(
                        coeffs[j] ** 2 * masks_third[i]
                    )

            sum_eigenvalues = sum(self._sorted_eigenvalues)
            sobol_indices_third = np.zeros(len(param_combinations))

            for i in range(len(param_combinations)):
                sobol_indices_third[i] = (
                    sum(sum_coeff_per_param_combination[:, i]) / sum_eigenvalues
                )

            sobol_indices_third = pd.DataFrame(
                sobol_indices_third, columns=["third"], index=param_combinations
            )
            self.higher_order_sobol_indices["third_KL"] = sobol_indices_third
            self._param_combinations_third_order = param_combinations

        else:
            raise ValueError(
                f"Unknown method: {method}. Please choose from Karhunen-Loève ('KL') or Polynomial Chaos Expansion ('PCE').\n"
            )

        key = "third_" + method
        return (self.higher_order_sobol_indices[key], param_combinations)

    # TODO make evaluation of KL surrogate possible too
    def evaluate_surrogate_model(self, param: NDArray, method: str) -> NDArray:
        if method == "PCE":
            if self._polynomial_pointwise["PCE"] is None:
                raise ValueError(
                    "No PCE surrogate model available. Please run the PCE method first.\n"
                )
            result = [
                cp.call(self._polynomial_pointwise["PCE"][m], param)
                for m in range(self._num_timesteps_quadrature)
            ]
        elif method == "KL":
            if self._polynomial_pointwise["KL"] is None:
                raise ValueError(
                    "No KL surrogate model available. Please run the KL method first.\n"
                )
            result = [
                self._KL_mean[m]
                + sum(
                    [
                        cp.call(self._polynomial_pointwise["KL"][i], param)
                        * self._sorted_eigenvectors[m][i]
                        for i in range(self._KL_truncation_level)
                    ]
                )
                for m in range(self._num_timesteps_quadrature)
            ]
        else:
            raise ValueError(
                f"Unknown method: {method}. Please choose from Karhunen-Loève ('KL') or Polynomial Chaos Expansion ('PCE').\n"
            )
        return np.array(result)

    def plot(self, plot_option: str) -> None:
        """A method that plots the results of the time-dependent sensitivity analysis.
        Options:
            - 'sobol_indices': bar plot of the generalized Sobol' indices
            - 'time_dependent_sobol_indices': line plot of the time-dependent generalized Sobol' indices
            - 'higher_order_sobol_indices': bar plot of the second and third order Sobol' indices
            - 'simulator_output': line plot of the simulator output
            - 'sampled_parameters': histograms of the sampled parameters
            - 'covariance_matrix': heatmap of the covariance matrix
            - 'eigenvalue_spectrum': scatter plot of the eigenvalue spectrum
        """
        if plot_option == "sobol_indices":
            self._plot_sobol_indices()
        elif plot_option == "time_dependent_sobol_indices":
            self._plot_time_dependent_sobol_indices()
        elif plot_option == "higher_order_sobol_indices":
            self._plot_higher_order_sobol_indices()
        elif plot_option == "simulator_output":
            self._plot_simulator_output()
        elif plot_option == "sampled_parameters":
            self._plot_sampled_params()
        elif plot_option == "covariance_matrix":
            self._plot_covariance_matrix()
        elif plot_option == "eigenvalue_spectrum":
            self._plot_eigenvalue_spectrum()
        else:
            raise ValueError(
                f"Unknown plot option: {plot_option}. Please choose from 'sobol_indices', 'time_dependent_sobol_indices', 'higher_order_sobol_indices', 'simulator_output', 'sampled_parameters', 'covariance_matrix', or 'eigenvalue_spectrum'.\n"
            )

    def _plot_sobol_indices(self) -> None:
        """A method that plots the generalized Sobol' indices."""
        if self.sobol_indices == {}:
            raise ValueError(
                "No Sobol' indices available. Please run compute_sobol_indices() first.\n"
            )

        num_plots = len(self.sobol_indices)
        x = np.arange(self.num_params)

        if num_plots == 1:
            fig, ax = plt.subplots()
            for method, sobol_indices in self.sobol_indices.items():
                ax.bar(
                    x - 0.055, sobol_indices["first"], label="first order", width=0.1
                )
                ax.bar(
                    x + 0.055, sobol_indices["total"], label="total order", width=0.1
                )
                ax.set_xticks(x, self.param_names)
                ax.set_ylabel("Generalized Sobol' index")
                ax.set_title(f"{method} method")
                ax.legend()
        else:
            fig, ax = plt.subplots(1, num_plots, sharey=True)
            for index, (method, sobol_indices) in enumerate(self.sobol_indices.items()):
                ax[index].bar(
                    x - 0.055, sobol_indices["first"], label="first order", width=0.1
                )
                ax[index].bar(
                    x + 0.055, sobol_indices["total"], label="total order", width=0.1
                )
                ax[index].set_xticks(x, self.param_names)
                ax[0].set_ylabel("Generalized Sobol' index")
                ax[index].set_title(f"{method} method")
                ax[index].legend()
            plt.tight_layout()
            plt.show()

    def _plot_time_dependent_sobol_indices(self) -> None:
        """A method that plots the time-evolution of the generalized Sobol' indices."""
        timesteps_quadrature = np.linspace(
            self.timesteps_solver[0],
            self.timesteps_solver[-1],
            self._num_timesteps_quadrature,
        )

        if self.td_sobol_indices is {}:
            raise ValueError(
                "No time-dependent Sobol' indices available. Please run the PCE method first.\n"
            )

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

    # TODO plot second and third order sobol indices (depending on which was already computed)
    def _plot_higher_order_sobol_indices(self) -> None:
        plot_list = [
            key in self.higher_order_sobol_indices
            for key in ["second_KL", "second_PCE", "third_KL", "third_PCE"]
        ]
        num_plots = sum(plot_list)

        if num_plots == 0:
            raise ValueError(
                "No higher order Sobol' indices available. Please run compute_second_order_sobol_indices() and/or compute_third_order_sobol_indices() first.\n"
            )

        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        x_second = (
            np.arange(len(self._param_combinations_second_order))
            if self._param_combinations_second_order is not None
            else None
        )
        x_third = (
            np.arange(len(self._param_combinations_third_order))
            if self._param_combinations_third_order is not None
            else None
        )
        if "second_KL" in self.higher_order_sobol_indices:
            ax[0, 0].barh(
                x_second,
                self.higher_order_sobol_indices["second_KL"].to_numpy().reshape(-1),
                height=0.1,
            )
            ax[0, 0].set_xlabel("Generalized second order Sobol' index")
            ax[0, 0].set_yticks(x_second, self._param_combinations_second_order)
            ax[0, 0].set_title("Second order KL")
        else:
            ax[0, 0].remove()
        if "second_PCE" in self.higher_order_sobol_indices:
            ax[0, 1].barh(
                x_second,
                self.higher_order_sobol_indices["second_PCE"].to_numpy().reshape(-1),
                height=0.1,
            )
            ax[0, 1].set_xlabel("Generalized second order Sobol' index")
            ax[0, 1].set_yticks(x_second, self._param_combinations_second_order)
            ax[0, 1].set_title("Second order PCE")
        else:
            ax[0, 1].remove()
        if "third_KL" in self.higher_order_sobol_indices:
            ax[1, 0].barh(
                x_third,
                self.higher_order_sobol_indices["third_KL"].to_numpy().reshape(-1),
                height=0.1,
            )
            ax[1, 0].set_xlabel("Generalized third order Sobol' index")
            ax[1, 0].set_yticks(x_third, self._param_combinations_third_order)
            ax[1, 0].set_title("Third order KL")
        else:
            ax[1, 0].remove()
        if "third_PCE" in self.higher_order_sobol_indices:
            ax[1, 1].barh(
                x_third,
                self.higher_order_sobol_indices["third_PCE"].to_numpy().reshape(-1),
                height=0.1,
            )
            ax[1, 1].set_xlabel("Generalized third order Sobol' index")
            ax[1, 1].set_yticks(x_third, self._param_combinations_third_order)
            ax[1, 1].set_title("Third order PCE")
        else:
            ax[1, 1].remove()
        plt.tight_layout()
        plt.show()

    def _plot_simulator_output(self) -> None:
        """A method that plots the mean and standard deviation of the simulator output."""
        if self.outputs is None:
            raise ValueError("No simulator outputs available.")

        mean_output = self.outputs.mean(axis=0)
        std_output = self.outputs.std(axis=0)

        plt.figure()
        plt.plot(self.timesteps_solver, mean_output, label="Mean")
        plt.fill_between(
            self.timesteps_solver,
            mean_output - std_output,
            mean_output + std_output,
            alpha=0.3,
            label="Standard Deviation",
        )
        plt.xlabel("Time")
        plt.ylabel("Simulator Output")
        plt.legend()
        plt.show()

    def _plot_sampled_params(self) -> None:

        def plot_hist(ax, data, param_name, num_bins=50):
            sns.histplot(data, bins=num_bins, ax=ax)
            ax.set_xlabel(param_name)

        def plot_log_hist(ax, data, param_name, num_bins=50):
            counts, bins, _ = ax.hist(data, bins=num_bins)
            logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
            ax.cla()
            sns.histplot(data, bins=logbins, ax=ax, edgecolor="k", linewidth=0.5)
            ax.set_xscale("log")
            ax.set_xlabel(param_name)

        fig, axes = plt.subplots(self.num_params, 1, figsize=(6, 4 * self.num_params))
        for i, param_name in enumerate(self.param_names):
            if (
                self.distribution.dist_dict[param_name][0] == "loguniform"
                or self.distribution.dist_dict[param_name][0] == "lognormal"
            ):
                plot_log_hist(axes[i], self.params[param_name], param_name)
            else:
                plot_hist(axes[i], self.params[param_name], param_name)

    def _plot_covariance_matrix(self) -> None:
        """A method that plots the covariance matrix."""
        if self._covariance_matrix is None:
            raise ValueError(
                "No covariance matrix available. Please run the KL method first.\n"
            )
        plt.figure()
        sns.heatmap(self._covariance_matrix)
        plt.show()

    def _plot_eigenvalue_spectrum(self) -> None:
        """A method that plots the eigenvalue spectrum to check spectral decay."""
        if self._sorted_eigenvalues_normed is None:
            raise ValueError(
                "No eigenvalues available. Please run the KL method first.\n"
            )
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
