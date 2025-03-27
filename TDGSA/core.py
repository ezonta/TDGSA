#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import chaospy as cp
import multiprocessing
from joblib import Parallel, delayed
from sklearn import linear_model
from . import utils
from typing import Optional, Union, Dict
from numpy.typing import NDArray
from tqdm.autonotebook import tqdm
import logging

# for logging purposes
logger = logging.getLogger(__name__)
logging.basicConfig(filename="sensitivity.log", encoding="utf-8", level=logging.INFO)


class time_dependent_sensitivity_analysis:
    """A class that performs time-dependent sensitivity analysis using either the 'KL' or 'PC' approach.

    Attributes:
        simulator (utils.simulator): The simulator object used for running simulations.
        distribution (utils.distribution): The distribution object representing the parameter distributions.
        param_names (list[str]): The names of the parameters.
        params (Optional[pd.DataFrame]): The parameter samples used for analysis.
        outputs (Optional[pd.DataFrame]): The corresponding simulation outputs.
        num_samples (Optional[int]): The number of parameter samples.
        timesteps_solver (NDArray): The time steps used in the simulator.
        sobol_indices (Dict[str, pd.DataFrame]): The computed Sobol indices.
        td_sobol_indices (Dict[str, pd.DataFrame]): The computed time-dependent Sobol indices.
        higher_order_sobol_indices (Dict[str, pd.DataFrame]): The computed higher-order Sobol indices.
        td_higher_order_sobol_indices (Dict[str, pd.DataFrame]): The computed time-dependent higher-order Sobol indices.
        _num_timesteps_quadrature (Optional[int]): The number of quadrature nodes in time.
        _PC_option (Optional[str]): The option for Polynomial Chaos (PC) expansion analysis.
        _PC_quad_weights (Optional[NDArray]): The quadrature weights for PC analysis.
        _KL_truncation_level (Optional[int]): The truncation level for Karhunen-Loève (KL) expansion.
        _covariance_matrix (Optional[NDArray]): The covariance matrix used in KL analysis.
        _sorted_eigenvalues (Optional[NDArray]): The sorted eigenvalues from KL analysis.
        _sorted_eigenvectors (Optional[NDArray]): The sorted eigenvectors from KL analysis.
        _sorted_eigenvalues_normed (Optional[NDArray]): The normalized sorted eigenvalues from KL analysis.
        _r_Nkl (Optional[NDArray]): The variance ratio from KL analysis.
        _KL_mean (Optional[NDArray]): The mean value used in KL analysis.
        _polynomial_dict (Optional[Dict]): The polynomial dictionary used in PC analysis.
        _PC_coeffs (Dict[str, list[NDArray]]): The PC coefficients from PC analysis.
        _polynomial_pointwise (Dict[str, Optional[list[cp.ndpoly]]]): The pointwise polynomials from PC analysis.
        _param_combinations_second_order (Optional[list[str]]): The second-order parameter combinations.
        _param_combinations_third_order (Optional[list[str]]): The third-order parameter combinations.

    Methods:
        __init__(self, simulator: utils.simulator, distribution: utils.distribution, data: Optional[tuple[pd.DataFrame, pd.DataFrame]] = None, **kwargs: Union[int, str, NDArray, None]) -> None:
            Initializes the time_dependent_sensitivity_analysis object.
        sample_params_and_run_simulator(self, num_samples: int, sampling_method: str = "random", **kwargs: Union[int, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
            Generates parameter samples and runs the simulator.
        compute_sobol_indices(self, method: str, **kwargs) -> pd.DataFrame:
            Runs the time-dependent sensitivity analysis and returns the generalized Sobol indices.
        _KL_analysis(self, params: NDArray, output: NDArray, **kwargs: Union[int, str]) -> None:
            Performs time-dependent sensitivity analysis using Karhunen-Loève (KL) expansion.
        _PC_analysis(self, params: NDArray, output: NDArray, **kwargs: Union[int, str]) -> None:
            Performs time-dependent sensitivity analysis using Polynomial Chaos Expansion (PC).
    """

    # public
    simulator: utils.simulator
    distribution: utils.distribution
    param_names: list[str]

    params: Optional[pd.DataFrame]
    outputs: Optional[pd.DataFrame]

    num_samples: Optional[int]
    timesteps_solver: NDArray

    failed_simulations: Optional[pd.DataFrame]

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
        **kwargs: Union[int, str, NDArray, None],
    ) -> None:

        self.simulator = simulator
        self.distribution = distribution

        if data is not None:
            self.params = data[0]
            self.outputs = data[1]
            self._PCE_option = kwargs.get("PCE_option", "regression")
            self._PCE_quad_weights = kwargs.get("PCE_quad_weights", None)
            self.num_samples = self.params.shape[0]
        else:
            self.params = None
            self.outputs = None
            self._PCE_option = None
            self._PCE_quad_weights = None
            self.num_samples = None

        self.param_names = self.distribution.param_names
        self.num_params = self.distribution.dim
        self.timesteps_solver = self.simulator.time

        self.failed_simulations = None

        self.sobol_indices = {}
        self.td_sobol_indices = {}
        self.higher_order_sobol_indices = {}
        self.td_higher_order_sobol_indices = {}

        self._num_timesteps_quadrature = None

        self._KL_truncation_level = None
        self._covariance_matrix = None
        self._sorted_eigenvalues = None
        self._sorted_eigenvectors = None
        self._sorted_eigenvalues_normed = None
        self._r_Nkl = None
        self._KL_mean = None

        self._polynomial_dict = None
        self._PCE_coeffs = {}
        self._polynomial_pointwise = {"PC": None, "KL": None}

        self._param_combinations_second_order = None
        self._param_combinations_third_order = None

    def sample_params_and_run_simulator(
        self,
        num_samples: int,
        sampling_method: str = "random",
        **kwargs: Union[int, str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Samples parameters and runs the simulator.

        Options:
            - num_samples: number of parameter samples
            - sampling_method: 'random', 'quasirandom', 'quadrature'

        kwargs:
            - quasirandom_method: 'halton', 'sobol', 'latin_hypercube' (default is 'halton')
            - quadrature_method: 'clenshaw_curtis', 'gaussian', 'legendre' (default is 'clenshaw_curtis')
            - quadrature_order: order of quadrature rule (default is 4)
        """

        ## sampling of parameters
        logger.info(f"Sampling {num_samples} parameters ...")

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
                error = ValueError(
                    f"Unknown quasirandom method: {quasirandom_rule}. Please choose from 'halton', 'sobol', or 'latin_hypercube'."
                )
                logger.error(error)
                raise error
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
                error = ValueError(
                    f"Unknown quadrature method: {quadrature_rule}. Please choose from 'clenshaw_curtis', 'gaussian', or 'legendre'."
                )
                logger.error(error)
                raise error
            samples, weights = cp.generate_quadrature(
                quadrature_order, self.distribution.dist, rule=quadrature_rule
            )
            self._PCE_option = "quadrature"
            self._PCE_quad_weights = weights

        else:
            error = ValueError(
                f"Unknown sampling method: {sampling_method}. Please choose from 'random', 'quasirandom', or 'quadrature'."
            )
            logger.error(error)
            raise error

        ## run simulator and get outputs
        logger.info(f"Finished sampling {num_samples} parameters.")
        logger.info(f"Running simulator ...")

        outputs = self.simulator.run(samples)

        ## check if outputs match in length and shorten them if necessary
        min_length = min([len(out) for out in outputs])
        max_length = max([len(out) for out in outputs])
        if min_length != max_length:
            logger.warning(
                f"Outputs have different lengths. Shortening them to the minimum length of {min_length}."
            )
            outputs = [out[:min_length] for out in outputs]
            self.timesteps_solver = self.timesteps_solver[:min_length]

        ## convert from numpy array to pandas dataframe
        params = pd.DataFrame(samples, columns=self.param_names)
        outputs = pd.DataFrame(outputs, columns=self.timesteps_solver)

        ## check if there are NaN outputs and remove them
        nan_index = outputs.loc[pd.isna(outputs.iloc[:, 0]), :].index
        if len(nan_index) > 0:
            logger.warning(
                f"Found NaN outputs for {len(nan_index)} samples. Removing them ..."
            )
            failed_simulations = params.loc[nan_index, :]
            params.drop(nan_index, inplace=True)
            outputs.drop(nan_index, inplace=True)
            num_samples = num_samples - len(nan_index)
            self.failed_simulations = failed_simulations
            logger.info(
                f"Removed {len(nan_index)} samples with NaN outputs. They can be accessed via the 'failed_simulations' attribute."
            )

        self.params = params
        self.outputs = outputs
        self.num_samples = num_samples

        return params, outputs

    def compute_sobol_indices(self, method: str, **kwargs) -> pd.DataFrame:
        """A method that runs the time-dependent sensitivity analysis and returns the generalized sobol indices.

        Options:
            - method: 'KL', 'PC'

        kwargs:
            - num_timesteps_quadrature: number of quadrature nodes in time (default is 100)
            - KL_truncation_level: truncation level for the Karhunen-Loève expansion (default is 8)
            - PCE_order: order of the Polynomial Chaos Expansion (default is 4)
            - cross_truncation: cross truncation parameter for the PC expansion (default is 1.0)
            - regression_model: 'OLS' or 'LARS' (default is 'OLS')
        """
        if self.outputs is None:
            error = ValueError(
                "No data available. Please run sample_params_and_run_simulator() first."
            )
            logger.error(error)
            raise error

        if method == "KL":
            self._KL_analysis(self.params.to_numpy(), self.outputs.to_numpy(), **kwargs)
        elif method == "PC":
            self._PC_analysis(self.params.to_numpy(), self.outputs.to_numpy(), **kwargs)
        else:
            error = ValueError(
                f"Unknown method: {method}. Please choose from Karhunen-Loève ('KL') or Polynomial Chaos ('PC')."
            )
            logger.error(error)
            raise error

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
            np.sum(sorted_eigenvalues[:i]) / np.sum(sorted_eigenvalues)
            for i in range(len(sorted_eigenvalues))
        ]
        if self._r_Nkl[-1] < 0.90:
            logger.warning(
                "The variance ratio is less than 90%. Consider increasing the truncation level.\nYou can view the eigenvalue spectrum by calling the plot() method.\n"
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

        logger.info(f"Generating PC expansion of order {PCE_order} ...")

        cross_truncation = kwargs.get("cross_truncation", 1.0)

        expansion, norms = cp.generate_expansion(
            PCE_order,
            joint_dist,
            normed=True,
            graded=False,
            retall=True,
            cross_truncation=cross_truncation,
        )

        logger.info("Fitting surrogate models ...")

        num_cores = multiprocessing.cpu_count()
        if PCE_option == "regression":
            regression_model = kwargs.get("regression_model", "OLS")
            if regression_model == "OLS":
                model = linear_model.LinearRegression(fit_intercept=False)
            elif regression_model == "LARS":
                model = linear_model.Lars(
                    fit_intercept=False, n_nonzero_coefs=len(expansion)
                )
            else:
                error = ValueError(
                    f"Unknown regression model: {regression_model}. Please choose from 'OLS' or 'LARS'."
                )
                logger.error(error)
                raise error
            surrogate_models = Parallel(n_jobs=num_cores)(
                delayed(cp.fit_regression)(
                    expansion,
                    params.T,
                    KL_modes[:, i],
                    retall=1,
                    model=model,
                )
                for i in tqdm(range(N_kl))
            )
        elif PCE_option == "quadrature":
            if self._PCE_quad_weights is None:
                error = ValueError(
                    "No quadrature weights available. Please run sample_params_and_run_simulator() with the quadrature option first."
                )
                logger.error(error)
                raise error
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
        surrogate_model_poly_dict = expansion.todict()

        # Compute the generalized Sobol indices
        self._polynomial_dict = surrogate_model_poly_dict
        self._PCE_coeffs["KL"] = surrogate_model_coeffs
        self._polynomial_pointwise["KL"] = polynomial_pointwise
        sum_coeff_per_param_total = np.zeros((N_kl, self.num_params))
        sum_coeff_per_param_first = np.zeros((N_kl, self.num_params))

        masks_total = []
        masks_first = []
        for i in range(self.num_params):
            mask_total = [
                1 if key[i] != 0 else 0 for key in surrogate_model_poly_dict.keys()
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

        # check if sum of eigenvalues and squared sum of PC coefficients agree on total variance
        # add a warning if the sum of eigenvalues and sum of squared PC coefficients do not agree and use the sum of squared PC coefficients as total variance
        sum_eigenvalues = np.sum(sorted_eigenvalues[:N_kl])
        sum_coefficients = np.sum(
            [np.sum(surrogate_model_coeffs[j][1:] ** 2) for j in range(N_kl)]
        )
        rel_error_variance = (
            np.abs(sum_eigenvalues - sum_coefficients) / sum_eigenvalues
        )

        if rel_error_variance > 0.1:
            denum = sum_coefficients
            logger.warning(
                f"The relative error between the sum of eigenvalues and the sum of squared PC coefficients is larger than 10%: {rel_error_variance:.2f}. The sum of squared PC coefficients will be used as total variance to compute the Sobol' indices."
            )
        else:
            denum = sum_eigenvalues

        sobol_indices_total = np.zeros(self.num_params)
        sobol_indices_first = np.zeros(self.num_params)

        for i in range(self.num_params):
            sobol_indices_total[i] = np.sum(sum_coeff_per_param_total[:, i]) / denum
            sobol_indices_first[i] = np.sum(sum_coeff_per_param_first[:, i]) / denum
        sobol_indices = np.zeros((self.num_params, 2))
        sobol_indices[:, 0] = sobol_indices_first
        sobol_indices[:, 1] = sobol_indices_total
        sobol_indices = pd.DataFrame(
            sobol_indices, columns=["first", "total"], index=self.param_names
        )
        self.sobol_indices["KL"] = sobol_indices

    def _PC_analysis(
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

        logger.info(f"Generating PC expansion of order {PCE_order} ...")

        cross_truncation = kwargs.get("cross_truncation", 1.0)

        expansion, norms = cp.generate_expansion(
            PCE_order,
            joint_dist,
            normed=True,
            graded=False,
            retall=True,
            cross_truncation=cross_truncation,
        )

        logger.info("Fitting surrogate models ...")

        num_cores = multiprocessing.cpu_count()
        if PCE_option == "regression":
            regression_model = kwargs.get("regression_model", "OLS")
            if regression_model == "OLS":
                model = linear_model.LinearRegression(fit_intercept=False)
            elif regression_model == "LARS":
                model = linear_model.Lars(
                    fit_intercept=False, n_nonzero_coefs=len(expansion)
                )
            else:
                error = ValueError(
                    f"Unknown regression model: {regression_model}. Please choose from 'OLS' or 'LARS'."
                )
                logger.error(error)
                raise error
            surrogate_models_pointwise = Parallel(n_jobs=num_cores)(
                delayed(cp.fit_regression)(
                    expansion,
                    params.T,
                    outputs_quadrature[:, m],
                    retall=1,
                    model=model,
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
        coeff_pointwise = [
            surrogate_models_pointwise[i][1] for i in range(len(timesteps_quadrature))
        ]
        polynomial_pointwise = [
            surrogate_models_pointwise[i][0] for i in range(len(timesteps_quadrature))
        ]
        # save for later computation of second and third order sobol indices and PC surrogate evaluation
        polynomial_pointwise_dict = expansion.todict()
        self._polynomial_dict = polynomial_pointwise_dict
        self._PCE_coeffs["PC"] = coeff_pointwise
        self._polynomial_pointwise["PC"] = polynomial_pointwise

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
        self.sobol_indices["PC"] = sobol_indices

        td_sobol_indices_total = pd.DataFrame(
            td_sobol_indices_total, columns=self.param_names, index=timesteps_quadrature
        )
        td_sobol_indices_first = pd.DataFrame(
            td_sobol_indices_first, columns=self.param_names, index=timesteps_quadrature
        )
        self.td_sobol_indices["first"] = td_sobol_indices_first
        self.td_sobol_indices["total"] = td_sobol_indices_total

    def compute_second_order_sobol_indices(
        self, method: str
    ) -> tuple[pd.DataFrame, list[str]]:
        """A method that computes second order Sobol' indices.

        Options:
            - method: 'KL', 'PC'
        """

        if self._PCE_coeffs == {}:
            error = ValueError(
                "No polynomial coefficients available. Please run compute_sobol_indices() first."
            )
            logger.error(error)
            raise error
        elif self._num_timesteps_quadrature is None:
            error = ValueError(
                "No quadrature nodes available. Please run compute_sobol_indices() first."
            )
            logger.error(error)
            raise error

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

        if method == "PC":

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
            self.higher_order_sobol_indices["second_PC"] = sobol_indices_second
            self.td_higher_order_sobol_indices["second_PC"] = td_sobol_indices_second
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
            error = ValueError(
                f"Unknown method: {method}. Please choose from Karhunen-Loève ('KL') or Polynomial Chaos Expansion ('PC')."
            )
            logger.error(error)
            raise error

        key = "second_" + method
        return (self.higher_order_sobol_indices[key], param_combinations)

    def compute_third_order_sobol_indices(
        self, method: str
    ) -> tuple[pd.DataFrame, list[str]]:
        """A method that computes third order Sobol' indices.

        Options:
            - method: 'KL', 'PC'
        """

        if self._PCE_coeffs == {}:
            error = ValueError(
                "No polynomial coefficients available. Please run compute_sobol_indices() first."
            )
            logger.error(error)
            raise error
        elif self._num_timesteps_quadrature is None:
            error = ValueError(
                "No quadrature nodes available. Please run compute_sobol_indices() first."
            )
            logger.error(error)
            raise error

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

        if method == "PC":

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
            self.higher_order_sobol_indices["third_PC"] = sobol_indices_third
            self.td_higher_order_sobol_indices["third_PC"] = td_sobol_indices_third
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
            error = ValueError(
                f"Unknown method: {method}. Please choose from Karhunen-Loève ('KL') or Polynomial Chaos Expansion ('PC')."
            )
            logger.error(error)
            raise error

        key = "third_" + method
        return (self.higher_order_sobol_indices[key], param_combinations)

    def evaluate_surrogate_model(self, param: NDArray, method: str) -> NDArray:
        if method == "PC":
            if self._polynomial_pointwise["PC"] is None:
                error = ValueError(
                    "No PC surrogate model available. Please run the PC method first."
                )
                logger.error(error)
                raise error
            result = [
                cp.call(self._polynomial_pointwise["PC"][m], param)
                for m in range(self._num_timesteps_quadrature)
            ]
        elif method == "KL":
            if self._polynomial_pointwise["KL"] is None:
                error = ValueError(
                    "No KL surrogate model available. Please run the KL method first."
                )
                logger.error(error)
                raise error
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
            error = ValueError(
                f"Unknown method: {method}. Please choose from Karhunen-Loève ('KL') or Polynomial Chaos Expansion ('PC')."
            )
            logger.error(error)
            raise error
        return np.array(result)

    def plot(self, plot_option: str) -> None:
        """A method that plots the results of the time-dependent sensitivity analysis.

        Options for plot_option:
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
            error = ValueError(
                f"Unknown plot option: {plot_option}. Please choose from 'sobol_indices', 'time_dependent_sobol_indices', 'higher_order_sobol_indices', 'simulator_output', 'sampled_parameters', 'covariance_matrix', or 'eigenvalue_spectrum'."
            )
            logger.error(error)
            raise error

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

        if self.td_sobol_indices == {}:
            raise ValueError(
                "No time-dependent Sobol' indices available. Please run the PC method first.\n"
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

    def _plot_higher_order_sobol_indices(self) -> None:
        """A method that plots the second and third order Sobol' indices."""

        plot_list = [
            key in self.higher_order_sobol_indices
            for key in ["second_KL", "second_PC", "third_KL", "third_PC"]
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
        if "second_PC" in self.higher_order_sobol_indices:
            ax[0, 1].barh(
                x_second,
                self.higher_order_sobol_indices["second_PC"].to_numpy().reshape(-1),
                height=0.1,
            )
            ax[0, 1].set_xlabel("Generalized second order Sobol' index")
            ax[0, 1].set_yticks(x_second, self._param_combinations_second_order)
            ax[0, 1].set_title("Second order PC")
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
        if "third_PC" in self.higher_order_sobol_indices:
            ax[1, 1].barh(
                x_third,
                self.higher_order_sobol_indices["third_PC"].to_numpy().reshape(-1),
                height=0.1,
            )
            ax[1, 1].set_xlabel("Generalized third order Sobol' index")
            ax[1, 1].set_yticks(x_third, self._param_combinations_third_order)
            ax[1, 1].set_title("Third order PC")
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
        """A method that plots histograms of the sampled parameters."""

        def plot_hist(ax, data, param_name, num_bins=50):
            sns.histplot(data, bins=num_bins, ax=ax, edgecolor="k", linewidth=0.5)
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
        """A method that plots the covariance matrix of the KL method."""

        if self._covariance_matrix is None:
            raise ValueError(
                "No covariance matrix available. Please run the KL method first.\n"
            )
        plt.figure()
        sns.heatmap(self._covariance_matrix)
        plt.show()

    def _plot_eigenvalue_spectrum(self) -> None:
        """A method that plots the eigenvalue spectrum to check spectral decay in the KL method."""

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
