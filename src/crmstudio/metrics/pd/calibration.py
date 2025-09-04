"""
Calibration metrics for probability of default (PD) models.

This module provides metrics for assessing the calibration quality of PD models,
including Hosmer-Lemeshow test, Brier score, Expected Calibration Error (ECE),
calibration curves, heterogeneity testing, and concentration metrics.

The metrics in this module assess various aspects of model calibration:
1. Overall calibration quality (Hosmer-Lemeshow, Brier score)
2. Granular calibration assessment (ECE, calibration curves)
3. Statistical tests for calibration (Binomial test, Normal test, Jeffreys test)
4. Heterogeneity testing (calibration consistency across segments/subgroups)
5. Concentration analysis (Herfindahl Index for rating grades)

Based on EBA Supervisory handbook on the validation of IRB rating systems:
https://www.eba.europa.eu/sites/default/files/document_library/Publications/Reports/2023/1061495/Supervisory%20handbook%20on%20the%20validation%20of%20IRB%20rating%20systems%20revised.pdf

TODO:
We need an overtime measure, all our metrics are point in time, looking at a specific snapshot data.
We also need to check whether DRs observed in the long term cross any of the CI lines.
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

from ...core.base import BaseMetric
from ...core.data_classes import MetricResult

class HosmerLemeshow(BaseMetric):
    """
    Hosmer-Lemeshow goodness-of-fit test for binary classification models.
    
    This test evaluates whether the observed event rates match expected event rates
    in subgroups of the model population. A small p-value indicates poor calibration.
    
    Hypothesis test:
    - H0: The model is well-calibrated (observed rates match expected rates)
    - H1: The model is not well-calibrated (observed rates differ from expected rates)
    
    References:
    ----------
    - Regulatory: Basel Committee on Banking Supervision (2005). "Studies on the Validation 
      of Internal Rating Systems," Working Paper No. 14.
    - Academic: Hosmer, D.W., Lemeshow, S. (2000). "Applied Logistic Regression," 
      2nd Edition, John Wiley & Sons.
    - Regulatory: European Banking Authority (2017). "Guidelines on PD estimation, 
      LGD estimation and the treatment of defaulted exposures," EBA/GL/2017/16.
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="calibration", config=config, config_path=config_path, **kwargs)
        
    def _compute_raw(self, y_true=None, y_pred=None, ratings=None, **kwargs):
        """
        Calculate the Hosmer-Lemeshow statistic and p-value.
        
        Parameters
        ----------
        y_true : array-like
            Binary target values (0,1)
        y_pred : array-like
            Predicted probabilities
        ratings : array-like, optional
            Rating grades or bucket assignments for each observation.
            If provided, calculations will be performed at the rating level
            instead of using equal-sized bins.
            
        Returns
        -------
        MetricResult
            Containing the test statistic, p-value and group details
        """
        n_bins = self._get_param("n_bins", default=10)
        
        # Convert inputs to numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Check if ratings are provided
        use_ratings = ratings is not None
        
        if use_ratings:
            ratings = np.asarray(ratings)
            unique_ratings = np.unique(ratings)
            n_ratings = len(unique_ratings)
            
            # Sort ratings from worst to best (assuming higher PD means worse rating)
            rating_avg_pd = {r: np.mean(y_pred[ratings == r]) for r in unique_ratings}
            sorted_ratings = sorted(unique_ratings, key=lambda r: -rating_avg_pd[r])
            
            # Use the number of ratings as the number of bins
            n_bins = n_ratings
            
            # Initialize arrays for results based on ratings
            observed = np.zeros(n_bins)
            expected = np.zeros(n_bins)
            bin_counts = np.zeros(n_bins)
            bin_probs = np.zeros(n_bins)
            bin_names = []
            
            # Calculate observed and expected counts for each rating
            for i, rating in enumerate(sorted_ratings):
                mask = ratings == rating
                bin_y_true = y_true[mask]
                bin_y_pred = y_pred[mask]
                
                bin_counts[i] = len(bin_y_true)
                observed[i] = np.sum(bin_y_true)
                expected[i] = np.sum(bin_y_pred)
                bin_probs[i] = np.mean(bin_y_pred) if len(bin_y_pred) > 0 else 0
                bin_names.append(str(rating))
                
        else:
            # Sort by predicted probabilities for standard binning
            indices = np.argsort(y_pred)
            y_true_sorted = y_true[indices]
            y_pred_sorted = y_pred[indices]
            
            # Create equal-sized bins (as close as possible)
            n_samples = len(y_true)
            bin_size = n_samples // n_bins
            remainder = n_samples % n_bins
            
            # Calculate bin boundaries ensuring all observations are included
            bin_boundaries = [0]
            for i in range(n_bins):
                # Add one extra to the first 'remainder' bins to distribute remainder
                current_bin_size = bin_size + (1 if i < remainder else 0)
                bin_boundaries.append(bin_boundaries[-1] + current_bin_size)
        
            # Initialize arrays for results
            observed = np.zeros(n_bins)
            expected = np.zeros(n_bins)
            bin_counts = np.zeros(n_bins)
            bin_probs = np.zeros(n_bins)
            bin_names = [f"Bin {i+1}" for i in range(n_bins)]
            
            # Calculate observed and expected counts for each bin
            for i in range(n_bins):
                start_idx = bin_boundaries[i]
                end_idx = bin_boundaries[i+1]
                
                bin_y_true = y_true_sorted[start_idx:end_idx]
                bin_y_pred = y_pred_sorted[start_idx:end_idx]
                
                bin_counts[i] = len(bin_y_true)
                observed[i] = np.sum(bin_y_true)
                expected[i] = np.sum(bin_y_pred)
                bin_probs[i] = np.mean(bin_y_pred) if len(bin_y_pred) > 0 else 0
        
        # Calculate Hosmer-Lemeshow statistic
        with np.errstate(divide='ignore', invalid='ignore'):
            non_zero_expected = expected > 0
            non_zero_expected_complement = (bin_counts - expected) > 0
            
            terms = np.zeros(n_bins)
            
            # Handle bins with non-zero expected defaults
            if np.any(non_zero_expected):
                terms[non_zero_expected] = (observed[non_zero_expected] - expected[non_zero_expected])**2 / expected[non_zero_expected]
            
            # Handle bins with non-zero expected non-defaults
            if np.any(non_zero_expected_complement):
                terms[non_zero_expected_complement] += ((bin_counts[non_zero_expected_complement] - observed[non_zero_expected_complement]) - 
                                                     (bin_counts[non_zero_expected_complement] - expected[non_zero_expected_complement]))**2 / \
                                                     (bin_counts[non_zero_expected_complement] - expected[non_zero_expected_complement])
        
        hl_statistic = np.sum(terms)
        
        # Calculate p-value (chi-squared with n_bins-2 degrees of freedom)
        # We use n_bins-2 because we're estimating 2 parameters (in logistic regression)
        p_value = 1 - stats.chi2.cdf(hl_statistic, n_bins - 2)
        
        # Check if the test passes the significance level
        alpha = self._get_param("alpha", default=0.05)
        passed = p_value > alpha
        
        # Create bin details for the figure
        bin_details = []
        for i in range(n_bins):
            bin_details.append({
                'bin': i + 1,
                'name': bin_names[i],
                'n_obs': int(bin_counts[i]),
                'observed_defaults': int(observed[i]),
                'expected_defaults': float(expected[i]),
                'observed_rate': float(observed[i] / bin_counts[i]) if bin_counts[i] > 0 else 0,
                'expected_rate': float(bin_probs[i])
            })
        
        bin_df = pd.DataFrame(bin_details)
        
        # Title suffix
        bin_type = "Rating Grades" if use_ratings else "Probability Deciles"
        
        return MetricResult(
            name=self.__class__.__name__,
            value=p_value,
            threshold=alpha,
            passed=passed,
            details={
                'test_statistic': float(hl_statistic),
                'degrees_of_freedom': n_bins - 2,
                'n_bins': n_bins,
                'alpha': alpha,
                'bin_details': bin_details,
                'use_ratings': use_ratings
            },
            figure_data={
                'x': list(range(1, n_bins + 1)),  # Bin numbers (1-indexed)
                'y': bin_df['observed_rate'].tolist(),  # Observed default rates
                'x_ref': list(range(1, n_bins + 1)),  # Same bins for expected
                'y_ref': bin_df['expected_rate'].tolist(),  # Expected default rates
                'bin_labels': bin_names,  # Bin names/labels for x-axis
                'actual_label': 'Observed Default Rate',
                'ref_label': 'Expected Default Rate',
                'title': f'Hosmer-Lemeshow Test by {bin_type} (p={p_value:.4f}, {"Passed" if passed else "Failed"})',
                'xlabel': bin_type,
                'ylabel': 'Default Rate',
                'annotations': [
                    f'H-L Statistic: {hl_statistic:.2f}',
                    f'p-value: {p_value:.4f}',
                    f'Significance: {alpha}',
                ],
            }
        )


class CalibrationCurve(BaseMetric):
    """
    Calculate and plot the calibration curve (reliability diagram).
    
    This metric compares the predicted probabilities to the observed 
    frequencies. A perfectly calibrated model will have points along
    the diagonal line.
    
    The metric can operate in two modes:
    1. Standard mode: Using equal-width bins (default)
    2. Rating mode: Using provided rating grades
    
    When ratings are provided, the calibration is evaluated at the rating level,
    which is more relevant for credit risk models where observations are often
    grouped into discrete rating grades.
    
    References:
    ----------
    - Regulatory: Oesterreichische Nationalbank (OeNB) (2004). "Rating Models and Validation," 
      Guidelines on Credit Risk Management. 
      https://www.oenb.at/dam/jcr:1db13877-21a0-40f8-b46c-d8448f162794/rating_models_tcm16-22933.pdf
    - Academic: DeGroot, M.H., Fienberg, S.E. (1983). "The Comparison and Evaluation of Forecasters," 
      Journal of the Royal Statistical Society: Series D, 32(1), 12-22.
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="calibration", config=config, config_path=config_path, **kwargs)
        
    def _compute_raw(self, y_true=None, y_pred=None, ratings=None, **kwargs):
        n_bins = self._get_param("n_bins", default=10)
        strategy = self._get_param("strategy", default="uniform")
        
        # Check if ratings are provided
        use_ratings = ratings is not None
        
        if use_ratings:
            # Convert inputs to numpy arrays
            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)
            ratings = np.asarray(ratings)
            
            # Get unique ratings and sort by average PD
            unique_ratings = np.unique(ratings)
            n_ratings = len(unique_ratings)
            rating_avg_pd = {r: np.mean(y_pred[ratings == r]) for r in unique_ratings}
            sorted_ratings = sorted(unique_ratings, key=lambda r: rating_avg_pd[r])
            
            # Calculate observed and predicted probabilities for each rating
            prob_true = []  # Observed frequencies
            prob_pred = []  # Predicted probabilities
            rating_counts = []  # Number of observations in each rating
            rating_names = []   # Rating names/labels
            
            for rating in sorted_ratings:
                mask = ratings == rating
                if np.sum(mask) > 0:  # Skip empty ratings
                    prob_true.append(np.mean(y_true[mask]))
                    prob_pred.append(np.mean(y_pred[mask]))
                    rating_counts.append(np.sum(mask))
                    rating_names.append(str(rating))
            
            # Convert lists to numpy arrays
            prob_true = np.array(prob_true)
            prob_pred = np.array(prob_pred)
            rating_counts = np.array(rating_counts)
            
            # Compute calibration error metrics
            calibration_error = np.mean(np.abs(prob_true - prob_pred))
            
            # Calculate total observations and defaults
            n_obs = len(y_true)
            n_defaults = int(np.sum(y_true))
            
            # Prepare figure data specifically for ratings
            figure_data = {
                'x': prob_pred.tolist(),  # Predicted probabilities
                'y': prob_true.tolist(),  # Observed frequencies
                'x_ref': [0, 1],  # Diagonal reference line x-coords
                'y_ref': [0, 1],  # Diagonal reference line y-coords
                'actual_label': 'Observed Default Rate',
                'ref_label': 'Perfect Calibration',
                'title': f'Calibration Curve by Rating Grade (Error = {calibration_error:.4f})',
                'xlabel': 'Mean Predicted Probability',
                'ylabel': 'Observed Default Rate',
                'weights': rating_counts.tolist(),  # For bubble size in plot
                'rating_names': rating_names,  # For annotations in plot
                'plot_type': 'bubble',  # Indicate this should be a bubble plot
                'annotations': [
                    f'Calibration Error: {calibration_error:.4f}',
                    f'Samples: {n_obs}',
                    f'Default Rate: {n_defaults/n_obs:.2%}',
                ],
            }
            
            return MetricResult(
                name=self.__class__.__name__,
                value=calibration_error,
                details={
                    'use_ratings': True,
                    'n_ratings': n_ratings,
                    'n_obs': n_obs,
                    'n_defaults': n_defaults,
                    'rating_counts': rating_counts.tolist(),
                },
                figure_data=figure_data
            )
        else:
            # Standard calibration curve using sklearn's function
            # Compute calibration curve
            prob_true, prob_pred = calibration_curve(
                y_true=y_true, y_prob=y_pred, n_bins=n_bins, strategy=strategy
            )
        
        # Compute calibration error metrics
        # Mean absolute error between predicted and actual probabilities
        calibration_error = np.mean(np.abs(prob_true - prob_pred))
        
        # Calculate bin populations
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred, bin_edges[1:-1])
        bin_counts = np.bincount(bin_indices, minlength=n_bins)
        
        # Calculate total observations and defaults
        n_obs = len(y_true)
        n_defaults = int(np.sum(y_true))
        
        return MetricResult(
            name=self.__class__.__name__,
            value=calibration_error,
            details={
                'n_bins': n_bins,
                'strategy': strategy,
                'n_obs': n_obs,
                'n_defaults': n_defaults,
                'bin_counts': bin_counts.tolist(),
            },
            figure_data={
                'x': prob_pred.tolist(),  # Predicted probabilities
                'y': prob_true.tolist(),  # Observed frequencies
                'x_ref': [0, 1],  # Diagonal reference line x-coords
                'y_ref': [0, 1],  # Diagonal reference line y-coords
                'actual_label': 'Fraction of Positives',
                'ref_label': 'Perfectly Calibrated',
                'title': f'Calibration Curve (Error = {calibration_error:.4f})',
                'xlabel': 'Mean Predicted Probability',
                'ylabel': 'Observed Frequency',
                'bin_counts': bin_counts.tolist(),
                'annotations': [
                    f'Calibration Error: {calibration_error:.4f}',
                    f'Samples: {n_obs}',
                    f'Default Rate: {n_defaults/n_obs:.2%}',
                ],
            }
        )


class ExpectedCalibrationError(BaseMetric):
    """
    Calculate the Expected Calibration Error (ECE).

    ECE is a weighted average of the absolute difference between
    predicted probabilities and observed frequencies, where weights
    are determined by the number of samples in each bin.

    In credit risk modeling, this metric assesses how well a model's predicted
    probabilities align with actual default rates. Unlike formal hypothesis tests,
    ECE does not have explicit H0/H1 formulations, but rather measures calibration
    quality directly, with lower values indicating better calibration.

    The metric can operate in two modes:
    1. Standard mode: Using equal-width probability bins (default)
    2. Rating mode: Using provided rating grades

    When ratings are provided, the calibration error is evaluated at the rating level,
    which is more relevant for credit risk models where observations are often
    grouped into discrete rating grades.

    References:
    ----------
    - Academic: Naeini, M.P., Cooper, G., Hauskrecht, M. (2015). "Obtaining Well Calibrated Probabilities Using Bayesian Binning," 
      Proceedings of the AAAI Conference on Artificial Intelligence, 29(1).
    - Academic: Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q. (2017). "On Calibration of Modern Neural Networks," 
      Proceedings of the 34th International Conference on Machine Learning (ICML), PMLR 70:1321-1330.
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="calibration", config=config, config_path=config_path, **kwargs)
    
    def _compute_raw(self, y_true=None, y_pred=None, ratings=None, **kwargs):
        """
        Calculate the Expected Calibration Error.
        
        Parameters
        ----------
        y_true : array-like
            Binary target values (0,1)
        y_pred : array-like
            Predicted probabilities
        ratings : array-like, optional
            Rating grades or bucket assignments for each observation.
            If provided, calculations will be performed at the rating level
            instead of using equal-sized bins.
            
        Returns
        -------
        MetricResult
            Containing the ECE value and bin details
        """
        n_bins = self._get_param("n_bins", default=10)
        
        # Convert inputs to numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Check if ratings are provided
        use_ratings = ratings is not None
        
        if use_ratings:
            ratings = np.asarray(ratings)
            unique_ratings = np.unique(ratings)
            n_ratings = len(unique_ratings)
            
            # Sort ratings by average PD (ascending)
            rating_avg_pd = {r: np.mean(y_pred[ratings == r]) for r in unique_ratings}
            sorted_ratings = sorted(unique_ratings, key=lambda r: rating_avg_pd[r])
            
            # Initialize variables for ECE calculation
            total_samples = len(y_true)
            ece = 0.0
            bin_details = []
            
            # Calculate ECE using ratings
            for i, rating in enumerate(sorted_ratings):
                mask = ratings == rating
                bin_samples = np.sum(mask)
                
                if bin_samples > 0:
                    bin_y_true = y_true[mask]
                    bin_y_pred = y_pred[mask]
                    
                    # Calculate observed and predicted probabilities
                    observed_prob = np.mean(bin_y_true)
                    predicted_prob = np.mean(bin_y_pred)
                    
                    # Update ECE (weighted absolute difference)
                    ece += (bin_samples / total_samples) * abs(observed_prob - predicted_prob)
                    
                    # Store bin details
                    bin_details.append({
                        'bin': i + 1,
                        'rating': str(rating),
                        'n_samples': int(bin_samples),
                        'observed_prob': float(observed_prob),
                        'predicted_prob': float(predicted_prob),
                        'calibration_error': float(abs(observed_prob - predicted_prob))
                    })
            
            # Prepare figure data for rating-based ECE
            bin_labels = [b['rating'] for b in bin_details]
            observed_probs = [b['observed_prob'] for b in bin_details]
            predicted_probs = [b['predicted_prob'] for b in bin_details]
            calibration_errors = [b['calibration_error'] for b in bin_details]
            sample_counts = [b['n_samples'] for b in bin_details]
            
            figure_data = {
                'x': list(range(len(bin_details))),
                'y_observed': observed_probs,
                'y_predicted': predicted_probs,
                'bin_labels': bin_labels,
                'errors': calibration_errors,
                'weights': sample_counts,
                'title': f'Expected Calibration Error by Rating (ECE = {ece:.4f})',
                'xlabel': 'Rating Grade',
                'ylabel': 'Probability',
                'plot_type': 'ece',
                'annotations': [
                    f'ECE: {ece:.4f}',
                    f'Number of Ratings: {len(bin_details)}',
                    f'Total Samples: {total_samples}',
                ],
            }
            
            return MetricResult(
                name=self.__class__.__name__,
                value=ece,
                details={
                    'use_ratings': True,
                    'n_ratings': n_ratings,
                    'bin_details': bin_details,
                },
                figure_data=figure_data
            )
        else:
            # Standard ECE calculation using equal-width bins
            
            # Divide predictions into bins
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(y_pred, bin_edges[1:-1])
            
            # Initialize variables for ECE calculation
            total_samples = len(y_true)
            ece = 0.0
            bin_details = []
            
            # Calculate ECE using bins
            for i in range(n_bins):
                bin_mask = bin_indices == i
                bin_samples = np.sum(bin_mask)
                
                if bin_samples > 0:
                    bin_y_true = y_true[bin_mask]
                    bin_y_pred = y_pred[bin_mask]
                    
                    # Calculate observed and predicted probabilities
                    observed_prob = np.mean(bin_y_true)
                    predicted_prob = np.mean(bin_y_pred)
                    
                    # Update ECE (weighted absolute difference)
                    ece += (bin_samples / total_samples) * abs(observed_prob - predicted_prob)
                    
                    # Store bin details
                    bin_details.append({
                        'bin': i + 1,
                        'range': f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})",
                        'n_samples': int(bin_samples),
                        'observed_prob': float(observed_prob),
                        'predicted_prob': float(predicted_prob),
                        'calibration_error': float(abs(observed_prob - predicted_prob))
                    })
            
            # Prepare figure data for bin-based ECE
            bin_labels = [f"Bin {b['bin']}" for b in bin_details]
            observed_probs = [b['observed_prob'] for b in bin_details]
            predicted_probs = [b['predicted_prob'] for b in bin_details]
            calibration_errors = [b['calibration_error'] for b in bin_details]
            sample_counts = [b['n_samples'] for b in bin_details]
            
            figure_data = {
                'x': list(range(len(bin_details))),
                'y_observed': observed_probs,
                'y_predicted': predicted_probs,
                'bin_labels': bin_labels,
                'errors': calibration_errors,
                'weights': sample_counts,
                'title': f'Expected Calibration Error (ECE = {ece:.4f})',
                'xlabel': 'Predicted Probability Bin',
                'ylabel': 'Probability',
                'plot_type': 'ece',
                'annotations': [
                    f'ECE: {ece:.4f}',
                    f'Number of Bins: {n_bins}',
                    f'Total Samples: {total_samples}',
                ],
            }
            
            return MetricResult(
                name=self.__class__.__name__,
                value=ece,
                details={
                    'use_ratings': False,
                    'n_bins': n_bins,
                    'bin_edges': bin_edges.tolist(),
                    'bin_details': bin_details,
                },
                figure_data=figure_data
            )

class BrierScore(BaseMetric):
    """
    Computes the Brier score for probabilistic predictions.
    
    The Brier score measures the mean squared difference between predicted
    probabilities and actual outcomes. Lower values indicate better calibration,
    with 0 being perfect.
    
    Unlike hypothesis tests, the Brier score is a direct measure of predictive
    accuracy that quantifies the magnitude of calibration errors. It is particularly
    useful for comparing multiple models, with lower scores indicating better
    probabilistic forecasting performance.
    
    References:
    ----------
    - Regulatory: European Central Bank (2019). "ECB Guide to Internal Models: Risk-type-specific chapters," Section 7.
    - Academic: Brier, G.W. (1950). "Verification of Forecasts Expressed in Terms of Probability," Monthly Weather Review, 78(1), 1-3.
    - Regulatory: Oesterreichische Nationalbank (OeNB) (2004). "Rating Models and Validation," Guidelines on Credit Risk Management.

    When ratings are provided, the Brier score is calculated both overall
    and for each rating grade separately, providing more granular evaluation
    of model calibration across different segments of the portfolio.
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="calibration", config=config, config_path=config_path, **kwargs)
        
    def _compute_raw(self, y_true=None, y_pred=None, ratings=None, **kwargs):
        brier = brier_score_loss(y_true, y_pred)
        
        # Calculate the Brier score of a no-information model that predicts the base rate
        base_rate = np.mean(y_true)
        baseline_brier = base_rate * (1 - base_rate)
        
        # Calculate Brier skill score (improvement over baseline)
        # 1 = perfect, 0 = same as baseline, <0 = worse than baseline
        if baseline_brier > 0:
            brier_skill = 1 - (brier / baseline_brier)
        else:
            brier_skill = 0
        
        details = {
            'n_obs': len(y_true),
            'n_defaults': int(np.sum(y_true)),
            'baseline_brier': float(baseline_brier),
            'brier_skill_score': float(brier_skill),
            'use_ratings': ratings is not None
        }
        
        # If ratings are provided, calculate Brier score per rating
        if ratings is not None:
            ratings = np.asarray(ratings)
            unique_ratings = np.unique(ratings)
            rating_details = []
            
            for rating in unique_ratings:
                mask = ratings == rating
                if np.sum(mask) > 0:  # Skip empty ratings
                    rating_y_true = y_true[mask]
                    rating_y_pred = y_pred[mask]
                    
                    rating_brier = brier_score_loss(rating_y_true, rating_y_pred)
                    rating_base_rate = np.mean(rating_y_true)
                    rating_baseline_brier = rating_base_rate * (1 - rating_base_rate)
                    
                    if rating_baseline_brier > 0:
                        rating_brier_skill = 1 - (rating_brier / rating_baseline_brier)
                    else:
                        rating_brier_skill = 0
                    
                    rating_details.append({
                        'rating': str(rating),
                        'n_obs': int(np.sum(mask)),
                        'n_defaults': int(np.sum(rating_y_true)),
                        'brier_score': float(rating_brier),
                        'baseline_brier': float(rating_baseline_brier),
                        'brier_skill_score': float(rating_brier_skill)
                    })
            
            details['rating_details'] = rating_details
            
        return MetricResult(
            name=self.__class__.__name__,
            value=brier,
            details=details
        )

class JeffreysTest(BaseMetric):
    """
    Implement the Jeffreys test for PD calibration assessment.
    
    This test is defined in ECB's instructions on validation reporting section 2.5.3.1.
    
    The Jeffreys test uses a beta distribution with parameters a = D + 0.5 and b = N - D + 0.5,
    where D is the number of defaults and N is the total number of observations.
    
    Hypothesis test:
    - H0: The PD calibration is accurate (predicted PD falls within the confidence interval)
    - H1: The PD calibration is inaccurate (predicted PD falls outside the confidence interval)
    
    References:
    ----------
    - Regulatory: European Central Bank (2019). "Instructions for reporting the validation results of internal models," 
      Section 2.5.3.1. https://www.bankingsupervision.europa.eu/activities/internal_models/shared/pdf/instructions_validation_reporting_credit_risk.en.pdf
    - Academic: Tasche, D. (2008). "Validation of internal rating systems and PD estimates," 
      The Analytics of Risk Model Validation, 169-196.
    
    When ratings are provided, the test is performed at the rating level,
    which provides a more granular assessment of calibration across different
    segments of the portfolio.
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="calibration", config=config, config_path=config_path, **kwargs)

    def _compute_raw(self, y_true=None, y_pred=None, ratings=None, **kwargs):
        """
        Calculate the Jeffreys test for PD calibration.
        
        Parameters
        ----------
        y_true : array-like
            Binary target values (0,1)
        y_pred : array-like
            Predicted probabilities
        ratings : array-like, optional
            Rating grades or bucket assignments for each observation.
            If provided, the test is performed at the rating level.
            
        Returns
        -------
        MetricResult
            Containing the test result and confidence intervals
        """
        # Check if ratings are provided
        use_ratings = ratings is not None
        
        # Convert inputs to numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Set confidence level
        confidence_level = self._get_param("confidence_level", default=0.95)
        alpha = 1 - confidence_level
        
        if use_ratings:
            ratings = np.asarray(ratings)
            unique_ratings = np.unique(ratings)
            n_ratings = len(unique_ratings)
            
            # Sort ratings by average PD (ascending)
            rating_avg_pd = {r: np.mean(y_pred[ratings == r]) for r in unique_ratings}
            sorted_ratings = sorted(unique_ratings, key=lambda r: rating_avg_pd[r])
            
            rating_results = []
            overall_passed = True
            
            for rating in sorted_ratings:
                mask = ratings == rating
                rating_y_true = y_true[mask]
                rating_y_pred = y_pred[mask]
                
                # Skip ratings with no observations
                if len(rating_y_true) == 0:
                    continue
                
                # Calculate observed defaults and total observations
                D = np.sum(rating_y_true)
                N = len(rating_y_true)
                
                # Calculate average predicted PD
                avg_pd = np.mean(rating_y_pred)
                
                # Jeffreys parameters
                a = D + 0.5
                b = N - D + 0.5
                p_value = stats.beta.cdf(avg_pd, a, b)
                # Calculate confidence interval
                lower_bound = stats.beta.ppf(alpha/2, a, b)
                upper_bound = stats.beta.ppf(1 - alpha/2, a, b)
                
                # Determine if predicted PD is within confidence interval
                passed = lower_bound <= avg_pd <= upper_bound
                
                # Update overall test result
                if not passed:
                    overall_passed = False
                
                rating_results.append({
                    'rating': str(rating),
                    'n_obs': int(N),
                    'n_defaults': int(D),
                    'observed_dr': float(D/N if N > 0 else 0),
                    'avg_pd': float(avg_pd),
                    'p_value': float(p_value),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'passed': bool(passed)
                })
            
            # Prepare figure data
            rating_labels = [r['rating'] for r in rating_results]
            observed_drs = [r['observed_dr'] for r in rating_results]
            avg_pds = [r['avg_pd'] for r in rating_results]
            lower_bounds = [r['lower_bound'] for r in rating_results]
            upper_bounds = [r['upper_bound'] for r in rating_results]
            sample_counts = [r['n_obs'] for r in rating_results]
            
            figure_data = {
                'x': list(range(len(rating_results))),  # x-positions for ratings
                'y': observed_drs,  # Observed default rates
                'y_pred': avg_pds,  # Predicted PDs
                'y_lower': lower_bounds,  # Lower confidence bounds
                'y_upper': upper_bounds,  # Upper confidence bounds
                'rating_labels': rating_labels,  # Rating labels for x-axis
                'weights': sample_counts,  # Sample counts for marker size
                'title': f'Jeffreys Test by Rating Grade ({confidence_level*100:.0f}% CI, {"Passed" if overall_passed else "Failed"})',
                'xlabel': 'Rating Grade',
                'ylabel': 'Default Rate / PD',
                'plot_type': 'jeffreys',  # Custom plot type for Jeffreys test
                'annotations': [
                    f'Confidence Level: {confidence_level*100:.0f}%',
                    f'Test Result: {"Passed" if overall_passed else "Failed"}',
                ],
            }
            
            return MetricResult(
                name=self.__class__.__name__,
                value=1.0 if overall_passed else 0.0,  # Binary pass/fail as value
                passed=overall_passed,
                details={
                    'use_ratings': True,
                    'confidence_level': confidence_level,
                    'n_ratings': n_ratings,
                    'rating_results': rating_results
                },
                figure_data=figure_data
            )
        else:
            # Calculate observed defaults and total observations
            D = np.sum(y_true)
            N = len(y_true)
            
            # Calculate average predicted PD
            avg_pd = np.mean(y_pred)
            
            # Jeffreys parameters
            a = D + 0.5
            b = N - D + 0.5
            p_value = stats.beta.cdf(avg_pd, a, b)
            # Calculate confidence interval
            lower_bound = stats.beta.ppf(alpha/2, a, b)
            upper_bound = stats.beta.ppf(1 - alpha/2, a, b)
            
            # Determine if predicted PD is within confidence interval
            passed = lower_bound <= avg_pd <= upper_bound
            
            return MetricResult(
                name=self.__class__.__name__,
                value=1.0 if passed else 0.0,  # Binary pass/fail as value
                passed=passed,
                details={
                    'use_ratings': False,
                    'confidence_level': confidence_level,
                    'n_obs': int(N),
                    'n_defaults': int(D),
                    'observed_dr': float(D/N if N > 0 else 0),
                    'avg_pd': float(avg_pd),
                    'p_value': float(p_value),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                },
                figure_data={
                    'x': [0],  # Single point for overall portfolio
                    'y': [D/N if N > 0 else 0],  # Observed default rate
                    'y_pred': [avg_pd],  # Predicted PD
                    'y_lower': [lower_bound],  # Lower confidence bound
                    'y_upper': [upper_bound],  # Upper confidence bound
                    'title': f'Jeffreys Test ({confidence_level*100:.0f}% CI, {"Passed" if passed else "Failed"})',
                    'xlabel': 'Portfolio',
                    'ylabel': 'Default Rate / PD',
                    'plot_type': 'jeffreys',  # Custom plot type for Jeffreys test
                    'annotations': [
                        f'Confidence Level: {confidence_level*100:.0f}%',
                        f'Observed DR: {D/N*100:.2f}%',
                        f'Average PD: {avg_pd*100:.2f}%',
                        f'CI: [{lower_bound*100:.2f}%, {upper_bound*100:.2f}%]',
                    ],
                }
            )

class BinomialTest(BaseMetric):
    """
    Binomial test for PD calibration assessment.
    
    This test compares the observed number of defaults with the expected number
    based on predicted probabilities, using a binomial distribution.
    
    The implementation follows OeNB methodology (Rating Models and Validation, p.121) using a one-sided test:
    - H0: Observed defaults <= Expected defaults (model is well-calibrated or conservative)
    - H1: Observed defaults > Expected defaults (model underestimates risk)
    
    The p-value represents the probability of observing at least as many defaults as were actually observed,
    given the expected default rate from the model. Small p-values indicate the model may be underestimating risk.
    
    The test can be performed at the portfolio level or at the rating grade level
    when ratings are provided.
    
    References:
    ----------
    - Regulatory: Basel Committee on Banking Supervision (2005). "Studies on validation of internal rating systems," 
      Working Paper No. 14, pp. 30-31.
    - Regulatory: European Banking Authority (2017). "Guidelines on PD estimation, LGD estimation and the treatment of defaulted exposures," 
      EBA/GL/2017/16, Section 5.3.4.
    - Regulatory: Oesterreichische Nationalbank (OeNB) (2004). "Rating Models and Validation," 
      Guidelines on Credit Risk Management, p. 121.
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="calibration", config=config, config_path=config_path, **kwargs)
    
    def _compute_raw(self, y_true=None, y_pred=None, ratings=None, **kwargs):
        """
        Calculate the Binomial test for PD calibration.
        
        Parameters
        ----------
        y_true : array-like
            Binary target values (0,1)
        y_pred : array-like
            Predicted probabilities
        ratings : array-like, optional
            Rating grades or bucket assignments for each observation.
            If provided, the test is performed at the rating level.
            
        Returns
        -------
        MetricResult
            Containing the test result and p-values
        """
        # Check if ratings are provided
        use_ratings = ratings is not None
        
        # Convert inputs to numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Set confidence level
        confidence_level = self._get_param("confidence_level", default=0.95)
        alpha = 1 - confidence_level
        
        if use_ratings:
            ratings = np.asarray(ratings)
            unique_ratings = np.unique(ratings)
            n_ratings = len(unique_ratings)
            
            # Sort ratings by average PD (ascending)
            rating_avg_pd = {r: np.mean(y_pred[ratings == r]) for r in unique_ratings}
            sorted_ratings = sorted(unique_ratings, key=lambda r: rating_avg_pd[r])
            
            rating_results = []
            overall_passed = True
            
            for rating in sorted_ratings:
                mask = ratings == rating
                rating_y_true = y_true[mask]
                rating_y_pred = y_pred[mask]
                
                # Skip ratings with no observations
                if len(rating_y_true) == 0:
                    continue
                
                # Calculate observed defaults and expected defaults
                n_obs = len(rating_y_true)
                n_defaults = int(np.sum(rating_y_true))
                expected_defaults = np.sum(rating_y_pred)
                
                # Calculate p-value for one-sided binomial test (OeNB methodology)
                # This tests whether the observed defaults are significantly higher than expected
                # H0: observed defaults <= expected defaults (model is well-calibrated or conservative)
                # H1: observed defaults > expected defaults (model underestimates risk)
                p_value = 1 - stats.binom.cdf(n_defaults - 1, n_obs, expected_defaults / n_obs)
                
                # Determine if test passes
                passed = p_value > alpha
                
                # Update overall test result
                if not passed:
                    overall_passed = False
                
                rating_results.append({
                    'rating': str(rating),
                    'n_obs': int(n_obs),
                    'n_defaults': int(n_defaults),
                    'expected_defaults': float(expected_defaults),
                    'observed_dr': float(n_defaults / n_obs),
                    'expected_dr': float(expected_defaults / n_obs),
                    'p_value': float(p_value),
                    'passed': bool(passed)
                })
            
            # Prepare figure data
            rating_labels = [r['rating'] for r in rating_results]
            n_obs_values = [r['n_obs'] for r in rating_results]
            observed_defaults = [r['n_defaults'] for r in rating_results]
            expected_defaults = [r['expected_defaults'] for r in rating_results]
            
            figure_data = {
                'x': list(range(len(rating_results))),  # x-positions for ratings
                'y': observed_defaults,  # Observed defaults
                'y_expected': expected_defaults,  # Expected defaults
                'rating_labels': rating_labels,  # Rating labels for x-axis
                'n_obs': n_obs_values,  # Number of observations for reference
                'title': f'Binomial Test by Rating Grade ({confidence_level*100:.0f}% CL, {"Passed" if overall_passed else "Failed"})',
                'xlabel': 'Rating Grade',
                'ylabel': 'Number of Defaults',
                'plot_type': 'binomial',  # Custom plot type for binomial test
                'annotations': [
                    f'Confidence Level: {confidence_level*100:.0f}%',
                    f'Test Result: {"Passed" if overall_passed else "Failed"}',
                ],
            }
            
            return MetricResult(
                name=self.__class__.__name__,
                value=1.0 if overall_passed else 0.0,  # Binary pass/fail as value
                passed=overall_passed,
                details={
                    'use_ratings': True,
                    'confidence_level': confidence_level,
                    'n_ratings': n_ratings,
                    'rating_results': rating_results
                },
                figure_data=figure_data
            )
        else:
            # Calculate observed defaults and expected defaults
            n_obs = len(y_true)
            n_defaults = int(np.sum(y_true))
            expected_defaults = np.sum(y_pred)
            
            # Calculate p-value for one-sided binomial test (OeNB methodology)
            # This tests whether the observed defaults are significantly higher than expected
            # H0: observed defaults <= expected defaults (model is well-calibrated or conservative)
            # H1: observed defaults > expected defaults (model underestimates risk)
            p_value = 1 - stats.binom.cdf(n_defaults - 1, n_obs, expected_defaults / n_obs)
            
            # Determine if test passes
            passed = p_value > alpha
            
            return MetricResult(
                name=self.__class__.__name__,
                value=p_value,  # p-value as the metric value
                passed=passed,
                details={
                    'use_ratings': False,
                    'confidence_level': confidence_level,
                    'n_obs': int(n_obs),
                    'n_defaults': int(n_defaults),
                    'expected_defaults': float(expected_defaults),
                    'observed_dr': float(n_defaults / n_obs),
                    'expected_dr': float(expected_defaults / n_obs),
                },
                figure_data={
                    'x': [0],  # Single point for overall portfolio
                    'y': [n_defaults],  # Observed defaults
                    'y_expected': [expected_defaults],  # Expected defaults
                    'title': f'Binomial Test ({confidence_level*100:.0f}% CL, {"Passed" if passed else "Failed"})',
                    'xlabel': 'Portfolio',
                    'ylabel': 'Number of Defaults',
                    'plot_type': 'binomial',  # Custom plot type for binomial test
                    'annotations': [
                        f'Confidence Level: {confidence_level*100:.0f}%',
                        f'Observed Defaults: {n_defaults}',
                        f'Expected Defaults: {expected_defaults:.1f}',
                        f'p-value: {p_value:.4f}',
                    ],
                }
            )


class NormalTest(BaseMetric):
    """
    Normal test for PD calibration assessment.
    
    This test compares the observed number of defaults with the expected number
    based on predicted probabilities, using a normal distribution.
    
    The implementation follows OeNB methodology (Rating Models and Validation, p.120) using a one-sided test:
    - H0: Observed defaults <= Expected defaults (model is well-calibrated or conservative)
    - H1: Observed defaults > Expected defaults (model underestimates risk)
    
    The test can be performed at the portfolio level or at the rating grade level
    when ratings are provided.
    
    References:
    ----------
    - Regulatory: Basel Committee on Banking Supervision (2005). "Studies on validation of internal rating systems," 
      Working Paper No. 14, pp. 30-31.
    - Regulatory: European Banking Authority (2017). "Guidelines on PD estimation, LGD estimation and the treatment of defaulted exposures," 
      EBA/GL/2017/16, Section 5.3.4.
    - Regulatory: Oesterreichische Nationalbank (OeNB) (2004). "Rating Models and Validation," 
      Guidelines on Credit Risk Management, p. 121.


    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="calibration", config=config, config_path=config_path, **kwargs)
    
    def _compute_raw(self, y_true=None, y_pred=None, ratings=None, **kwargs):
        """
        Calculate the Normal test for PD calibration.
        
        Parameters
        ----------
        y_true : array-like
            Binary target values (0,1)
        y_pred : array-like
            Predicted probabilities
        ratings : array-like, optional
            Rating grades or bucket assignments for each observation.
            If provided, the test is performed at the rating level.
            
        Returns
        -------
        MetricResult
            Containing the test result and p-values
        """
        # Check if ratings are provided
        use_ratings = ratings is not None
        
        # Convert inputs to numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Set confidence level
        confidence_level = self._get_param("confidence_level", default=0.95)
        alpha = 1 - confidence_level
        
        if use_ratings:
            ratings = np.asarray(ratings)
            unique_ratings = np.unique(ratings)
            n_ratings = len(unique_ratings)
            
            # Sort ratings by average PD (ascending)
            rating_avg_pd = {r: np.mean(y_pred[ratings == r]) for r in unique_ratings}
            sorted_ratings = sorted(unique_ratings, key=lambda r: rating_avg_pd[r])
            
            rating_results = []
            overall_passed = True
            
            for rating in sorted_ratings:
                mask = ratings == rating
                rating_y_true = y_true[mask]
                rating_y_pred = y_pred[mask]
                
                # Skip ratings with no observations
                if len(rating_y_true) == 0:
                    continue
                
                # Calculate observed defaults and expected defaults
                n_obs = len(rating_y_true)
                n_defaults = int(np.sum(rating_y_true))
                expected_defaults = np.sum(rating_y_pred)
                
                # Calculate z-score for normal test
                if n_obs > 0:
                    observed_dr = n_defaults / n_obs
                    expected_dr = expected_defaults / n_obs
                    se = np.sqrt(expected_dr * (1 - expected_dr) / n_obs) if n_obs > 1 else 1
                    z_score = (observed_dr - expected_dr) / se if se > 0 else 0
                    
                    # Calculate p-value from z-score (one-sided test)
                    p_value = 1 - stats.norm.cdf(z_score)
                else:
                    p_value = 1.0
                
                # Determine if test passes
                passed = p_value > alpha
                
                # Update overall test result
                if not passed:
                    overall_passed = False
                
                rating_results.append({
                    'rating': str(rating),
                    'n_obs': int(n_obs),
                    'n_defaults': int(n_defaults),
                    'expected_defaults': float(expected_defaults),
                    'observed_dr': float(observed_dr),
                    'expected_dr': float(expected_dr),
                    'z_score': float(z_score),
                    'p_value': float(p_value),
                    'passed': bool(passed)
                })
            
            # Prepare figure data
            rating_labels = [r['rating'] for r in rating_results]
            n_obs_values = [r['n_obs'] for r in rating_results]
            observed_defaults = [r['n_defaults'] for r in rating_results]
            expected_defaults = [r['expected_defaults'] for r in rating_results]
            z_scores = [r['z_score'] for r in rating_results]
            
            figure_data = {
                'x': list(range(len(rating_results))),  # x-positions for ratings
                'y': observed_defaults,  # Observed defaults
                'y_expected': expected_defaults,  # Expected defaults
                'z_scores': z_scores,  # Z-scores for each rating
                'rating_labels': rating_labels,  # Rating labels for x-axis
                'n_obs': n_obs_values,  # Number of observations for reference
                'title': f'Normal Test by Rating Grade ({confidence_level*100:.0f}% CL, {"Passed" if overall_passed else "Failed"})',
                'xlabel': 'Rating Grade',
                'ylabel': 'Number of Defaults',
                'plot_type': 'normal',  # Custom plot type for normal test
                'annotations': [
                    f'Confidence Level: {confidence_level*100:.0f}%',
                    f'Test Result: {"Passed" if overall_passed else "Failed"}',
                ],
            }
            
            return MetricResult(
                name=self.__class__.__name__,
                value=1.0 if overall_passed else 0.0,  # Binary pass/fail as value
                passed=overall_passed,
                details={
                    'use_ratings': True,
                    'confidence_level': confidence_level,
                    'n_ratings': n_ratings,
                    'rating_results': rating_results
                },
                figure_data=figure_data
            )
        else:
            # Calculate observed defaults and expected defaults
            n_obs = len(y_true)
            n_defaults = int(np.sum(y_true))
            expected_defaults = np.sum(y_pred)

            # Calculate z-score for normal test
            if n_obs > 0:
                observed_dr = n_defaults / n_obs
                expected_dr = expected_defaults / n_obs
                se = np.sqrt(expected_dr * (1 - expected_dr) / n_obs) if n_obs > 1 else 1
                z_score = (observed_dr - expected_dr) / se if se > 0 else 0

                # Calculate p-value from z-score (one-sided test)
                p_value = 1 - stats.norm.cdf(z_score)
            else:
                observed_dr = 0
                expected_dr = 0
                z_score = 0
                p_value = 1.0

            # Determine if test passes
            passed = p_value > alpha

            return MetricResult(
                name=self.__class__.__name__,
                value=p_value,  # p-value as the metric value
                passed=passed,
                details={
                    'use_ratings': False,
                    'confidence_level': confidence_level,
                    'n_obs': int(n_obs),
                    'n_defaults': int(n_defaults),
                    'expected_defaults': float(expected_defaults),
                    'observed_dr': float(observed_dr),
                    'expected_dr': float(expected_dr),
                    'z_score': float(z_score)
                },
                figure_data={
                    'x': [0],  # Single point for overall portfolio
                    'y': [n_defaults],  # Observed defaults
                    'y_expected': [expected_defaults],  # Expected defaults
                    'z_score': [z_score],  # Z-score
                    'title': f'Normal Test ({confidence_level*100:.0f}% CL, {"Passed" if passed else "Failed"})',
                    'xlabel': 'Portfolio',
                    'ylabel': 'Number of Defaults',
                    'plot_type': 'normal',  # Custom plot type for normal test
                    'annotations': [
                        f'Confidence Level: {confidence_level*100:.0f}%',
                        f'Observed Defaults: {n_defaults}',
                        f'Expected Defaults: {expected_defaults:.1f}',
                        f'Z-score: {z_score:.2f}',
                        f'p-value: {p_value:.4f}',
                    ],
                }
            )

class RatingHeterogeneityTest(BaseMetric):
    """
    Test for assessing the heterogeneity between adjacent rating grades.
    
    This test evaluates whether consecutive rating grades show statistically significant
    differences in observed default rates, which is essential for a well-differentiated
    rating system. A properly functioning rating system should have distinct and statistically
    significant differences in default rates between neighboring grades.
    
    The test performs pairwise comparisons between adjacent rating grades using statistical
    tests (e.g., Fisher's exact test or proportion tests) to determine if the observed
    differences in default rates are significant or could occur by random chance.
    
    Null Hypothesis (H0): For each pair of adjacent rating grades, the default rates are
                          not significantly different (the grades do not effectively 
                          differentiate risk).
    
    Alternative Hypothesis (H1): For each pair of adjacent rating grades, the default rates
                                are significantly different (the grades effectively
                                differentiate risk).
    
    Interpretation Guidelines:
        - A low p-value for a pair of adjacent grades indicates statistically significant
          differentiation (desirable).
        
        - High p-values suggest adjacent grades may not be sufficiently differentiated,
          potentially indicating rating grade compression or boundary issues.
        
        - The test results help identify where the rating scale may need recalibration
          or where adjacent grades might be candidates for merging.
        
        - Consider both statistical significance and the magnitude of differences between
          adjacent grades.
    
    Regulatory Context:
        - Basel Committee on Banking Supervision guidelines require that rating systems 
          provide for a meaningful differentiation of risk.
        
        - EBA guidelines on PD estimation require institutions to demonstrate that the 
          rating system provides for a meaningful differentiation of risk.
        
        - ECB TRIM guidelines emphasize the need for clear separation between rating grades
          in terms of default risk.
    
    References:
        - Basel Committee on Banking Supervision (2006). "International Convergence of 
          Capital Measurement and Capital Standards: A Revised Framework."
        - European Banking Authority (2017). "Guidelines on PD estimation, LGD estimation
          and the treatment of defaulted exposures."
        - Tasche, D. (2009). "Estimating discriminatory power and PD curves when the number
          of defaults is small."
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="calibration", config=config, config_path=config_path, **kwargs)

    def _compute_raw(self, y_true=None, y_pred=None, ratings=None, **kwargs):
        """
        Compute heterogeneity test for adjacent rating grades.
        
        Parameters
        ----------
        y_true : array-like
            Binary target values (0 = non-default, 1 = default).
        y_pred : array-like
            Predicted probabilities of default.
        ratings : array-like
            Rating grade assignments for each observation. This parameter is required.
            
        Returns
        -------
        MetricResult
            Results of the rating heterogeneity test, including:
                - value: proportion of adjacent rating pairs with significant differentiation
                - passed: boolean indicating whether all adjacent rating pairs show significant differentiation
                - details: pair-level statistics and test results
                - figure_data: data for visualization of default rates across rating grades
        """
        # Input validation
        if ratings is None:
            raise ValueError("Ratings parameter is required for rating heterogeneity testing")
            
        if len(y_true) != len(ratings):
            raise ValueError("Input arrays must have the same length")
        
        # Convert inputs to numpy arrays
        y_true = np.asarray(y_true)
        ratings = np.asarray(ratings)
        
        # Get parameters from config or use defaults
        confidence_level = self._get_param("confidence_level", default=0.95)
        min_obs_per_rating = self._get_param("min_obs_per_rating", default=30)
        test_method = self._get_param("test_method", default="fisher")  # Options: "fisher", "proportion", "chi2"
        ascending = self._get_param("ascending", default=True)  # True if ratings increase with better quality
        
        # Create a DataFrame for easier analysis
        df = pd.DataFrame({
            'actual': y_true,
            'rating': ratings
        })
        
        # Get unique ratings and sort them
        unique_ratings = np.unique(ratings)
        
        # Sort ratings based on whether lower ratings indicate better or worse quality
        # By default (ascending=True), we assume ratings increase with better quality (lower PD)
        # This means we expect default rates to decrease as rating values increase
        if ascending:
            sorted_ratings = sorted(unique_ratings)
        else:
            sorted_ratings = sorted(unique_ratings, reverse=True)
        
        # Calculate default rates and counts for each rating
        rating_stats = []
        excluded_ratings = []
        
        for rating in sorted_ratings:
            mask = ratings == rating
            n_obs = mask.sum()
            
            # Skip ratings with too few observations
            if n_obs < min_obs_per_rating:
                excluded_ratings.append({
                    'rating': rating,
                    'n_obs': n_obs,
                    'reason': f'Fewer than {min_obs_per_rating} observations'
                })
                continue
                
            n_defaults = y_true[mask].sum()
            default_rate = n_defaults / n_obs
            
            rating_stats.append({
                'rating': rating,
                'n_obs': n_obs,
                'n_defaults': n_defaults,
                'default_rate': default_rate
            })
        
        # If less than 2 valid ratings, cannot perform heterogeneity test
        if len(rating_stats) < 2:
            warnings.warn("Insufficient valid ratings for heterogeneity testing.")
            return MetricResult(
                name=self.__class__.__name__,
                value=np.nan,
                passed=None,
                details={
                    'confidence_level': confidence_level,
                    'ratings_analyzed': len(rating_stats),
                    'excluded_ratings': excluded_ratings,
                    'rating_stats': rating_stats,
                    'error': "Insufficient valid ratings for heterogeneity testing"
                }
            )
        
        # Perform pairwise comparisons between adjacent rating grades
        pair_results = []
        
        for i in range(len(rating_stats) - 1):
            r1 = rating_stats[i]
            r2 = rating_stats[i + 1]
            
            # Compute the statistical test between adjacent rating grades
            if test_method == "fisher":
                # Fisher's exact test
                table = np.array([
                    [r1['n_defaults'], r1['n_obs'] - r1['n_defaults']],
                    [r2['n_defaults'], r2['n_obs'] - r2['n_defaults']]
                ])
                _, p_value = stats.fisher_exact(table)
            elif test_method == "proportion":
                # Proportion z-test
                count = np.array([r1['n_defaults'], r2['n_defaults']])
                nobs = np.array([r1['n_obs'], r2['n_obs']])
                stat, p_value = stats.proportions_ztest(count, nobs)
            elif test_method == "chi2":
                # Chi-square test
                table = np.array([
                    [r1['n_defaults'], r1['n_obs'] - r1['n_defaults']],
                    [r2['n_defaults'], r2['n_obs'] - r2['n_defaults']]
                ])
                stat, p_value, _, _ = stats.chi2_contingency(table)
            else:
                raise ValueError(f"Unknown test method: {test_method}")
            
            # Calculate absolute and relative difference
            abs_diff = abs(r1['default_rate'] - r2['default_rate'])
            rel_diff = abs_diff / max(r1['default_rate'], r2['default_rate']) if max(r1['default_rate'], r2['default_rate']) > 0 else np.nan
            
            # Determine if difference is significant
            alpha = 1 - confidence_level
            is_significant = p_value < alpha
            
            pair_results.append({
                'rating1': r1['rating'],
                'rating2': r2['rating'],
                'dr1': r1['default_rate'],
                'dr2': r2['default_rate'],
                'n_obs1': r1['n_obs'],
                'n_obs2': r2['n_obs'],
                'n_defaults1': r1['n_defaults'],
                'n_defaults2': r2['n_defaults'],
                'abs_diff': abs_diff,
                'rel_diff': rel_diff,
                'p_value': p_value,
                'is_significant': is_significant,
                'test_method': test_method
            })
        
        # Calculate overall result
        significant_pairs = sum(1 for pair in pair_results if pair['is_significant'])
        proportion_significant = significant_pairs / len(pair_results) if pair_results else 0
        
        # Test is passed if all adjacent pairs show significant differentiation
        passed = proportion_significant == 1.0
        
        # Prepare data for visualization
        ratings_for_plot = [str(stat['rating']) for stat in rating_stats]
        default_rates = [stat['default_rate'] for stat in rating_stats]
        n_obs_values = [stat['n_obs'] for stat in rating_stats]
        
        # Mark ratings with significant difference to next rating
        significant_markers = []
        for i in range(len(ratings_for_plot)):
            if i < len(ratings_for_plot) - 1:
                pair_idx = i
                is_sig = pair_results[pair_idx]['is_significant']
                significant_markers.append(is_sig)
            else:
                # Last rating has no pair
                significant_markers.append(None)
        
        return MetricResult(
            name=self.__class__.__name__,
            value=proportion_significant,  # Proportion of significantly different adjacent pairs
            passed=passed,
            details={
                'confidence_level': confidence_level,
                'test_method': test_method,
                'ratings_analyzed': len(rating_stats),
                'excluded_ratings': excluded_ratings,
                'rating_stats': rating_stats,
                'pair_results': pair_results,
                'significant_pairs': significant_pairs,
                'total_pairs': len(pair_results),
                'proportion_significant': proportion_significant
            },
            figure_data={
                'x': ratings_for_plot,
                'y': default_rates,
                'n_obs': n_obs_values,
                'significant_markers': significant_markers,
                'title': f'Rating Heterogeneity Test ({confidence_level*100:.0f}% CL, {"Passed" if passed else "Failed"})',
                'xlabel': 'Rating Grade',
                'ylabel': 'Default Rate',
                'plot_type': 'rating_heterogeneity',
                'annotations': [
                    f'Confidence Level: {confidence_level*100:.0f}%',
                    f'Test Method: {test_method}',
                    f'Ratings Analyzed: {len(rating_stats)}',
                    f'Significant Pairs: {significant_pairs}/{len(pair_results)} ({proportion_significant*100:.0f}%)',
                    f'Result: {"Passed" if passed else "Failed"}'
                ],
            }
        )

class RatingHomogeneityTest(BaseMetric):
    """
    Test for rating homogeneity across different segments within a single rating grade.
    
    This test evaluates whether default rates are consistent across different segments
    within the same rating grade. A homogeneous rating grade should have similar default
    rates across all segments, regardless of other characteristics (e.g., PD, region, etc.).
    
    The test operates by taking one rating grade at a time, splitting it into subbuckets
    (usually based on predicted PD), and checking if there are significant differences
    in observed default rates across these subbuckets.
    
    Null Hypothesis (H0): Default rates within a rating grade are homogeneous across
                          different segments; any observed differences are due to 
                          random chance.
    
    Alternative Hypothesis (H1): There are statistically significant differences in
                                default rates within a rating grade across different
                                segments, indicating heterogeneity.
    
    Interpretation Guidelines:
        - A low p-value indicates statistically significant evidence against homogeneity
          within the rating grade, suggesting that the rating assignment may be influenced
          by factors not fully captured in the model.
          
        - High p-values suggest consistency in default rates within the rating grade,
          which is generally desirable.
          
        - If a rating grade shows significant heterogeneity, it may indicate issues with
          rating assignment criteria or potential biases in different segments.
    
    Regulatory Context:
        - Basel Committee on Banking Supervision guidance requires that a rating system
          must group obligors with similar risk characteristics in the same grade.
          
        - EBA guidelines on PD estimation require institutions to analyze whether the risk
          drivers and rating criteria allow for a meaningful differentiation of risk.
          
        - ECB TRIM guidelines emphasize the need for homogeneity within rating grades to
          ensure consistent risk quantification.
    
    References:
        - Basel Committee on Banking Supervision (2006). "International Convergence of 
          Capital Measurement and Capital Standards: A Revised Framework," paragraph 410.
          
        - European Banking Authority (2017). "Guidelines on PD estimation, LGD estimation
          and the treatment of defaulted exposures," EBA/GL/2017/16, section 4.2.
          
        - Basel Committee on Banking Supervision (2005). "Studies on validation of
          internal rating systems," Working Paper No. 14, Section 3.2.
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="calibration", config=config, config_path=config_path, **kwargs)

    def _compute_raw(self, y_true=None, y_pred=None, ratings=None, n_buckets=None, **kwargs):
        """
        Compute homogeneity test for each rating grade.
        
        Parameters
        ----------
        y_true : array-like
            Binary target values (0 = non-default, 1 = default).
        y_pred : array-like
            Predicted probabilities of default.
        ratings : array-like
            Rating grade assignments for each observation. This parameter is required.
        n_buckets : int, optional
            Number of buckets to split each rating grade into based on predicted PD.
            If not provided, will use a reasonable default based on sample size.
            
        Returns
        -------
        MetricResult
            Results of the rating homogeneity test, including:
                - value: proportion of rating grades with homogeneous default rates
                - passed: boolean indicating whether all rating grades show homogeneity
                - details: rating-level statistics and test results
                - figure_data: data for visualization of homogeneity within rating grades
        """
        # Input validation
        if ratings is None:
            raise ValueError("Ratings parameter is required for rating homogeneity testing")
            
        if len(y_true) != len(y_pred) or len(y_true) != len(ratings):
            raise ValueError("Input arrays must have the same length")
        
        # Convert inputs to numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        ratings = np.asarray(ratings)
        
        # Get parameters from config or use defaults
        confidence_level = self._get_param("confidence_level", default=0.95)
        min_obs_per_bucket = self._get_param("min_obs_per_bucket", default=20)
        min_obs_per_rating = self._get_param("min_obs_per_rating", default=100)
        
        # Determine number of buckets if not provided
        if n_buckets is None:
            # Default to 5 buckets, but adjust based on sample size
            n_buckets = self._get_param("n_buckets", default=5)
        
        # Create a DataFrame for easier analysis
        df = pd.DataFrame({
            'actual': y_true,
            'predicted': y_pred,
            'rating': ratings
        })
        
        # Get unique ratings
        unique_ratings = np.unique(ratings)
        
        # Test each rating grade for homogeneity
        rating_results = []
        excluded_ratings = []
        overall_passed = True
        
        for rating in unique_ratings:
            # Filter data for this rating grade
            rating_df = df[df['rating'] == rating].copy()
            n_obs = len(rating_df)
            
            # Skip ratings with too few observations
            if n_obs < min_obs_per_rating:
                excluded_ratings.append({
                    'rating': rating,
                    'n_obs': n_obs,
                    'reason': f'Fewer than {min_obs_per_rating} observations'
                })
                continue
            
            # Calculate overall default rate for this rating
            n_defaults = rating_df['actual'].sum()
            overall_dr = n_defaults / n_obs
            
            # Skip ratings with no defaults or all defaults (no variability to test)
            if overall_dr == 0 or overall_dr == 1:
                excluded_ratings.append({
                    'rating': rating,
                    'n_obs': n_obs,
                    'overall_dr': overall_dr,
                    'reason': 'No variability in default rate (all 0 or all 1)'
                })
                continue
            
            # Create buckets based on predicted PD
            rating_df['bucket'] = pd.qcut(
                rating_df['predicted'], 
                q=n_buckets, 
                labels=False, 
                duplicates='drop'
            )
            
            # Calculate statistics by bucket
            bucket_stats = []
            valid_buckets = []
            n_valid_buckets = 0
            
            for bucket, bucket_df in rating_df.groupby('bucket'):
                bucket_n_obs = len(bucket_df)
                
                # Skip buckets with too few observations
                if bucket_n_obs < min_obs_per_bucket:
                    continue
                
                bucket_n_defaults = bucket_df['actual'].sum()
                bucket_dr = bucket_n_defaults / bucket_n_obs
                bucket_avg_pd = bucket_df['predicted'].mean()
                
                bucket_stats.append({
                    'bucket': int(bucket),
                    'n_obs': int(bucket_n_obs),
                    'n_defaults': int(bucket_n_defaults),
                    'default_rate': float(bucket_dr),
                    'avg_pd': float(bucket_avg_pd),
                })
                
                valid_buckets.append(bucket)
                n_valid_buckets += 1
            
            # Skip ratings with fewer than 2 valid buckets
            if n_valid_buckets < 2:
                excluded_ratings.append({
                    'rating': rating,
                    'n_obs': n_obs,
                    'reason': f'Fewer than 2 valid buckets after applying minimum observation threshold'
                })
                continue
            
            # Perform chi-square test for homogeneity of default rates across buckets
            # Create contingency table: rows = buckets, columns = [defaults, non-defaults]
            contingency_table = np.array([
                [stat['n_defaults'], stat['n_obs'] - stat['n_defaults']]
                for stat in bucket_stats
            ])
            
            # Perform chi-square test
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            # Determine if test passes (null hypothesis of homogeneity not rejected)
            alpha = 1 - confidence_level
            passed = p_value >= alpha
            
            # Update overall test result
            if not passed:
                overall_passed = False
            
            # Calculate metrics for visualization
            min_dr = min(stat['default_rate'] for stat in bucket_stats)
            max_dr = max(stat['default_rate'] for stat in bucket_stats)
            range_dr = max_dr - min_dr
            relative_range = range_dr / overall_dr if overall_dr > 0 else np.nan
            
            rating_results.append({
                'rating': rating,
                'n_obs': int(n_obs),
                'n_defaults': int(n_defaults),
                'overall_dr': float(overall_dr),
                'n_buckets': int(n_valid_buckets),
                'chi2_statistic': float(chi2),
                'degrees_of_freedom': int(dof),
                'p_value': float(p_value),
                'min_dr': float(min_dr),
                'max_dr': float(max_dr),
                'range_dr': float(range_dr),
                'relative_range': float(relative_range),
                'passed': bool(passed),
                'bucket_stats': bucket_stats
            })
        
        # Calculate final result
        valid_ratings_count = len(rating_results)
        homogeneous_ratings_count = sum(1 for result in rating_results if result['passed'])
        proportion_homogeneous = homogeneous_ratings_count / valid_ratings_count if valid_ratings_count > 0 else np.nan
        
        # Prepare visualization data
        ratings_for_plot = [str(result['rating']) for result in rating_results]
        p_values = [result['p_value'] for result in rating_results]
        relative_ranges = [result['relative_range'] for result in rating_results]
        n_obs_values = [result['n_obs'] for result in rating_results]
        passed_values = [result['passed'] for result in rating_results]
        
        # Create detailed buckets data for visualization
        buckets_data = []
        for result in rating_results:
            rating = result['rating']
            for bucket in result['bucket_stats']:
                buckets_data.append({
                    'rating': str(rating),
                    'bucket': bucket['bucket'],
                    'default_rate': bucket['default_rate'],
                    'avg_pd': bucket['avg_pd'],
                    'n_obs': bucket['n_obs']
                })
        
        return MetricResult(
            name=self.__class__.__name__,
            value=proportion_homogeneous,  # Proportion of homogeneous rating grades
            passed=overall_passed,
            details={
                'confidence_level': confidence_level,
                'ratings_analyzed': valid_ratings_count,
                'homogeneous_ratings': homogeneous_ratings_count,
                'proportion_homogeneous': proportion_homogeneous,
                'excluded_ratings': excluded_ratings,
                'rating_results': rating_results,
                'buckets_data': buckets_data
            },
            figure_data={
                'x': ratings_for_plot,
                'y': p_values,
                'relative_ranges': relative_ranges,
                'n_obs': n_obs_values,
                'passed': passed_values,
                'buckets_data': buckets_data,
                'title': f'Rating Homogeneity Test ({confidence_level*100:.0f}% CL, {"Passed" if overall_passed else "Failed"})',
                'xlabel': 'Rating Grade',
                'ylabel': 'p-value',
                'plot_type': 'rating_homogeneity',
                'annotations': [
                    f'Confidence Level: {confidence_level*100:.0f}%',
                    f'Ratings Analyzed: {valid_ratings_count}',
                    f'Homogeneous Ratings: {homogeneous_ratings_count}/{valid_ratings_count} ({proportion_homogeneous*100:.0f}%)',
                    f'Ratings Excluded: {len(excluded_ratings)}',
                    f'Result: {"Passed" if overall_passed else "Failed"}'
                ],
            }
        )


class HerfindahlIndex(BaseMetric):
    """
    Calculate the Herfindahl Index for rating grade concentration.
    
    The Herfindahl Index (HI) is a measure of concentration used to assess
    whether exposures are too concentrated in certain rating grades. According
    to ECB guidelines on validation reporting (p. 26), institutions should monitor
    the concentration of exposures across rating grades.
    
    The index is calculated as the sum of squared proportions of exposures in each
    rating grade. A higher value indicates greater concentration, with 1 being the
    maximum (all exposures in a single grade) and 1/N being the minimum (exposures
    equally distributed across all N grades).
    
    HI = Σ(s_i^2) where s_i is the proportion of exposures in rating grade i
    
    Interpretation Guidelines:
        - HI = 1: Maximum concentration (all exposures in one rating grade)
        - HI = 1/N: Minimum concentration (exposures equally distributed across N grades)
        - Higher values indicate greater concentration risk
        
    The normalized Herfindahl Index (HI*) is also calculated, which ranges from 0 to 1:
        - HI* = (HI - 1/N) / (1 - 1/N)
        - HI* = 0: Perfect diversification
        - HI* = 1: Maximum concentration
    
    Regulatory Context:
        - ECB Instructions on validation reporting: "... institutions should analyse
          the portfolio composition by rating grades in order to identify possible
          excessive concentrations... for example... using the Herfindahl index" (p. 26)
        - High concentration in certain grades may indicate issues with the rating
          process or model calibration
    
    References:
        - European Central Bank (2019). "Instructions for reporting the validation results
          of internal models," Section 2.5.3.3, p. 26.
        - Basel Committee on Banking Supervision (2005). "Studies on validation of
          internal rating systems," Working Paper No. 14, Section 3.1.1.
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="calibration", config=config, config_path=config_path, **kwargs)
    
    def _compute_raw(self, ratings=None, exposures=None, **kwargs):
        """
        Calculate the Herfindahl Index for rating grade concentration.
        
        Parameters
        ----------
        ratings : array-like
            Rating grade assignments for each observation.
        exposures : array-like, optional
            Exposure values for each observation. If not provided, each observation
            is given equal weight (exposure=1).
            
        Returns
        -------
        MetricResult
            Results of the Herfindahl Index calculation, including:
                - value: Herfindahl Index value
                - details: normalized index, grade-level statistics
                - figure_data: data for visualization of rating distribution
        """
        # Input validation
        if ratings is None:
            raise ValueError("Ratings parameter is required for Herfindahl Index calculation")
        
        # Convert inputs to numpy arrays
        ratings = np.asarray(ratings)
        
        # If exposures not provided, use equal weights
        if exposures is None:
            exposures = np.ones_like(ratings, dtype=float)
        else:
            exposures = np.asarray(exposures, dtype=float)
            
            if len(ratings) != len(exposures):
                raise ValueError("Ratings and exposures arrays must have the same length")
        
        # Get threshold for concentration warning
        hi_threshold = self._get_param("hi_threshold", default=0.18)  # Common threshold for moderate concentration
        
        # Get unique ratings
        unique_ratings = np.unique(ratings)
        n_ratings = len(unique_ratings)
        
        # Calculate total exposure
        total_exposure = np.sum(exposures)
        
        if total_exposure <= 0:
            raise ValueError("Total exposure must be positive")
        
        # Calculate exposure by rating grade
        rating_stats = []
        
        for rating in unique_ratings:
            mask = ratings == rating
            rating_exposure = np.sum(exposures[mask])
            rating_proportion = rating_exposure / total_exposure
            
            rating_stats.append({
                'rating': rating,
                'exposure': float(rating_exposure),
                'proportion': float(rating_proportion),
                'squared_proportion': float(rating_proportion ** 2)
            })
        
        # Sort rating stats by proportion (descending)
        rating_stats = sorted(rating_stats, key=lambda x: x['proportion'], reverse=True)
        
        # Calculate Herfindahl Index
        hi = sum(stat['squared_proportion'] for stat in rating_stats)
        
        # Calculate minimum possible HI (equal distribution)
        min_hi = 1 / n_ratings if n_ratings > 0 else 1
        
        # Calculate normalized Herfindahl Index (ranges from 0 to 1)
        # HI* = (HI - 1/N) / (1 - 1/N)
        if n_ratings > 1:
            hi_normalized = (hi - min_hi) / (1 - min_hi)
        else:
            hi_normalized = 1.0  # If only one rating, maximum concentration
        
        # Determine if concentration is concerning
        passed = hi <= hi_threshold
        
        # Calculate additional metrics
        top_3_concentration = sum(stat['proportion'] for stat in rating_stats[:min(3, len(rating_stats))])
        top_5_concentration = sum(stat['proportion'] for stat in rating_stats[:min(5, len(rating_stats))])
        
        # Prepare data for visualization
        ratings_for_plot = [str(stat['rating']) for stat in rating_stats]
        proportions = [stat['proportion'] for stat in rating_stats]
        
        # Calculate effective number of rating grades
        # This represents how many equally-sized grades would give the same HI
        effective_n = 1 / hi if hi > 0 else n_ratings
        
        return MetricResult(
            name=self.__class__.__name__,
            value=hi,  # Herfindahl Index as the metric value
            threshold=hi_threshold,
            passed=passed,
            details={
                'n_ratings': n_ratings,
                'hi_normalized': float(hi_normalized),
                'min_hi': float(min_hi),
                'effective_n': float(effective_n),
                'top_3_concentration': float(top_3_concentration),
                'top_5_concentration': float(top_5_concentration),
                'rating_stats': rating_stats,
                'concentration_category': self._get_concentration_category(hi_normalized)
            },
            figure_data={
                'x': ratings_for_plot,
                'y': proportions,
                'title': f'Rating Grade Concentration (HI = {hi:.4f}, {"Passed" if passed else "Failed"})',
                'xlabel': 'Rating Grade',
                'ylabel': 'Proportion of Total Exposure',
                'plot_type': 'concentration',
                'annotations': [
                    f'Herfindahl Index: {hi:.4f}',
                    f'Normalized HI: {hi_normalized:.4f}',
                    f'Effective # of Grades: {effective_n:.1f} of {n_ratings}',
                    f'Concentration: {self._get_concentration_category(hi_normalized)}',
                    f'Top 3 Concentration: {top_3_concentration:.1%}',
                ],
            }
        )
    
    def _get_concentration_category(self, hi_normalized):
        """
        Categorize the concentration level based on the normalized Herfindahl Index.
        
        Parameters
        ----------
        hi_normalized : float
            Normalized Herfindahl Index value (0-1)
            
        Returns
        -------
        str
            Concentration category description
        """
        if hi_normalized < 0.15:
            return "Low Concentration"
        elif hi_normalized < 0.25:
            return "Moderate Concentration"
        elif hi_normalized < 0.40:
            return "High Concentration"
        else:
            return "Very High Concentration"