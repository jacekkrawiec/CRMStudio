"""
Calibration metrics for probability of default (PD) models.

This module provides metrics for assessing the calibration quality of PD models,
including Hosmer-Lemeshow test, Brier score, Expected Calibration Error (ECE),
and calibration curves.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from typing import Tuple, Dict, List, Optional, Union

from ...core.base import BaseMetric, MetricResult

class HosmerLemeshow(BaseMetric):
    """
    Hosmer-Lemeshow goodness-of-fit test for binary classification models.
    
    This test evaluates whether the observed event rates match expected event rates
    in subgroups of the model population. A small p-value indicates poor calibration.
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


class BrierScore(BaseMetric):
    """
    Computes the Brier score for probabilistic predictions.
    
    The Brier score measures the mean squared difference between predicted
    probabilities and actual outcomes. Lower values indicate better calibration,
    with 0 being perfect.
    
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
    (https://www.bankingsupervision.europa.eu/activities/internal_models/shared/pdf/instructions_validation_reporting_credit_risk.en.pdf)
    
    The Jeffreys test uses a beta distribution with parameters a = D + 0.5 and b = N - D + 0.5,
    where D is the number of defaults and N is the total number of observations.
    
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

class ExpectedCalibrationError(BaseMetric):
    """
    Calculate the Expected Calibration Error (ECE).
    
    ECE is a weighted average of the absolute difference between
    predicted probabilities and observed frequencies, where weights
    are determined by the number of samples in each bin.
    
    The metric can operate in two modes:
    1. Standard mode: Using equal-width probability bins (default)
    2. Rating mode: Using provided rating grades
    
    When ratings are provided, the calibration error is evaluated at the rating level,
    which is more relevant for credit risk models where observations are often
    grouped into discrete rating grades.
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="calibration", config=config, config_path=config_path, **kwargs)
        
    def _compute_raw(self, y_true=None, y_pred=None, ratings=None, **kwargs):
        n_bins = self._get_param("n_bins", default=10)
        
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
            
            ece = 0.0
            bin_details = []
            
            # Calculate statistics for each rating
            for i, rating in enumerate(unique_ratings):
                # Get predictions and labels for current rating
                bin_mask = (ratings == rating)
                bin_count = np.sum(bin_mask)
                
                if bin_count > 0:
                    bin_preds = y_pred[bin_mask]
                    bin_true = y_true[bin_mask]
                    
                    # Calculate average predicted probability and observed frequency
                    bin_pred_mean = np.mean(bin_preds)
                    bin_true_mean = np.mean(bin_true)
                    
                    # Calculate absolute calibration error for this rating
                    bin_error = np.abs(bin_pred_mean - bin_true_mean)
                    
                    # Weight the error by the proportion of samples in this rating
                    bin_weight = bin_count / len(y_true)
                    ece += bin_weight * bin_error
                    
                    bin_details.append({
                        'bin': i + 1,
                        'rating': str(rating),
                        'n_samples': int(bin_count),
                        'mean_pred': float(bin_pred_mean),
                        'mean_true': float(bin_true_mean),
                        'error': float(bin_error),
                        'weight': float(bin_weight),
                    })
            
            # Create arrays for plot
            mean_preds = [d['mean_pred'] for d in bin_details]
            mean_trues = [d['mean_true'] for d in bin_details]
            weights = [d['weight'] for d in bin_details]
            rating_labels = [d['rating'] for d in bin_details]
            
            return MetricResult(
                name=self.__class__.__name__,
                value=float(ece),
                details={
                    'use_ratings': True,
                    'n_ratings': n_ratings,
                    'bin_details': bin_details,
                },
                figure_data={
                    'x': mean_preds,  # Predicted probabilities
                    'y': mean_trues,  # Observed frequencies
                    'x_ref': [0, 1],  # Diagonal reference line x-coords
                    'y_ref': [0, 1],  # Diagonal reference line y-coords
                    'weights': weights,  # Sample weights for bubble size
                    'rating_names': rating_labels,  # Rating labels for annotation
                    'actual_label': 'Observed Default Rate',
                    'ref_label': 'Perfect Calibration',
                    'title': f'Expected Calibration Error by Rating Grade (ECE = {ece:.4f})',
                    'xlabel': 'Mean Predicted Probability',
                    'ylabel': 'Observed Default Rate',
                    'plot_type': 'bubble',  # Indicate this should be a bubble plot
                }
            )
        else:
            # Create bins of equal width
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(y_pred, bin_edges[1:-1])
            
            ece = 0.0
            bin_details = []
            
            for bin_idx in range(n_bins):
                # Get predictions and labels for current bin
                bin_mask = (bin_indices == bin_idx)
                bin_count = np.sum(bin_mask)
                
                if bin_count > 0:
                    bin_preds = y_pred[bin_mask]
                    bin_true = y_true[bin_mask]
                    
                    # Calculate average predicted probability and observed frequency
                    bin_pred_mean = np.mean(bin_preds)
                    bin_true_mean = np.mean(bin_true)
                    
                    # Calculate absolute calibration error for this bin
                    bin_error = np.abs(bin_pred_mean - bin_true_mean)
                    
                    # Weight the error by the proportion of samples in this bin
                    bin_weight = bin_count / len(y_true)
                    ece += bin_weight * bin_error
                    
                    bin_details.append({
                        'bin': bin_idx + 1,
                        'bin_lower': float(bin_edges[bin_idx]),
                        'bin_upper': float(bin_edges[bin_idx + 1]),
                        'n_samples': int(bin_count),
                        'mean_pred': float(bin_pred_mean),
                        'mean_true': float(bin_true_mean),
                        'error': float(bin_error),
                        'weight': float(bin_weight),
                    })
                else:
                    bin_details.append({
                        'bin': bin_idx + 1,
                        'bin_lower': float(bin_edges[bin_idx]),
                        'bin_upper': float(bin_edges[bin_idx + 1]),
                        'n_samples': 0,
                        'mean_pred': float('nan'),
                        'mean_true': float('nan'),
                        'error': float('nan'),
                        'weight': 0.0,
                    })
            
            # Create arrays for plot
            bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(n_bins)]
            mean_preds = [d['mean_pred'] for d in bin_details if d['n_samples'] > 0]
            mean_trues = [d['mean_true'] for d in bin_details if d['n_samples'] > 0]
            weights = [d['weight'] for d in bin_details if d['n_samples'] > 0]
            valid_bin_centers = [bin_centers[i] for i, d in enumerate(bin_details) if d['n_samples'] > 0]
            
            return MetricResult(
                name=self.__class__.__name__,
                value=float(ece),
                details={
                    'use_ratings': False,
                    'n_bins': n_bins,
                    'bin_details': bin_details,
                },
                figure_data={
                    'x': valid_bin_centers,  # Bin centers
                    'y': mean_trues,  # Observed frequencies
                    'x_ref': [0, 1],  # Diagonal reference line x-coords
                    'y_ref': [0, 1],  # Diagonal reference line y-coords
                    'weights': weights,  # Sample weights for bubble size
                    'predictions': mean_preds,  # Average predictions for annotation
                    'actual_label': 'Observed Frequency',
                    'ref_label': 'Perfect Calibration',
                    'title': f'Expected Calibration Error (ECE = {ece:.4f})',
                    'xlabel': 'Predicted Probability',
                    'ylabel': 'Observed Frequency',
                    'plot_type': 'bubble',  # Indicate this should be a bubble plot
                }
            )

class BinomialTest(BaseMetric):
    """
    Binomial test for PD calibration assessment.
    
    This test compares the observed number of defaults with the expected number
    based on predicted probabilities, using a binomial distribution.
    
    The test can be performed at the portfolio level or at the rating grade level
    when ratings are provided.
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
                
                # Calculate p-value for two-sided binomial test
                # Probability of observing <= n_defaults
                p_lower = stats.binom.cdf(n_defaults, n_obs, expected_defaults / n_obs)
                # Probability of observing >= n_defaults
                p_upper = 1 - stats.binom.cdf(n_defaults - 1, n_obs, expected_defaults / n_obs)
                
                # Two-sided p-value is twice the minimum of p_lower and p_upper
                p_value = 2 * min(p_lower, p_upper)
                if p_value > 1:
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
            
            # Calculate p-value for two-sided binomial test
            # Probability of observing <= n_defaults
            p_lower = stats.binom.cdf(n_defaults, n_obs, expected_defaults / n_obs)
            # Probability of observing >= n_defaults
            p_upper = 1 - stats.binom.cdf(n_defaults - 1, n_obs, expected_defaults / n_obs)
            
            # Two-sided p-value is twice the minimum of p_lower and p_upper
            p_value = 2 * min(p_lower, p_upper)
            if p_value > 1:
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
