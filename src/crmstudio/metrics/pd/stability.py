"""
Stability metrics for probability of default (PD) models.

This module provides metrics for assessing the stability of PD models over time,
including Population Stability Index (PSI), Characteristic Stability Index (CSI),
and temporal drift detection methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
import warnings

from ...core.base import BaseMetric
from ...core.data_classes import MetricResult


class PSI(BaseMetric):
    """
    Population Stability Index (PSI) for monitoring score or PD stability over time.
    
    PSI measures the shift in the distribution of scores/PDs between a reference
    (development/validation) dataset and a recent dataset. It is calculated by 
    binning the scores and comparing the proportion of observations in each bin.
    
    Calculation:
    PSI = Sum[ (% Recent - % Reference) * ln(% Recent / % Reference) ]
    
    PSI interpretation:
    - PSI < 0.1: No significant change in population
    - 0.1 ≤ PSI < 0.2: Moderate shift, requires investigation
    - PSI ≥ 0.2: Significant shift, model recalibration/redevelopment needed
    
    Hypothesis test:
    - H0: The distribution is stable (PSI < threshold)
    - H1: The distribution has shifted (PSI ≥ threshold)
    
    References:
    ----------
    - Regulatory: European Central Bank (2019). "Instructions for reporting the validation results 
      of internal models," Section 2.7.3. 
      https://www.bankingsupervision.europa.eu/banking/tasks/internal_models/shared/pdf/instructions_validation_reporting_credit_risk.en.pdf
    - Industry: Siddiqi, N. (2017). "Intelligent Credit Scoring: Building and Implementing 
      Better Credit Risk Scorecards," 2nd Edition, Wiley.
    """
    
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="psi", config=config, config_path=config_path, **kwargs)
    
    def _compute_raw(self, reference_scores: np.ndarray, recent_scores: np.ndarray, 
                    bins: Union[int, list] = 10, **kwargs):
        """
        Calculate the Population Stability Index.
        
        Parameters
        ----------
        reference_scores : array-like
            Scores or PDs from the reference period (development/validation)
        recent_scores : array-like
            Scores or PDs from the recent period
        bins : int or list, optional
            Number of bins or explicit bin edges
            
        Returns
        -------
        MetricResult
            Object containing PSI value and detailed bin information
        """
        # Parameter validation
        reference_scores = np.asarray(reference_scores)
        recent_scores = np.asarray(recent_scores)
        
        # Get threshold from configuration
        threshold = self._get_param("threshold", default=0.1)
        
        # Create bins based on reference distribution
        if isinstance(bins, int):
            # Use quantiles for more even distribution of observations
            bin_edges = np.percentile(reference_scores, np.linspace(0, 100, bins+1))
            # Ensure unique bin edges
            bin_edges = np.unique(bin_edges)
            # Add small buffer to the last bin edge to include the maximum value
            bin_edges[-1] += 1e-8
        else:
            bin_edges = np.asarray(bins)
        
        # Count observations in each bin
        ref_counts, _ = np.histogram(reference_scores, bins=bin_edges)
        recent_counts, _ = np.histogram(recent_scores, bins=bin_edges)
        
        # Calculate proportions for each bin
        ref_props = ref_counts / ref_counts.sum()
        recent_props = recent_counts / recent_counts.sum()
        
        # Add small constant to avoid division by zero or log(0)
        epsilon = 1e-10
        ref_props = np.maximum(ref_props, epsilon)
        recent_props = np.maximum(recent_props, epsilon)
        
        # Calculate PSI for each bin
        bin_psi = (recent_props - ref_props) * np.log(recent_props / ref_props)
        total_psi = np.sum(bin_psi)
        
        # Create bin details for visualization
        bin_details = []
        for i in range(len(bin_psi)):
            if i < len(bin_edges) - 1:
                bin_details.append({
                    'bin': i + 1,
                    'lower_bound': float(bin_edges[i]),
                    'upper_bound': float(bin_edges[i+1]),
                    'ref_count': int(ref_counts[i]),
                    'recent_count': int(recent_counts[i]),
                    'ref_prop': float(ref_props[i]),
                    'recent_prop': float(recent_props[i]),
                    'psi_contrib': float(bin_psi[i])
                })
        
        # Determine stability category
        if total_psi < 0.1:
            stability_category = "Stable"
        elif total_psi < 0.2:
            stability_category = "Moderately Shifted"
        else:
            stability_category = "Significantly Shifted"
        
        # Determine if test passes
        passed = total_psi < threshold
        
        # Create visualization data
        figure_data = {
            'x': [d['bin'] for d in bin_details],
            'y_ref': [d['ref_prop'] for d in bin_details],
            'y_recent': [d['recent_prop'] for d in bin_details],
            'bin_psi': [d['psi_contrib'] for d in bin_details],
            'bin_edges': [d['lower_bound'] for d in bin_details] + [bin_details[-1]['upper_bound']],
            'title': f'Population Stability Index (PSI = {total_psi:.4f}, {"Passed" if passed else "Failed"})',
            'xlabel': 'Score/PD Bins',
            'ylabel': 'Proportion of Population',
            'plot_type': 'psi',
            'annotations': [
                f'PSI: {total_psi:.4f}',
                f'Threshold: {threshold:.2f}',
                f'Category: {stability_category}',
                f'Reference: n={len(reference_scores)}',
                f'Recent: n={len(recent_scores)}'
            ],
        }
        
        return MetricResult(
            name=self.__class__.__name__,
            value=total_psi,
            threshold=threshold,
            passed=passed,
            details={
                'n_ref': len(reference_scores),
                'n_recent': len(recent_scores),
                'stability_category': stability_category,
                'bin_details': bin_details
            },
            figure_data=figure_data
        )


class CSI(BaseMetric):
    """
    Characteristic Stability Index (CSI) for monitoring input variable stability.
    
    CSI applies the PSI concept to individual input variables (characteristics) 
    in the model. It helps identify which variables have changed significantly 
    over time, potentially contributing to model performance degradation.
    
    CSI is calculated for each input variable by binning the values and comparing
    the distributions between reference and recent periods.
    
    CSI interpretation (typically same as PSI):
    - CSI < 0.1: No significant change
    - 0.1 ≤ CSI < 0.2: Moderate shift, requires investigation
    - CSI ≥ 0.2: Significant shift, potential issue with the variable
    
    References:
    ----------
    - Industry: Siddiqi, N. (2017). "Intelligent Credit Scoring: Building and Implementing 
      Better Credit Risk Scorecards," 2nd Edition, Wiley.
    - Industry: Anderson, R. (2007). "The Credit Scoring Toolkit: Theory and Practice for 
      Retail Credit Risk Management and Decision Automation," Oxford University Press.
    """
    
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="csi", config=config, config_path=config_path, **kwargs)
    
    def _compute_raw(self, reference_data: pd.DataFrame, recent_data: pd.DataFrame, 
                    variables: List[str] = None, n_bins: int = 10, **kwargs):
        """
        Calculate the Characteristic Stability Index for multiple variables.
        
        Parameters
        ----------
        reference_data : DataFrame
            Data from the reference period (development/validation)
        recent_data : DataFrame
            Data from the recent period
        variables : list of str, optional
            Variables to analyze. If None, all shared numeric columns are used.
        n_bins : int, optional
            Number of bins for each variable
            
        Returns
        -------
        MetricResult
            Object containing CSI values for each variable
        """
        # If no variables specified, use all shared numeric columns
        if variables is None:
            shared_cols = set(reference_data.columns) & set(recent_data.columns)
            variables = [col for col in shared_cols if np.issubdtype(reference_data[col].dtype, np.number)]
        
        # Get threshold from configuration
        threshold = self._get_param("threshold", default=0.1)
        
        # Calculate CSI for each variable
        csi_results = []
        for var in variables:
            if var not in reference_data.columns or var not in recent_data.columns:
                warnings.warn(f"Variable {var} not found in both datasets. Skipping.")
                continue
                
            # Get data for this variable
            ref_values = reference_data[var].dropna().values
            recent_values = recent_data[var].dropna().values
            
            if len(ref_values) == 0 or len(recent_values) == 0:
                warnings.warn(f"Variable {var} has no valid values in one of the datasets. Skipping.")
                continue
            
            # Create bins based on reference distribution
            try:
                bin_edges = np.percentile(ref_values, np.linspace(0, 100, n_bins+1))
                # Ensure unique bin edges
                bin_edges = np.unique(bin_edges)
                # Add small buffer to the last bin edge to include the maximum value
                bin_edges[-1] += 1e-8
                
                # Count observations in each bin
                ref_counts, _ = np.histogram(ref_values, bins=bin_edges)
                recent_counts, _ = np.histogram(recent_values, bins=bin_edges)
                
                # Calculate proportions for each bin
                ref_props = ref_counts / ref_counts.sum()
                recent_props = recent_counts / recent_counts.sum()
                
                # Add small constant to avoid division by zero or log(0)
                epsilon = 1e-10
                ref_props = np.maximum(ref_props, epsilon)
                recent_props = np.maximum(recent_props, epsilon)
                
                # Calculate CSI for each bin and total
                bin_csi = (recent_props - ref_props) * np.log(recent_props / ref_props)
                total_csi = np.sum(bin_csi)
                
                # Create bin details
                bin_details = []
                for i in range(len(bin_csi)):
                    bin_details.append({
                        'bin': i + 1,
                        'lower_bound': float(bin_edges[i]),
                        'upper_bound': float(bin_edges[i+1]),
                        'ref_count': int(ref_counts[i]),
                        'recent_count': int(recent_counts[i]),
                        'ref_prop': float(ref_props[i]),
                        'recent_prop': float(recent_props[i]),
                        'csi_contrib': float(bin_csi[i])
                    })
                
                # Determine stability category
                if total_csi < 0.1:
                    stability_category = "Stable"
                elif total_csi < 0.2:
                    stability_category = "Moderately Shifted"
                else:
                    stability_category = "Significantly Shifted"
                
                # Determine if variable passes
                passed = total_csi < threshold
                
                csi_results.append({
                    'variable': var,
                    'csi': float(total_csi),
                    'passed': passed,
                    'stability_category': stability_category,
                    'n_ref': len(ref_values),
                    'n_recent': len(recent_values),
                    'bin_details': bin_details
                })
            except Exception as e:
                warnings.warn(f"Error calculating CSI for variable {var}: {str(e)}")
        
        # Sort results by CSI value (descending)
        csi_results.sort(key=lambda x: x['csi'], reverse=True)
        
        # Check if any variables exceed threshold
        overall_passed = all(result['passed'] for result in csi_results)
        
        # Prepare data for visualization
        top_vars = [r['variable'] for r in csi_results[:min(10, len(csi_results))]]
        top_csi = [r['csi'] for r in csi_results[:min(10, len(csi_results))]]
        
        figure_data = {
            'x': top_vars,
            'y': top_csi,
            'threshold': threshold,
            'title': f'Characteristic Stability Index (CSI) - Top Variables',
            'xlabel': 'Variables',
            'ylabel': 'CSI Value',
            'plot_type': 'csi',
            'annotations': [
                f'Variables Analyzed: {len(csi_results)}',
                f'Variables Exceeding Threshold: {sum(1 for r in csi_results if not r["passed"])}',
                f'Threshold: {threshold:.2f}',
                f'Overall Result: {"Passed" if overall_passed else "Failed"}'
            ],
        }
        
        return MetricResult(
            name=self.__class__.__name__,
            value=max([r['csi'] for r in csi_results]) if csi_results else 0,
            threshold=threshold,
            passed=overall_passed,
            details={
                'variables_analyzed': len(csi_results),
                'variables_exceeding_threshold': sum(1 for r in csi_results if not r['passed']),
                'csi_results': csi_results
            },
            figure_data=figure_data
        )


class TemporalDriftDetection(BaseMetric):
    """
    Temporal Drift Detection for monitoring model stability over time.
    
    This metric analyzes trends in performance metrics (AUC, Gini, KS, etc.) over time
    to detect gradual deterioration. It applies statistical tests to determine if
    observed changes represent random fluctuations or significant drift.
    
    Methods:
    - Linear trend analysis using Mann-Kendall test
    - Change point detection using CUSUM
    - Moving window analysis
    
    References:
    ----------
    - Academic: Žliobaitė, I., Pechenizkiy, M., Gama, J. (2016). "An Overview of Concept Drift Applications," 
      Big Data Analysis: New Algorithms for a New Society, pp. 91-114.
    - Academic: Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., Bouchachia, A. (2014). 
      "A Survey on Concept Drift Adaptation," ACM Computing Surveys, 46(4), Article 44.
    """
    
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="drift", config=config, config_path=config_path, **kwargs)
    
    def _compute_raw(self, time_points: np.ndarray, metric_values: np.ndarray, 
                    metric_name: str = "Performance", **kwargs):
        """
        Detect temporal drift in model performance metrics.
        
        Parameters
        ----------
        time_points : array-like
            Time points (dates or numeric time indicators)
        metric_values : array-like
            Corresponding metric values (AUC, Gini, etc.)
        metric_name : str, optional
            Name of the metric being analyzed
            
        Returns
        -------
        MetricResult
            Object containing drift detection results
        """
        # Parameter validation
        time_points = np.asarray(time_points)
        metric_values = np.asarray(metric_values)
        
        if len(time_points) != len(metric_values):
            raise ValueError("time_points and metric_values must have the same length.")
        
        if len(time_points) < 4:
            warnings.warn("At least 4 time points are recommended for meaningful drift detection.")
        
        # Get confidence level from configuration
        confidence_level = self._get_param("confidence_level", default=0.95)
        alpha = 1 - confidence_level
        
        # 1. Mann-Kendall Trend Test
        # This non-parametric test determines if there's a monotonic trend in the time series
        try:
            trend, h, p_value, z = self._mann_kendall_test(metric_values, alpha)
            trend_detected = h == 1
            trend_direction = "Increasing" if z > 0 else "Decreasing" if z < 0 else "No trend"
        except Exception as e:
            warnings.warn(f"Error in Mann-Kendall test: {str(e)}")
            trend = "Error"
            trend_detected = False
            p_value = None
            z = None
            trend_direction = "Unknown"
        
        # 2. Linear regression for trend visualization
        try:
            # Convert to numeric time indices if they're not already
            time_indices = np.arange(len(time_points))
            
            # Simple linear regression
            slope, intercept, r_value, _, _ = stats.linregress(time_indices, metric_values)
            trend_line = intercept + slope * time_indices
            
            # Predict future value (one step ahead)
            future_value = intercept + slope * len(time_indices)
        except Exception as e:
            warnings.warn(f"Error in linear regression: {str(e)}")
            slope = intercept = r_value = future_value = None
            trend_line = np.full_like(metric_values, np.nan)
        
        # Determine if drift is concerning based on trend direction and metric type
        # For metrics where higher is better (AUC, Gini, etc.), decreasing trend is concerning
        # This is a simplification - in practice, would depend on the specific metric
        metrics_higher_better = ['auc', 'gini', 'ks', 'accuracy', 'ks_stat', 'pietra']
        if metric_name.lower() in metrics_higher_better:
            concerning_drift = trend_detected and z < 0  # Decreasing trend is concerning
        else:
            concerning_drift = trend_detected and z > 0  # Increasing trend is concerning
        
        # Prepare results
        trend_results = {
            'trend_detected': trend_detected,
            'trend_direction': trend_direction,
            'p_value': p_value,
            'z_statistic': z,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2 if r_value is not None else None,
            'future_prediction': future_value,
            'concerning_drift': concerning_drift
        }
        
        # Prepare visualization data
        figure_data = {
            'x': time_points.tolist() if isinstance(time_points[0], (int, float)) else list(range(len(time_points))),
            'y': metric_values.tolist(),
            'trend_line': trend_line.tolist() if trend_line is not None else [],
            'future_point': [len(time_points), future_value] if future_value is not None else [],
            'title': f'Temporal Drift Analysis: {metric_name}',
            'xlabel': 'Time',
            'ylabel': metric_name,
            'plot_type': 'drift',
            'annotations': [
                f'Trend: {trend_direction}',
                f'p-value: {p_value:.4f}' if p_value is not None else 'p-value: N/A',
                f'Concerning Drift: {"Yes" if concerning_drift else "No"}',
                f'Confidence Level: {confidence_level*100:.0f}%'
            ],
        }
        
        return MetricResult(
            name=self.__class__.__name__,
            value=1.0 if concerning_drift else 0.0,  # Binary indicator of concerning drift
            passed=not concerning_drift,  # Pass if no concerning drift
            details={
                'metric_name': metric_name,
                'n_time_points': len(time_points),
                'trend_results': trend_results,
                'time_points': time_points.tolist() if isinstance(time_points[0], (int, float)) else [str(t) for t in time_points],
                'metric_values': metric_values.tolist()
            },
            figure_data=figure_data
        )
    
    @staticmethod
    def _mann_kendall_test(x, alpha=0.05):
        """
        Non-parametric test for monotonic trend detection.
        
        Returns
        -------
        trend : str
            'increasing', 'decreasing', or 'no trend'
        h : int
            Hypothesis test result (1 = trend, 0 = no trend)
        p_value : float
            p-value of the test
        z : float
            z-statistic
        """
        n = len(x)
        s = 0
        for i in range(n-1):
            for j in range(i+1, n):
                s += np.sign(x[j] - x[i])
        
        # Calculate variance (with adjustment for ties)
        unique_x = np.unique(x)
        g = len(unique_x)
        
        if n == g:  # No ties
            var_s = n * (n - 1) * (2 * n + 5) / 18
        else:  # Ties exist
            tp = np.zeros(g)
            for i in range(g):
                tp[i] = sum(1 for val in x if val == unique_x[i])
            var_s = (n * (n - 1) * (2 * n + 5) - np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18
        
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # Two-tailed test
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        h = p < alpha
        
        if z > 0:
            trend = "increasing"
        elif z < 0:
            trend = "decreasing"
        else:
            trend = "no trend"
            
        return trend, h, p, z


class MigrationAnalysis(BaseMetric):
    """
    Analyze the migration of borrowers between rating grades over time.
    
    Migration analysis tracks how obligors move between rating grades from
    one period to another, helping identify patterns of rating volatility
    or systematic shifts in rating assignments.
    
    The metric provides migration matrices, stability ratios, and statistical
    tests for significant migration patterns.
    
    References:
    ----------
    - Regulatory: BCBS (2005). "Studies on validation of internal rating systems"
    - Academic: Jafry, Y., Schuermann, T. (2004). "Measurement, estimation and 
      comparison of credit migration matrices"
    """
    
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="migration", config=config, config_path=config_path, **kwargs)
    
    def _compute_raw(self, initial_ratings: np.ndarray, current_ratings: np.ndarray,
                    ids: np.ndarray = None, rating_scale: List = None, **kwargs):
        """
        Calculate migration matrix and stability statistics.
        
        Parameters
        ----------
        initial_ratings : array-like
            Ratings from the initial period
        current_ratings : array-like
            Ratings from the current period
        ids : array-like, optional
            Unique identifiers for each obligor to ensure proper matching
            If None, assumes initial_ratings and current_ratings are already aligned
        rating_scale : list, optional
            Ordered list of all possible rating grades
            If None, will be inferred from the data
            
        Returns
        -------
        MetricResult
            Object containing migration matrix and stability statistics
        """
        # Parameter validation
        initial_ratings = np.asarray(initial_ratings)
        current_ratings = np.asarray(current_ratings)
        
        if ids is not None:
            ids = np.asarray(ids)
            if len(ids) != len(initial_ratings) or len(ids) != len(current_ratings):
                raise ValueError("ids, initial_ratings, and current_ratings must have the same length.")
        elif len(initial_ratings) != len(current_ratings):
            raise ValueError("initial_ratings and current_ratings must have the same length if ids are not provided.")
        
        # Determine rating scale if not provided
        if rating_scale is None:
            rating_scale = sorted(list(set(np.concatenate([initial_ratings, current_ratings]))))
        
        n_ratings = len(rating_scale)
        rating_to_idx = {rating: i for i, rating in enumerate(rating_scale)}
        
        # Initialize migration matrix
        migration_matrix = np.zeros((n_ratings, n_ratings))
        
        # Count migrations
        for i in range(len(initial_ratings)):
            try:
                from_idx = rating_to_idx[initial_ratings[i]]
                to_idx = rating_to_idx[current_ratings[i]]
                migration_matrix[from_idx, to_idx] += 1
            except (KeyError, IndexError):
                warnings.warn(f"Rating value not in rating_scale: {initial_ratings[i]} or {current_ratings[i]}")
        
        # Calculate row sums (number of obligors in each initial rating)
        row_sums = migration_matrix.sum(axis=1)
        
        # Convert to probability matrix (normalize rows)
        with np.errstate(divide='ignore', invalid='ignore'):
            prob_matrix = np.zeros_like(migration_matrix, dtype=float)
            for i in range(n_ratings):
                if row_sums[i] > 0:
                    prob_matrix[i, :] = migration_matrix[i, :] / row_sums[i]
        
        # Calculate stability statistics
        
        # 1. Diagonal elements represent stability (same rating)
        diagonal_probs = np.diag(prob_matrix)
        
        # 2. Off-diagonal elements represent transitions
        off_diagonal_mask = ~np.eye(n_ratings, dtype=bool)
        
        # 3. Upgrades (transitions to better ratings - lower indices assuming better ratings have lower indices)
        # This assumes rating_scale is ordered from best to worst
        upgrades_mask = np.zeros((n_ratings, n_ratings), dtype=bool)
        downgrades_mask = np.zeros((n_ratings, n_ratings), dtype=bool)
        
        for i in range(n_ratings):
            for j in range(n_ratings):
                if i > j:  # Upgrade
                    upgrades_mask[i, j] = True
                elif i < j:  # Downgrade
                    downgrades_mask[i, j] = True
        
        # Calculate overall statistics
        total_obligors = migration_matrix.sum()
        stable_obligors = np.sum(np.diag(migration_matrix))
        upgraded_obligors = np.sum(migration_matrix * upgrades_mask)
        downgraded_obligors = np.sum(migration_matrix * downgrades_mask)
        
        stability_ratio = stable_obligors / total_obligors if total_obligors > 0 else 0
        upgrade_ratio = upgraded_obligors / total_obligors if total_obligors > 0 else 0
        downgrade_ratio = downgraded_obligors / total_obligors if total_obligors > 0 else 0
        
        # Calculate mobility indices
        # 1. Simple index: 1 - trace(P) / n
        mobility_simple = 1 - np.trace(prob_matrix) / n_ratings
        
        # 2. Shorrocks index: n - trace(P) / (n - 1)
        mobility_shorrocks = (n_ratings - np.trace(prob_matrix)) / (n_ratings - 1) if n_ratings > 1 else 0
        
        # Determine if stability is acceptable
        # Get threshold from configuration
        stability_threshold = self._get_param("stability_threshold", default=0.7)
        passed = stability_ratio >= stability_threshold
        
        # Prepare data for visualization
        figure_data = {
            'matrix': prob_matrix.tolist(),
            'rating_scale': rating_scale,
            'title': f'Rating Migration Matrix',
            'xlabel': 'Current Rating',
            'ylabel': 'Initial Rating',
            'plot_type': 'migration',
            'annotations': [
                f'Stability Ratio: {stability_ratio:.2f}',
                f'Upgrade Ratio: {upgrade_ratio:.2f}',
                f'Downgrade Ratio: {downgrade_ratio:.2f}',
                f'Mobility Index: {mobility_shorrocks:.2f}',
                f'Total Obligors: {int(total_obligors)}'
            ],
        }
        
        return MetricResult(
            name=self.__class__.__name__,
            value=stability_ratio,  # Stability ratio as primary metric
            threshold=stability_threshold,
            passed=passed,
            details={
                'migration_matrix': migration_matrix.tolist(),
                'probability_matrix': prob_matrix.tolist(),
                'rating_scale': rating_scale,
                'n_ratings': n_ratings,
                'total_obligors': int(total_obligors),
                'stable_obligors': int(stable_obligors),
                'upgraded_obligors': int(upgraded_obligors),
                'downgraded_obligors': int(downgraded_obligors),
                'stability_ratio': float(stability_ratio),
                'upgrade_ratio': float(upgrade_ratio),
                'downgrade_ratio': float(downgrade_ratio),
                'mobility_simple': float(mobility_simple),
                'mobility_shorrocks': float(mobility_shorrocks),
                'diagonal_probabilities': diagonal_probs.tolist()
            },
            figure_data=figure_data
        )


class RatingStabilityAnalysis(BaseMetric):
    """
    Analyze the stability of rating assignments over time.
    
    This metric evaluates whether ratings exhibit appropriate stability while
    still reflecting meaningful changes in credit quality. Excessively volatile
    ratings can lead to operational challenges, while ratings that never change
    may not be adequately capturing risk dynamics.
    
    The analysis includes transition rates, volatility measures, and
    assessments of whether rating changes are directionally aligned with 
    defaults or other credit events.
    
    References:
    ----------
    - Regulatory: EBA Guidelines on PD estimation, LGD estimation and treatment
      of defaulted exposures (2017)
    - Industry: Moody's Analytics (2013). "Rating System Stability"
    """
    
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="rating_stability", config=config, config_path=config_path, **kwargs)
    
    def _compute_raw(self, rating_time_series: List[np.ndarray], 
                    ids_time_series: List[np.ndarray] = None,
                    time_points: List = None, 
                    defaults: np.ndarray = None, 
                    default_ids: np.ndarray = None,
                    **kwargs):
        """
        Analyze rating stability across multiple time periods.
        
        Parameters
        ----------
        rating_time_series : list of arrays
            List of rating arrays, one for each time period
        ids_time_series : list of arrays, optional
            List of ID arrays corresponding to each time period's ratings
            If None, assumes ratings are already aligned across time periods
        time_points : list, optional
            Labels for each time period (e.g., dates)
            If None, will use sequential integers
        defaults : array, optional
            Binary indicator of default (1) or non-default (0) status at the end of the analysis period
        default_ids : array, optional
            IDs corresponding to the defaults array, for matching with rating IDs
            
        Returns
        -------
        MetricResult
            Object containing stability statistics and volatility measures
        """
        # Parameter validation
        if len(rating_time_series) < 2:
            raise ValueError("At least two time periods are required for stability analysis.")
            
        # Set up time points if not provided
        if time_points is None:
            time_points = list(range(len(rating_time_series)))
        
        # Get configuration parameters
        volatility_threshold = self._get_param("volatility_threshold", default=0.3)
        
        # Set up common IDs across time periods
        if ids_time_series is not None:
            # Find common IDs across all time periods
            common_ids = set(ids_time_series[0])
            for ids in ids_time_series[1:]:
                common_ids = common_ids.intersection(set(ids))
            
            # Filter to keep only common IDs
            filtered_ratings = []
            for i, (ratings, ids) in enumerate(zip(rating_time_series, ids_time_series)):
                id_map = {id_val: idx for idx, id_val in enumerate(ids)}
                period_ratings = [ratings[id_map[id_val]] for id_val in common_ids if id_val in id_map]
                filtered_ratings.append(np.array(period_ratings))
            
            rating_time_series = filtered_ratings
            common_ids = list(common_ids)
        else:
            # Assume ratings are already aligned
            common_ids = None
        
        # Check if we have any data left after filtering
        if any(len(ratings) == 0 for ratings in rating_time_series):
            warnings.warn("No common IDs found across all time periods.")
            return MetricResult(
                name=self.__class__.__name__,
                value=np.nan,
                details={'error': 'No common IDs found across all time periods.'}
            )
        
        # Calculate rating changes for each obligor across time periods
        n_obligors = len(rating_time_series[0])
        n_periods = len(rating_time_series)
        
        # Statistics for each obligor
        obligor_stats = []
        
        for i in range(n_obligors):
            # Extract rating history for this obligor
            rating_history = [ratings[i] for ratings in rating_time_series]
            
            # Calculate statistics
            n_changes = sum(1 for j in range(1, len(rating_history)) if rating_history[j] != rating_history[j-1])
            change_ratio = n_changes / (n_periods - 1) if n_periods > 1 else 0
            max_rating = max(rating_history)
            min_rating = min(rating_history)
            rating_range = max_rating - min_rating
            
            # Add to obligor stats
            obligor_stats.append({
                'id': common_ids[i] if common_ids else i,
                'rating_history': rating_history,
                'n_changes': n_changes,
                'change_ratio': change_ratio,
                'max_rating': max_rating,
                'min_rating': min_rating,
                'rating_range': rating_range
            })
        
        # Calculate overall statistics
        total_possible_changes = n_obligors * (n_periods - 1)
        total_changes = sum(stat['n_changes'] for stat in obligor_stats)
        overall_change_ratio = total_changes / total_possible_changes if total_possible_changes > 0 else 0
        
        # Calculate volatility
        mean_change_ratio = np.mean([stat['change_ratio'] for stat in obligor_stats])
        median_change_ratio = np.median([stat['change_ratio'] for stat in obligor_stats])
        mean_rating_range = np.mean([stat['rating_range'] for stat in obligor_stats])
        
        # Rating reversals (up then down or down then up)
        reversals = 0
        for stat in obligor_stats:
            history = stat['rating_history']
            for j in range(2, len(history)):
                if (history[j-2] < history[j-1] and history[j-1] > history[j]) or \
                   (history[j-2] > history[j-1] and history[j-1] < history[j]):
                    reversals += 1
        
        reversal_ratio = reversals / (n_obligors * (n_periods - 2)) if n_periods > 2 and n_obligors > 0 else 0
        
        # Determine if volatility is acceptable
        # Low volatility is good, but too low might indicate a non-responsive system
        passed = overall_change_ratio <= volatility_threshold
        
        # Prepare data for visualization
        # 1. Distribution of change ratios
        change_ratios = [stat['change_ratio'] for stat in obligor_stats]
        
        # 2. Change rates over time
        period_change_rates = []
        for j in range(1, n_periods):
            period_changes = sum(1 for i in range(n_obligors) 
                               if rating_time_series[j][i] != rating_time_series[j-1][i])
            period_change_rates.append(period_changes / n_obligors)
        
        figure_data = {
            'change_ratios': change_ratios,
            'period_change_rates': period_change_rates,
            'time_points': time_points[1:],  # Skip first time point for period change rates
            'title': f'Rating Stability Analysis',
            'xlabel': 'Time Period',
            'ylabel': 'Change Rate',
            'plot_type': 'rating_stability',
            'annotations': [
                f'Overall Change Ratio: {overall_change_ratio:.2f}',
                f'Mean Change Ratio: {mean_change_ratio:.2f}',
                f'Reversal Ratio: {reversal_ratio:.2f}',
                f'Mean Rating Range: {mean_rating_range:.2f}',
                f'Total Obligors: {n_obligors}'
            ],
        }
        
        return MetricResult(
            name=self.__class__.__name__,
            value=overall_change_ratio,  # Overall change ratio as primary metric
            threshold=volatility_threshold,
            passed=passed,
            details={
                'n_obligors': n_obligors,
                'n_periods': n_periods,
                'total_changes': total_changes,
                'overall_change_ratio': float(overall_change_ratio),
                'mean_change_ratio': float(mean_change_ratio),
                'median_change_ratio': float(median_change_ratio),
                'mean_rating_range': float(mean_rating_range),
                'reversals': reversals,
                'reversal_ratio': float(reversal_ratio),
                'period_change_rates': period_change_rates,
                'obligor_stats_summary': {
                    'n_obligors_with_no_changes': sum(1 for stat in obligor_stats if stat['n_changes'] == 0),
                    'n_obligors_with_one_change': sum(1 for stat in obligor_stats if stat['n_changes'] == 1),
                    'n_obligors_with_multiple_changes': sum(1 for stat in obligor_stats if stat['n_changes'] > 1)
                }
            },
            figure_data=figure_data
        )
