import math
from ..core.base import BaseMetric, CurveMetric, DistributionAssociationMetric, FigureResult
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Tuple, Union

class AUC(CurveMetric):
    """
    AUC (Area Under the ROC Curve) metric.
    
    This metric calculates the area under the ROC curve, 
    which is a measure of the model's ability to distinguish between classes.
    """
    def _compute_raw(self, fr: FigureResult = None, y_true = None, y_pred = None, **kwargs):
        """
        Calculate the AUC score.
        
        :param y_true: True binary labels in range {0, 1} or {-1, 1}.
        :param y_pred: Target scores, can either be probability estimates of the positive class, confidence values, or binary decisions.
        :return: AUC score as a MetricResult object.
        """
        if fr is not None:
            fpr = np.array(fr['x'])
            tpr = np.array(fr['y'])
            # Calculate AUC using trapezoidal rule 
            auc_score = np.trapz(tpr, fpr)
            return auc_score, {"n_obs": len(y_true), "n_defaults": int(np.sum(y_true))}
        elif y_true is not None and y_pred is not None:
            auc_score = roc_auc_score(y_true, y_pred)
            return auc_score, {"n_obs": len(y_true), "n_defaults": int(np.sum(y_true))}
        else:
            raise ValueError("Either 'fr' (FigureResult) or both 'y_true' and 'y_pred' must be provided.")

class AUCDelta(CurveMetric):
    """
    Calculate the difference in AUC between two time periods following ECB methodology.
    
    The implementation follows ECB instructions for Model Validation:
    - Calculates test statistic S = (auc_init - auc_curr) / s
    - Computes p-value = 1 - Φ(S), where Φ is standard normal CDF
    - Variance calculation follows formula from ECB validation reporting guidelines
    
    Reference: 
    https://www.bankingsupervision.europa.eu/activities/internal_models/shared/pdf/instructions_validation_reporting_credit_risk.en.pdf
    """
    # ------- Core ECB Annex 1 helper -------
    @staticmethod
    def _rowwise_u_components(def_scores: np.ndarray, ndef_scores_sorted: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        """
        For each defaulter score 'a' compute:
            - rowU_a = count_b[ r(a) < r(b) ] + 0.5 * count_b[ r(a) == r(b) ]
            - V10_a = rowU_a / |B|
        Returns (rowU vector over A, V10 vector over A)
        """
        ndef = len(ndef_scores_sorted)
        rowU = np.array([np.sum(def_scores[i] > ndef_scores_sorted) + 0.5 * np.sum(def_scores[i] == ndef_scores_sorted) for i in range(len(def_scores))])
        V10 = rowU / ndef if ndef > 0 else np.zeros_like(rowU)
        return rowU, V10
    
    @staticmethod
    def _colwise_u_components(ndef_scores: np.ndarray, def_scores_sorted: np.ndarray) -> np.ndarray:
        """
        For each non-defaulter score 'b' compute:
            - colU_b = count_a[ r(a) < r(b) ] + 0.5 * count_a[ r(a) == r(b) ]
            - V01_b = colU_b / |A|
        Returns V01 vector over B
        """
        ndef = len(def_scores_sorted)
        colU = np.array([np.sum(ndef_scores[i] > def_scores_sorted) + 0.5 * np.sum(ndef_scores[i] == def_scores_sorted) for i in range(len(ndef_scores))])
        V01 = colU / ndef if ndef > 0 else np.zeros_like(colU)
        return V01

    @staticmethod
    def _unbiased_variance(x: np.ndarray) -> float:
        """
        Unbiased variance estimator
        """
        n = len(x)
        if n <= 1:
            return 0.0
        return float(np.var(x, ddof=1))

    @staticmethod
    def _auc_and_variance(y_true: np.ndarray, y_pred: np.ndarray, higher_better: bool) -> Tuple[float, float, int, int]:
        """
        Calculate AUC and its variance based on ECB's validation instruction.
        """
        # sanitize
        mask = np.isfinite(y_pred) & np.isfinite(y_true) #checks for NaN and Inf
        y = y_true[mask].astype(int)
        s = y_pred[mask].astype(float)

        if not higher_better:
            s = -s

        # Separate scores into defaulters and non-defaulters
        A = s[y == 1] # defaulters
        B = s[y == 0] # non-defaulters
        
        nA = len(A)
        nB = len(B)
        
        if nA == 0 or nB == 0:
            raise ValueError("AUC requires at least one defaulted (y_true = 1) and one non-defaulter (y_true = 0).")
        
        # Sort scores
        B_sorted = np.sort(B)
        A_sorted = np.sort(A)
        
        # Compute U components
        rowU_A, V10 = AUCDelta._rowwise_u_components(A, B_sorted)
        V01 = AUCDelta._colwise_u_components(B, A_sorted)
        
        # Calculate AUC
        auc = float(np.sum(rowU_A)) / (nA * nB)
        
        # Calculate variance
        var_V10 = AUCDelta._unbiased_variance(V10)
        var_V01 = AUCDelta._unbiased_variance(V01)
        
        var_auc = (var_V10 / nA) + (var_V01 / nB)
        
        return auc, var_auc, nA, nB

    def _compute_raw(self, y_true, y_pred, **kwargs):
        auc_curr, s2, nA, nB = self._auc_and_variance(
            y_true=np.asarray(y_true), 
            y_pred=np.asarray(y_pred),
            higher_better = self._get_param("score_higher_is_better", default=True)
        )
        auc_init = self._get_param("initial_auc", default = None)
        alpha = self._get_param("alpha", default = 0.05)
        
        if auc_init is None:
            raise ValueError("Initial AUC (auc_init) must be provided in metric params.")
        
        s = math.sqrt(max(s2,0.0)) 
        diff = auc_init - auc_curr

        if s == 0:
            S = math.inf if diff > 0 else -math.inf if diff < 0 else 0.0
        else:
            S = diff / s

        p_value = 1 - stats.norm.cdf(S)

        details = {
            "auc_curr": auc_curr, 
            "auc_init": auc_init,
            "variance": s2,
            "nA": nA, 
            "nB": nB, 
            "S": S, 
            "p_value": p_value,
            "decision": "pass" if p_value > alpha else "fail"
        }
        return float(p_value), details   
      

class ROCCurve(CurveMetric):
    """
    Generate ROC curve coordinates with AUC statistic.
    """
    def _compute_raw(self, y_true = None, y_pred = None, **kwargs):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        auc_score = roc_auc_score(y_true, y_pred)
        
        return {
            "x": fpr.tolist(),  # Standardized coordinate naming
            "y": tpr.tolist(),  # Standardized coordinate naming
            "thresholds": thresholds.tolist(),
            "auc": float(auc_score),
            "title": f"ROC Curve (AUC = {auc_score:.3f})",
            "xlabel": "False Positive Rate",
            "ylabel": "True Positive Rate"
        }

class PietraIndex(CurveMetric):
    """
    Calculate Pietra index (maximum deviation of ROC curve from diagonal).
    """
    def _compute_raw(self, fr: FigureResult = None, y_true = None, y_pred = None, **kwargs):
        if fr is not None:
            fpr = np.asarray(fr['x'])
            tpr = np.asarray(fr['y'])
        elif y_pred is not None and y_true is not None:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
        else:
            raise ValueError("Either 'fr' (FigureResult) or both 'y_true' and 'y_pred' must be provided.")
        diagonal_dist = np.abs(tpr - fpr) / np.sqrt(2)
        return np.max(diagonal_dist)

class KSStat(CurveMetric):
    """
    Calculate Kolmogorov-Smirnov statistic.
    """
    def _compute_raw(self, fr: FigureResult = None, y_true = None, y_pred = None, **kwargs):
        if fr is not None:
            fpr = np.asarray(fr['x'])
            tpr = np.asarray(fr['y'])
        elif y_pred is not None and y_true is not None:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
        else:
            raise ValueError("Either 'fr' (FigureResult) or both 'y_true' and 'y_pred' must be provided.")
        ks_stat = np.max(np.abs(tpr - fpr))
        return ks_stat
    
class Gini(CurveMetric):
    """
    Calculate Gini coefficient (2*AUC - 1).
    """
    def _compute_raw(self, y_true, y_pred, **kwargs):
        auc = roc_auc_score(y_true, y_pred)
        gini = 2 * auc - 1
        return gini

class GiniCI(CurveMetric):
    """
    Calculate Gini coefficient with confidence intervals.
    """
    def _compute_raw(self, y_true, y_pred, **kwargs):
        confidence = self._get_param("confidence", default = 0.95)
        auc = roc_auc_score(y_true, y_pred)
        gini = 2 * auc - 1
        
        # Calculate standard error using DeLong's method
        n1 = np.sum(y_true)
        n2 = len(y_true) - n1
        q1 = auc / 2
        q2 = 2 * auc ** 2 / (1 + auc)
        se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n2 - 1) * (q2 - auc ** 2)) / (n1 * n2))
        
        # Calculate confidence intervals
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        ci_lower = max(0, gini - z * 2 * se)
        ci_upper = min(1, gini + z * 2 * se)
        
        return {"gini": gini, "ci_lower": ci_lower, "ci_upper": ci_upper}


class CAPCurve(CurveMetric):
    """
    Generate Cumulative Accuracy Profile curve coordinates.
    
    In credit risk context, predictions (y_pred) often represent rating scores
    where multiple observations share the same score. This implementation:
    1. First sorts by predictions (ratings) in descending order
    2. Within each rating group, sorts by actual defaults (y_true)
    This ensures maximum discriminatory power within rating groups.
    """
    def _compute_raw(self, y_true, y_pred, **kwargs):
        # Convert inputs to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Create lexicographic sort key: first by pred (descending), then by true (descending)
        # This ensures that within each pred value, defaulters come first
        sorted_indices = np.lexsort((-y_true, -y_pred))
        
        y_true_sorted = y_true[sorted_indices]
        total_pos = np.sum(y_true)
        cum_pos = np.cumsum(y_true_sorted)
        
        # Generate x-coordinates (percentage of population)
        x = np.linspace(0, 1, len(y_true)+1)
        # Generate y-coordinates (percentage of captured defaults)
        y = np.insert(cum_pos / total_pos, 0, 0)
        
        # Calculate AR (Accuracy Ratio) which is equivalent to Gini coefficient
        # AR = 2*AUC - 1
        auc = roc_auc_score(y_true, y_pred)
        ar = 2 * auc - 1
    
        return {
            "x": x.tolist(), 
            "y": y.tolist(),
            "ar": float(ar),
            "title": f"CAP Curve (AR = {ar:.3f})",
            "xlabel": "Fraction of Population",
            "ylabel": "Fraction of Defaults Captured",
            "n_defaults": int(total_pos),
            "n_total": len(y_true)
        }

class CIER(DistributionAssociationMetric):
    """
    Calculate Conditional Information Entropy Ratio (CIER).
    
    CIER measures how much uncertainty about the target variable is reduced by knowing
    the predictions. It's calculated as:
    CIER = 1 - H(Y|X) / H(Y)
    where H(Y|X) is conditional entropy and H(Y) is entropy of target variable.
    """
    def _compute_raw(self, y_true, y_pred, **kwargs):
        n_bins = self._get_param("n_bins", default = 10)
        # Calculate entropy of target variable H(Y)
        p_y1 = np.mean(y_true)
        p_y0 = 1 - p_y1
        h_y = -p_y1 * np.log2(p_y1 + 1e-10) - p_y0 * np.log2(p_y0 + 1e-10)
        
        # Calculate conditional entropy H(Y|X)
        bins_pred = np.linspace(0, 1, n_bins + 1)
        bins_indices = np.digitize(y_pred, bins_pred)
        
        h_y_x = 0
        for i in range(1, n_bins + 1):
            mask = bins_indices == i
            if np.any(mask):
                # Probability of this bin
                p_x = np.mean(mask)
                
                # Probability of Y=1 in this bin
                p_y1_x = np.mean(y_true[mask])
                p_y0_x = 1 - p_y1_x
                
                # Contribution to conditional entropy
                h_yx = -p_y1_x * np.log2(p_y1_x + 1e-10) - p_y0_x * np.log2(p_y0_x + 1e-10)
                h_y_x += p_x * h_yx
        
        # Calculate CIER
        cier = 1 - h_y_x / (h_y + 1e-10)
        return cier

class KLDistance(DistributionAssociationMetric):
    """
    Calculate Kullback-Leibler divergence between predicted and actual distributions.
    """
    def _compute_raw(self, y_true, y_pred, **kwargs):
        n_bins = self._get_param("n_bins", default = 10)
        hist_true, _ = np.histogram(y_true, bins=n_bins, density=True)
        hist_pred, _ = np.histogram(y_pred, bins=n_bins, density=True)
        
        # Add small constant to avoid division by zero
        eps = 1e-10
        hist_true = hist_true + eps
        hist_pred = hist_pred + eps
        
        kl_div = np.sum(hist_true * np.log(hist_true / hist_pred))
        return kl_div

class InformationValue(DistributionAssociationMetric):
    """
    Calculate Information Value (IV) for the model predictions.
    """
    def _compute_raw(self, y_true, y_pred, **kwargs):
        n_bins = self._get_param("n_bins", default = 10)
        
        # Convert predictions to numpy array
        y_pred = np.asarray(y_pred)
        
        # Determine if input is already binned (discrete)
        unique_values = np.unique(y_pred)
        threshold = max(0.1 * len(y_true), 20)  # 10% of length or 20, whichever is larger
        is_discrete = len(unique_values) <= threshold
        
        if is_discrete:
            # For discrete ratings, use unique values as bins
            unique_ratings = np.unique(y_pred)
            bins_indices = np.zeros_like(y_pred, dtype=int)
            for i, rating in enumerate(unique_ratings, 1):
                bins_indices[y_pred == rating] = i
        else:
            # For continuous predictions, create bins between 0 and 1
            bins_pred = np.linspace(0, 1, n_bins + 1)
            bins_indices = np.digitize(y_pred, bins_pred)
        
        iv_total = 0
        n_bins_actual = len(np.unique(bins_indices))
        
        for i in range(1, n_bins_actual + 1):
            mask = bins_indices == i
            if np.any(mask):
                good = np.sum(y_true[mask] == 0)
                bad = np.sum(y_true[mask] == 1)
                
                # Add small constant to avoid division by zero
                good = good + 1e-10
                bad = bad + 1e-10
                
                good_rate = good / np.sum(y_true == 0)
                bad_rate = bad / np.sum(y_true == 1)
                
                iv = (good_rate - bad_rate) * np.log(good_rate / bad_rate)
                iv_total += iv
                
        return iv_total

class KendallsTau(DistributionAssociationMetric):
    """
    Calculate Kendall's Tau correlation coefficient.
    """
    def _compute_raw(self, y_true, y_pred, **kwargs):
        tau, _ = stats.kendalltau(y_true, y_pred)
        return tau

class SpearmansRho(DistributionAssociationMetric):
    """
    Calculate Spearman's rank correlation coefficient.
    """
    def _compute_raw(self, y_true, y_pred, **kwargs):
        rho, _ = stats.spearmanr(y_true, y_pred)
        return rho

class KSDistPlot(DistributionAssociationMetric):
    """
    Calculates cumulative distribution function of defaulted and non-defaulted populations.
    Shows maximum separation point, i.e. KS statistic.
    """
    def _compute_raw(self, y_true, y_pred, **kwargs):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Separate scores for defaulted and non-defaulted
        scores_def = y_pred[y_true == 1]
        scores_nondef = y_pred[y_true == 0]

        # Combine all scores for common binning
        all_scores = np.concatenate([scores_def, scores_nondef])
        # Use unique sorted scores as thresholds for stepwise CDF
        thresholds = np.unique(all_scores)
        # Add min/max for completeness
        thresholds = np.concatenate(([np.min(all_scores)], thresholds, [np.max(all_scores)]))

        # Compute empirical CDFs
        cdf_def = np.searchsorted(np.sort(scores_def), thresholds, side='right') / max(len(scores_def), 1)
        cdf_nondef = np.searchsorted(np.sort(scores_nondef), thresholds, side='right') / max(len(scores_nondef), 1)

        # KS statistic and location
        ks_stat = np.max(np.abs(cdf_def - cdf_nondef))
        ks_idx = np.argmax(np.abs(cdf_def - cdf_nondef))
        ks_threshold = thresholds[ks_idx]

        return {
            "x": thresholds.tolist(), #thresholds
            "y": cdf_def.tolist(), #cdf_defaulted
            "x_ref": thresholds.tolist(), #thresholds
            "y_ref": cdf_nondef.tolist(), #cdf_non_defaulted
            "actual_label": "CDF defaulted",
            "ref_label": "CDF nondefaulted",
            "value": float(ks_stat), #ks_stat
            "title": f"KS Distribution Plot (KS = {ks_stat:.3f})",
            "xlabel": "Score",
            "ylabel": "Cumulative Proportion",
            "n_obs": len(y_true),
            "n_defaults": int(np.sum(y_true)),
            "ks_threshold": float(ks_threshold) #ks_threshold - it is not used after changes?
        }

class ScoreHistogram(DistributionAssociationMetric):
    """
    Calculates 2 histograms of scores, one per each value of y_true,
    i.e. histogram of scores for defaulted and non-defaulted population.
    """
    def _compute_raw(self, y_true, y_pred, **kwargs):
        n_bins = self._get_param("n_bins", default=10)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Defaulted (y_true == 1)
        scores_def = y_pred[y_true == 1]
        # Non-defaulted (y_true == 0)
        scores_nondef = y_pred[y_true == 0]

        # Use same bin edges for both histograms
        bin_edges = np.linspace(np.min(y_pred), np.max(y_pred), n_bins + 1)

        hist_def, _ = np.histogram(scores_def, bins=bin_edges)
        hist_nondef, _ = np.histogram(scores_nondef, bins=bin_edges)

        default_rate = np.mean(y_true) * 100
        
        return {
            "x_def": bin_edges[:-1].tolist(), # bin_edges
            "y_def": hist_def.tolist(), # hist_defaulted
            "x_ndef": bin_edges[:-1].tolist(), # bin_edges - not present before changes
            "y_ndef": hist_nondef.tolist(), # hist_non_defaulted
            "bin_edges": bin_edges.tolist(), # bin_edges - not present before changes
            "title": f"Score Distribution (Default Rate: {default_rate:.1f}%)", # title
            "xlabel": "Score", # xlabel
            "ylabel": "Count", # ylabel
            "n_defaults": len(scores_def), # n_defaults
            "n_non_defaults": len(scores_nondef), # n_non_defaults
            "n_obs": len(y_true) # n_obs
        }

class PDLiftPlot(DistributionAssociationMetric):
    """
    Implements Lift Plotting.

    Shows the lift curve: for each quantile (e.g., decile) of the sorted population (by predicted score),
    computes the ratio of the observed default rate in that quantile to the overall default rate.
    The random model is a horizontal line at lift=1.
    """
    def _compute_raw(self, y_true, y_pred, **kwargs):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Sort by predicted score descending (worst to best)
        sorted_idx = np.argsort(-y_pred)
        y_true_sorted = y_true[sorted_idx]
        
        # Create percentile points (1 to 100%)
        percentiles = np.linspace(1, 100, 100)  # 100 points for 1% to 100% inclusive

        # Calculate lift for each percentile
        lift = []
        overall_rate = np.mean(y_true)
        
        for pct in percentiles:
            # Convert percentage to index
            end_idx = int(np.ceil(pct * len(y_true) / 100))
            if end_idx == 0:
                lift.append(0)  # No data point yet
                continue
                
            # Calculate default rate for this percentile
            bin_true = y_true_sorted[:end_idx]
            bin_rate = np.mean(bin_true)
            lift_value = bin_rate / overall_rate if overall_rate > 0 else np.nan
            lift.append(lift_value)

        max_lift = max(lift)
        max_pct = percentiles[lift.index(max_lift)]
        
        return {
            "x": percentiles.tolist(),  # standardized x coordinate
            "y": lift,                  # standardized y coordinate
            "x_ref": percentiles.tolist(),  # reference line x (same as x)
            "y_ref": [1] * len(percentiles),  # reference line y (lift = 1)
            "value": max_lift,  # max lift value
            "title": f"Lift Curve (Max Lift = {max_lift:.2f}x at {max_pct:.0f}%)",
            "xlabel": "Population Percentile",
            "ylabel": "Lift (Relative Default Rate)",
            "use_percentage_ticks": True,
            "n_obs": len(y_true),
            "n_defaults": int(np.sum(y_true))
        }
    
class PDGainPlot(DistributionAssociationMetric):
    """
    Implements Gain Plotting.

    Shows cumulative percentage of capture bads (or goods) as you move down the sorted population (usually sorted by predicted PD/score).
    It goes from the worst scores to the best, e.g. shows:
    With top 20% population we caputure 76% defaults.
    
    It looks really similar to CAP.
    """
    def _compute_raw(self, y_true, y_pred, **kwargs):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Sort by predicted score descending (worst to best)
        sorted_idx = np.argsort(-y_pred)
        y_true_sorted = y_true[sorted_idx]
        
        # Create percentile points (0 to 100%)
        percentiles = np.linspace(0, 100, 101)  # 101 points for 0% to 100% inclusive
        
        # Calculate cumulative defaults for each percentile
        total_defaults = np.sum(y_true)
        gains = []
        
        for pct in percentiles:
            # Convert percentage to index
            end_idx = int(np.ceil(pct * len(y_true) / 100))
            if end_idx == 0:
                gains.append(0)  # No data point yet
                continue
                
            # Calculate cumulative defaults up to this percentile
            cum_defaults = np.sum(y_true_sorted[:end_idx])
            gain = (cum_defaults / total_defaults) * 100 if total_defaults > 0 else 0
            gains.append(gain)

        # Calculate capture rate at 20%
        capture_20 = gains[20]
        
        return {
            "x": percentiles.tolist(),  # standardized x coordinate
            "y": gains,                 # standardized y coordinate
            "x_ref": percentiles.tolist(),  # reference line x (same as x)
            "y_ref": percentiles.tolist(),  # Diagonal Reference Line
            "value": capture_20,  # capture rate at 20%
            "title": f"Gain Chart (Captures {capture_20:.1f}% defaults in top 20% population)",
            "xlabel": "Population Percentile",
            "ylabel": "Cumulative % of Defaults Captured",
            "use_percentage_ticks": True,
            "use_percentage_ticks_y": True,
            "xlim": [0, 100],
            "ylim": [0, 100],
            "n_defaults": int(total_defaults),
            "n_obs": len(y_true)
        }
