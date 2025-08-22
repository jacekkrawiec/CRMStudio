from ..core.base import BaseMetric, MetricResult
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Tuple, Dict, Any

class AUC(BaseMetric):
    """
    AUC (Area Under the ROC Curve) metric.
    
    This metric calculates the area under the ROC curve, which is a measure of the model's ability to distinguish between classes.
    """
    def _compute_raw(self, y_true, y_pred, **kwargs):
        """
        Calculate the AUC score.
        
        :param y_true: True binary labels in range {0, 1} or {-1, 1}.
        :param y_pred: Target scores, can either be probability estimates of the positive class, confidence values, or binary decisions.
        :return: AUC score as a MetricResult object.
        """
        auc_score = roc_auc_score(y_true, y_pred)
        return auc_score

class AUCDelta(BaseMetric):
    """
    Calculate the difference in AUC between two time periods following ECB methodology.
    
    The implementation follows ECB instructions for Model Validation:
    - Calculates test statistic S = (auc_init - auc_curr) / s
    - Computes p-value = 1 - Φ(S), where Φ is standard normal CDF
    - Variance calculation follows formula from ECB validation reporting guidelines
    
    Reference: 
    https://www.bankingsupervision.europa.eu/activities/internal_models/shared/pdf/instructions_validation_reporting_credit_risk.en.pdf
    """
    def _compute_raw(self, y_true, y_pred, y_true_prev=None, y_pred_prev=None, **kwargs):
        if y_true_prev is None or y_pred_prev is None:
            raise ValueError("Previous period data required for AUC delta calculation")
        
        # Calculate AUCs
        auc_curr = roc_auc_score(y_true, y_pred)
        auc_init = roc_auc_score(y_true_prev, y_pred_prev)
        delta = auc_curr - auc_init
        
        # Calculate variance components for current period
        n1_curr = np.sum(y_true) # |A|
        n0_curr = len(y_true) - n1_curr # |B|
        q1_curr = auc_curr / 2
        q2_curr = 2 * auc_curr ** 2 / (1 + auc_curr)
        var_curr = (auc_curr * (1 - auc_curr) + 
                   (n1_curr - 1) * (q1_curr - auc_curr ** 2) + 
                   (n0_curr - 1) * (q2_curr - auc_curr ** 2)) / (n1_curr * n0_curr)

        # Calculate standard error and test statistic
        s = np.sqrt(var_curr)
        S = delta / s
        
        # Calculate p-value
        p_value = 1 - stats.norm.cdf(S)
        
        return {
            "delta": delta,
            "test_statistic": S,
            "p_value": p_value,
            "auc_current": auc_curr,
            "auc_initial": auc_init,
            "std_error": s
        }

class ROCCurve(BaseMetric):
    """
    Generate ROC curve coordinates.
    """
    def _compute_raw(self, y_true, y_pred, **kwargs):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()}

class PietraIndex(BaseMetric):
    """
    Calculate Pietra index (maximum deviation of ROC curve from diagonal).
    """
    def _compute_raw(self, y_true, y_pred, **kwargs):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        diagonal_dist = np.abs(tpr - fpr) / np.sqrt(2)
        return np.max(diagonal_dist)

class KSStat(BaseMetric):
    """
    Calculate Kolmogorov-Smirnov statistic.
    """
    def _compute_raw(self, y_true, y_pred, **kwargs):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        ks_stat = np.max(np.abs(tpr - fpr))
        return ks_stat

class CAPCurve(BaseMetric):
    """
    Generate Cumulative Accuracy Profile curve coordinates.
    """
    def _compute_raw(self, y_true, y_pred, **kwargs):
        sorted_indices = np.argsort(y_pred)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        total_pos = np.sum(y_true)
        cum_pos = np.cumsum(y_true_sorted)
        
        x = np.linspace(0, 1, len(y_true))
        y = cum_pos / total_pos
        
        return {"x": x.tolist(), "y": y.tolist()}

class Gini(BaseMetric):
    """
    Calculate Gini coefficient (2*AUC - 1).
    """
    def _compute_raw(self, y_true, y_pred, **kwargs):
        auc = roc_auc_score(y_true, y_pred)
        gini = 2 * auc - 1
        return gini

class GiniCI(BaseMetric):
    """
    Calculate Gini coefficient with confidence intervals.
    """
    def _compute_raw(self, y_true, y_pred, **kwargs):
        confidence = self.metric_config.get("params", {}).get("confidence", 0.95)
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

class CIER(BaseMetric):
    """
    Calculate Conditional Information Entropy Ratio (CIER).
    
    CIER measures how much uncertainty about the target variable is reduced by knowing
    the predictions. It's calculated as:
    CIER = 1 - H(Y|X) / H(Y)
    where H(Y|X) is conditional entropy and H(Y) is entropy of target variable.
    """
    def _compute_raw(self, y_true, y_pred, **kwargs):
        n_bins = self.metric_config.get("params", {}).get("n_bins", 10)
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

class KLDistance(BaseMetric):
    """
    Calculate Kullback-Leibler divergence between predicted and actual distributions.
    """
    def _compute_raw(self, y_true, y_pred, **kwargs):
        n_bins = self.metric_config.get("params", {}).get("n_bins", 10)
        hist_true, _ = np.histogram(y_true, bins=n_bins, density=True)
        hist_pred, _ = np.histogram(y_pred, bins=n_bins, density=True)
        
        # Add small constant to avoid division by zero
        eps = 1e-10
        hist_true = hist_true + eps
        hist_pred = hist_pred + eps
        
        kl_div = np.sum(hist_true * np.log(hist_true / hist_pred))
        return kl_div

class InformationValue(BaseMetric):
    """
    Calculate Information Value (IV) for the model predictions.
    """
    def _compute_raw(self, y_true, y_pred, **kwargs):
        n_bins = self.metric_config.get("params", {}).get("n_bins", 10)
        # Create bins based on predictions
        bins_pred = np.linspace(0, 1, n_bins + 1)
        bins_indices = np.digitize(y_pred, bins_pred)
        
        iv_total = 0
        for i in range(1, n_bins + 1):
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

class KendallsTau(BaseMetric):
    """
    Calculate Kendall's Tau correlation coefficient.
    """
    def _compute_raw(self, y_true, y_pred, **kwargs):
        tau, _ = stats.kendalltau(y_true, y_pred)
        return tau

class AccuracyRatio(BaseMetric):
    """
    Calculate Accuracy Ratio (equivalent to Gini coefficient).
    """
    def _compute_raw(self, y_true, y_pred, **kwargs):
        auc = roc_auc_score(y_true, y_pred)
        ar = 2 * auc - 1
        return ar

class SpearmansRho(BaseMetric):
    """
    Calculate Spearman's rank correlation coefficient.
    """
    def _compute_raw(self, y_true, y_pred, **kwargs):
        rho, _ = stats.spearmanr(y_true, y_pred)
        return rho
