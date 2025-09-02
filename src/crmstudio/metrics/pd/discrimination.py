"""
Discrimination metrics for probability of default (PD) models.

This module provides metrics for assessing the discriminatory power of PD models,
including ROC-AUC, Gini, KS statistic, and various visualization tools.

"""

import math
from ...core.base import BaseMetric, MetricResult
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Tuple, Union

class AUC(BaseMetric):
    """
    AUC (Area Under the ROC Curve) metric.
    
    This metric calculates the area under the ROC curve, 
    which is a measure of the model's ability to distinguish between classes.
    
    While not expressed as a formal hypothesis test, AUC has a natural null value:
    - AUC = 0.5 indicates no discriminatory power (equivalent to random prediction)
    - AUC > 0.5 indicates positive discriminatory power
    - AUC < 0.5 indicates inverse discriminatory power (model predicts opposite of reality)
    
    In credit risk modeling, AUC values are typically evaluated against thresholds:
    - AUC < 0.6: Poor discrimination
    - 0.6 ≤ AUC < 0.7: Weak discrimination
    - 0.7 ≤ AUC < 0.8: Acceptable discrimination
    - 0.8 ≤ AUC < 0.9: Excellent discrimination
    - AUC ≥ 0.9: Outstanding discrimination
    
    References:
    ----------
    - Regulatory: Basel Committee on Banking Supervision (2005). "Studies on the Validation of Internal Rating Systems," 
      Working Paper No. 14, pp. 23-25.
    - Regulatory: European Banking Authority (2017). "Guidelines on PD estimation, LGD estimation and the treatment of defaulted exposures," 
      EBA/GL/2017/16, Section 5.3.3.
    - Academic: Engelmann, B., Hayden, E., Tasche, D. (2003). "Testing rating accuracy," 
      Risk, 16(1), 82-86.
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="curve", config=config, config_path=config_path, **kwargs)
        
    def _compute_raw(self, y_true = None, y_pred = None, **kwargs):
        """
        Calculate the AUC score.
        
        :param y_true: True binary labels in range {0, 1} or {-1, 1}.
        :param y_pred: Target scores, can either be probability estimates of the positive class, confidence values, or binary decisions.
        :return: AUC score as a MetricResult object.
        """
        if y_true is not None and y_pred is not None:
            auc_score = roc_auc_score(y_true, y_pred)
            return MetricResult(
                name=self.__class__.__name__, 
                value=auc_score, 
                details={
                    "n_obs": len(y_true), 
                    "n_defaults": int(np.sum(y_true))
                    }
                )
        else:
            raise ValueError("Both 'y_true' and 'y_pred' must be provided.")

class AUCDelta(BaseMetric):
    """
    Calculate the difference in AUC between two time periods following ECB methodology.
    
    The implementation follows ECB instructions for Model Validation:
    - Calculates test statistic S = (auc_init - auc_curr) / s
    - Computes p-value = 1 - Φ(S), where Φ is standard normal CDF
    - Variance calculation follows formula from ECB validation reporting guidelines
    
    Hypothesis test:
    - H0: Current AUC >= Initial AUC (model performance is maintained or improved)
    - H1: Current AUC < Initial AUC (model performance has deteriorated)
    
    The p-value represents the probability of observing the current AUC or lower, assuming 
    that the true AUC is equal to the initial AUC. A low p-value (below significance level, 
    typically 0.05) indicates significant deterioration in model performance.
    
    References:
    ----------
    - Regulatory: European Central Bank (2019). "Instructions for reporting the validation results of internal models," 
      Section 2.5.3.2. https://www.bankingsupervision.europa.eu/activities/internal_models/shared/pdf/instructions_validation_reporting_credit_risk.en.pdf
    - Academic: DeLong, E.R., DeLong, D.M., Clarke-Pearson, D.L. (1988). "Comparing the areas under two or more 
      correlated receiver operating characteristic curves: a nonparametric approach," Biometrics, 44(3), 837-845.
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="curve", config=config, config_path=config_path, **kwargs)
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
        return MetricResult(
            name=self.__class__.__name__, 
            value=p_value, 
            details=details
            )
      

class ROCCurve(BaseMetric):
    """
    Generate ROC curve coordinates with AUC statistic.
    
    The ROC curve plots the True Positive Rate against the False Positive Rate at various
    threshold settings. The Area Under the Curve (AUC) provides a single measure of the
    model's discriminatory power.
    
    While not a formal hypothesis test, the ROC curve can be interpreted relative to:
    - The diagonal line (AUC = 0.5) representing a random model with no discriminatory power
    - The top-left corner (AUC = 1.0) representing perfect discrimination
    
    For credit risk validation, the ROC curve helps visualize the tradeoff between
    correctly identifying defaults (sensitivity) versus incorrectly classifying
    non-defaults (1-specificity) across the full range of possible cutoff thresholds.
    
    References:
    ----------
    - Regulatory: Basel Committee on Banking Supervision (2005). "Studies on the Validation of Internal Rating Systems," 
      Working Paper No. 14, pp. 23-25.
    - Academic: Fawcett, T. (2006). "An introduction to ROC analysis," 
      Pattern Recognition Letters, 27(8), 861-874.
    - Regulatory: European Banking Authority (2017). "Guidelines on PD estimation, LGD estimation and the treatment of defaulted exposures," 
      EBA/GL/2017/16, Section 5.3.3.
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="curve", config=config, config_path=config_path, **kwargs)
        
    def _compute_raw(self, y_true = None, y_pred = None, **kwargs):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        auc_score = roc_auc_score(y_true, y_pred)
        return MetricResult(
                name=self.__class__.__name__, 
                value=auc_score, 
                figure_data={
                    "x": fpr.tolist(),  # Standardized coordinate naming
                    "y": tpr.tolist(),  # Standardized coordinate naming
                    "thresholds": thresholds.tolist(),
                    "title": f"ROC Curve (AUC = {auc_score:.3f})",
                    "xlabel": "False Positive Rate",
                    "ylabel": "True Positive Rate"
                }
            )

class PietraIndex(BaseMetric):
    """
    Calculate Pietra index (maximum deviation of ROC curve from diagonal).
    
    The Pietra index measures the maximum vertical distance between the ROC curve
    and the diagonal line. It represents the maximum achievable difference between
    the true positive rate and false positive rate for any threshold.
    
    Similar to the KS statistic, the Pietra index can be interpreted as:
    - 0: No discriminatory power (model is equivalent to random guessing)
    - 1: Perfect discrimination (model completely separates classes)
    
    Unlike formal hypothesis tests, the Pietra index does not have explicit H0/H1
    formulations, but provides a direct measure of maximum separation capability.
    In credit risk applications, higher values indicate better discrimination between
    defaulters and non-defaulters.
    
    References:
    ----------
    - Academic: Pietra, G. (1915). "On the relationship between concentration indices and variability measures," 
      Atti del Reale Istituto Veneto di Scienze, Lettere ed Arti, 74, 775-804.
    - Academic: Hand, D.J. (2012). "Measuring classifier performance: a coherent alternative to the area under the ROC curve," 
      Machine Learning, 77(1), 103-123.
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="curve", config=config, config_path=config_path, **kwargs)
        
    def _compute_raw(self, y_true = None, y_pred = None, **kwargs):
        if y_pred is not None and y_true is not None:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
        else:
            raise ValueError("Both 'y_true' and 'y_pred' must be provided.")
        diagonal_dist = np.abs(tpr - fpr) / np.sqrt(2)
        return MetricResult(
            name=self.__class__.__name__, 
            value=np.max(diagonal_dist), 
            details={
                "n_obs": len(y_true), 
                "n_defaults": int(np.sum(y_true))
            }
        )

class KSStat(BaseMetric):
    """
    Calculate Kolmogorov-Smirnov statistic.
    
    The KS statistic measures the maximum vertical distance between the cumulative 
    distribution functions (CDFs) of scores for defaulters and non-defaulters.
    
    While not typically formulated as a hypothesis test in credit scoring, 
    the KS statistic can be interpreted as:
    - 0: No separation between defaulters and non-defaulters (random model)
    - 1: Perfect separation between defaulters and non-defaulters
    
    In credit risk applications, the KS statistic is often used with these guidelines:
    - KS < 20%: Poor discrimination
    - 20% ≤ KS < 40%: Fair discrimination
    - 40% ≤ KS < 60%: Good discrimination
    - 60% ≤ KS < 80%: Very good discrimination
    - KS ≥ 80%: Excellent discrimination
    
    The KS statistic is useful for identifying the score threshold that maximizes
    separation between good and bad customers, making it valuable for cutoff setting.
    
    References:
    ----------
    - Regulatory: Oesterreichische Nationalbank (OeNB) (2004). "Rating Models and Validation," 
      Guidelines on Credit Risk Management, pp. 80-81.
    - Academic: Mays, E. (2004). "Credit Scoring for Risk Managers: The Handbook for Lenders," 
      Thomson/South-Western.
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="curve", config=config, config_path=config_path, **kwargs)
        
    def _compute_raw(self, y_true = None, y_pred = None, **kwargs):
        if y_pred is not None and y_true is not None:
            fpr, tpr, _ = roc_curve(y_true, y_pred)
        else:
            raise ValueError("Both 'y_true' and 'y_pred' must be provided.")
        ks_stat = np.max(np.abs(tpr - fpr))
        return MetricResult(
            name=self.__class__.__name__, 
            value=ks_stat, 
            details={
                "n_obs": len(y_true), 
                "n_defaults": int(np.sum(y_true))
            }
        )

class Gini(BaseMetric):
    """
    Calculate Gini coefficient (2*AUC - 1).
    
    The Gini coefficient is a transformation of the AUC, providing a measure of model
    discrimination equivalent to twice the area between the ROC curve and the diagonal.
    
    While not expressed as a formal hypothesis test, the Gini coefficient has natural
    interpretation points:
    - Gini = 0: No discriminatory power (equivalent to random prediction)
    - Gini = 1: Perfect discrimination (model completely separates classes)
    - Gini < 0: Inverse discriminatory power (predicts opposite of reality)
    
    In credit risk modeling, Gini values are typically evaluated against thresholds:
    - Gini < 20%: Poor discrimination
    - 20% ≤ Gini < 40%: Fair discrimination
    - 40% ≤ Gini < 60%: Good discrimination
    - 60% ≤ Gini < 80%: Very good discrimination
    - Gini ≥ 80%: Excellent discrimination
    
    The Gini coefficient is commonly used in credit scoring due to its intuitive
    interpretation as the ratio of the area between the CAP curve and the random model
    to the area between the perfect model and the random model.
    
    References:
    ----------
    - Regulatory: Basel Committee on Banking Supervision (2005). "Studies on validation of internal rating systems," 
      Working Paper No. 14, pp. 26-27.
    - Academic: Hand, D.J., Till, R.J. (2001). "A Simple Generalisation of the Area Under the ROC Curve for Multiple Class Classification Problems," 
      Machine Learning, 45, 171-186.
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="curve", config=config, config_path=config_path, **kwargs)
        
    def _compute_raw(self, y_true, y_pred, **kwargs):
        auc = roc_auc_score(y_true, y_pred)
        gini = 2 * auc - 1
        return MetricResult(
            name=self.__class__.__name__, 
            value=gini, 
            details={
                "n_obs": len(y_true), 
                "n_defaults": int(np.sum(y_true))
            }
        )

class GiniCI(BaseMetric):
    """
    Calculate Gini coefficient with confidence intervals.
    
    This metric extends the Gini coefficient by computing confidence intervals
    using DeLong's method for variance estimation of the AUC.
    
    The confidence intervals allow for hypothesis testing:
    - If the confidence interval includes 0: Cannot reject the null hypothesis that
      the model has no discriminatory power
    - If the confidence interval is entirely above 0: Can conclude that the model
      has positive discriminatory power
    
    The confidence interval also enables comparison between models:
    - If confidence intervals of two models overlap: Cannot conclude that one model
      performs significantly better than the other
    - If confidence intervals don't overlap: Can conclude that there is a significant
      difference in performance between the models
    
    References:
    ----------
    - Regulatory: Basel Committee on Banking Supervision (2005). "Studies on validation of internal rating systems," 
      Working Paper No. 14, pp. 26-27.
    - Academic: Hand, D.J., Till, R.J. (2001). "A Simple Generalisation of the Area Under the ROC Curve for Multiple Class Classification Problems," 
      Machine Learning, 45, 171-186.
    - Academic: DeLong, E.R., DeLong, D.M., Clarke-Pearson, D.L. (1988). "Comparing the areas under two or more 
      correlated receiver operating characteristic curves: a nonparametric approach," Biometrics, 44(3), 837-845.
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="curve", config=config, config_path=config_path, **kwargs)
        
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

        return MetricResult(
            name=self.__class__.__name__,
            value=gini,
            details={
                "ci_lower": ci_lower,
                "ci_upper": ci_upper
            }
        )


class CAPCurve(BaseMetric):
    """
    Generate Cumulative Accuracy Profile curve coordinates.
    
    In credit risk context, predictions (y_pred) often represent rating scores
    where multiple observations share the same score. This implementation:
    1. First sorts by predictions (ratings) in descending order
    2. Within each rating group, sorts by actual defaults (y_true)
    This ensures maximum discriminatory power within rating groups.
    
    The CAP curve is similar to the ROC curve but plots:
    - x-axis: Fraction of all borrowers (sorted by predicted PD)
    - y-axis: Fraction of all defaulters captured
    
    While not a formal hypothesis test, the CAP curve can be interpreted relative to:
    - The diagonal line representing a random model with no discriminatory power
    - The top-left curve representing a perfect model that captures all defaults
      with the minimum fraction of the population
    
    The Accuracy Ratio (AR) is calculated as the ratio of the area between the model
    curve and the random model to the area between the perfect model and the random model.
    This is equivalent to the Gini coefficient (2*AUC-1).
    
    In credit risk validation, the CAP curve is particularly useful for assessing the
    efficiency of risk screening, showing what percentage of total defaults can be
    captured by classifying a given percentage of all borrowers as "bad".
    
    References:
    ----------
    - Regulatory: Basel Committee on Banking Supervision (2005). "Studies on the Validation of Internal Rating Systems," 
      Working Paper No. 14, pp. 28-29.
    - Academic: Sobehart, J., Keenan, S., Stein, R. (2000). "Benchmarking Quantitative Default Risk Models: A Validation Methodology," 
      Moody's Rating Methodology.
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="curve", config=config, config_path=config_path, **kwargs)
        
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
        return MetricResult(
            name=self.__class__.__name__, 
            value=auc, 
            figure_data={
                "x": x.tolist(), 
                "y": y.tolist(),
                "ar": float(ar),
                "title": f"CAP Curve (AR = {ar:.3f})",
                "xlabel": "Fraction of Population",
                "ylabel": "Fraction of Defaults Captured",
                "n_defaults": int(total_pos),
                "n_total": len(y_true)
            }
        )

class CIER(BaseMetric):
    """
    Calculate Conditional Information Entropy Ratio (CIER).
    
    CIER measures how much uncertainty about the target variable is reduced by knowing
    the predictions. It's calculated as:
    CIER = 1 - H(Y|X) / H(Y)
    where H(Y|X) is conditional entropy and H(Y) is entropy of target variable.
    
    While not typically formulated as a formal hypothesis test, CIER has a natural
    interpretation scale:
    - CIER = 0: No reduction in uncertainty (model provides no information about defaults)
    - CIER = 1: Complete reduction in uncertainty (model perfectly predicts defaults)
    
    CIER is an information-theoretic measure that quantifies the reduction in uncertainty
    about defaults when using the model predictions. Higher values indicate better
    discriminatory power, with similar interpretation to R² in regression analysis.
    
    The metric is particularly useful when assessing models with non-monotonic relationships
    between inputs and outputs, as it captures more complex dependence structures than
    traditional metrics like AUC.
    
    References:
    ----------
    - Academic: Cover, T.M., Thomas, J.A. (2006). "Elements of Information Theory," 2nd Edition, 
      Wiley-Interscience.
    - Academic: Sharma, A., Paliwal, K.K. (2015). "Linear discriminant analysis for the small sample size problem: 
      an overview," International Journal of Machine Learning and Cybernetics, 6(3), 443-454.
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="distribution", config=config, config_path=config_path, **kwargs)
        
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
        return MetricResult(
            name=self.__class__.__name__, 
            value=cier, 
            details={
            }
        )

class KLDistance(BaseMetric):
    """
    Calculate Kullback-Leibler divergence between predicted and actual distributions.
    
    The KL divergence measures the difference between two probability distributions -
    in this case, the distribution of predicted probabilities and the distribution
    of actual outcomes.
    
    While not typically formulated as a hypothesis test, KL divergence has a natural
    interpretation:
    - KL = 0: The distributions are identical (perfect alignment)
    - KL > 0: The distributions differ, with larger values indicating greater divergence
    
    In credit risk modeling, KL divergence can be used to:
    1. Compare how well the predicted PD distribution matches the actual default distribution
    2. Assess whether PD estimates are systematically biased in certain regions
    3. Detect distribution shifts between development and application datasets
    
    Unlike distance metrics like RMSE, KL divergence is asymmetric and captures differences
    in the shape of distributions, making it useful for identifying systematic biases
    in model predictions.
    
    References:
    ----------
    - Academic: Kullback, S., Leibler, R.A. (1951). "On Information and Sufficiency," 
      Annals of Mathematical Statistics, 22(1), 79-86.
    - Regulatory: Financial Services Authority (2009). "Variable scalar approaches to estimating through the cycle PDs," 
      UK FSA.
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="distribution", config=config, config_path=config_path, **kwargs)
        
    def _compute_raw(self, y_true, y_pred, **kwargs):
        n_bins = self._get_param("n_bins", default = 10)
        hist_true, _ = np.histogram(y_true, bins=n_bins, density=True)
        hist_pred, _ = np.histogram(y_pred, bins=n_bins, density=True)
        
        # Add small constant to avoid division by zero
        eps = 1e-10
        hist_true = hist_true + eps
        hist_pred = hist_pred + eps
        
        kl_div = np.sum(hist_true * np.log(hist_true / hist_pred))
        return MetricResult(
            name=self.__class__.__name__, 
            value=kl_div, 
            details={
                "hist_true": hist_true,
                "hist_pred": hist_pred
            }
        )

class InformationValue(BaseMetric):
    """
    Calculate Information Value (IV) for the model predictions.
    
    Information Value quantifies the predictive power of a model by measuring
    the difference between the distribution of defaulters and non-defaulters
    across prediction bins. It's calculated as:
    IV = Σ (% non-defaulters - % defaulters) × ln(% non-defaulters / % defaulters)
    
    While not expressed as a formal hypothesis test, IV has established
    interpretation thresholds:
    - IV < 0.1: Weak predictive power
    - 0.1 ≤ IV < 0.3: Medium predictive power
    - 0.3 ≤ IV < 0.5: Strong predictive power
    - IV ≥ 0.5: Extremely strong predictive power
    
    In credit risk modeling, IV is particularly useful for:
    1. Comparing the discriminatory power of different models
    2. Identifying which variables contribute most to model performance
    3. Detecting potential overfit when development IV is much higher than validation IV
    
    The IV is closely related to the Kullback-Leibler divergence and provides a
    similar information-theoretic measure of separation between good and bad populations.
    
    References:
    ----------
    - Academic: Kullback, S., Leibler, R.A. (1951). "On Information and Sufficiency," 
      Annals of Mathematical Statistics, 22(1), 79-86.
    - Regulatory: Financial Services Authority (2009). "Variable scalar approaches to estimating through the cycle PDs," 
      UK FSA.
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="distribution", config=config, config_path=config_path, **kwargs)
        
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

        return MetricResult(
            name=self.__class__.__name__,
            value=iv_total,
            details={}
        )

class KendallsTau(BaseMetric):
    """
    Calculate Kendall's Tau correlation coefficient.
    
    Kendall's Tau measures the ordinal association between the predicted probabilities
    and the actual outcomes. It assesses how well the ranking of predictions matches
    the ranking of actual values.
    
    The metric can be interpreted as a hypothesis test:
    - H0: There is no association between predictions and actual outcomes (tau = 0)
    - H1: There is a positive association between predictions and actual outcomes (tau > 0)
    
    The significance of Kendall's Tau can be determined through its p-value, with small
    p-values indicating that the observed correlation is unlikely to have occurred by chance.
    
    In credit risk modeling, Kendall's Tau is particularly useful for:
    1. Assessing ordinal discrimination in models with many ties (e.g., rating systems)
    2. Providing a non-parametric measure of association that is robust to outliers
    3. Evaluating whether higher predicted PDs correspond to higher observed default rates
    
    Kendall's Tau ranges from -1 (perfect negative association) to 1 (perfect positive
    association), with 0 indicating no association.
    
    References:
    ----------
    - Regulatory: Basel Committee on Banking Supervision (2005). "Studies on the Validation of Internal Rating Systems," 
      Working Paper No. 14, pp. 27-28.
    - Academic: Kendall, M.G. (1938). "A New Measure of Rank Correlation," 
      Biometrika, 30(1/2), 81-93.
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="distribution", config=config, config_path=config_path, **kwargs)
        
    def _compute_raw(self, y_true, y_pred, **kwargs):
        tau, _ = stats.kendalltau(y_true, y_pred)
        return MetricResult(
            name=self.__class__.__name__,
            value=tau,
            details={}
        )

class SpearmansRho(BaseMetric):
    """
    Calculate Spearman's rank correlation coefficient.
    
    Spearman's Rho measures the monotonic relationship between predicted probabilities
    and actual outcomes by calculating the Pearson correlation between the rank values.
    
    The metric can be interpreted as a hypothesis test:
    - H0: There is no monotonic association between predictions and actual outcomes (rho = 0)
    - H1: There is a monotonic association between predictions and actual outcomes (rho ≠ 0)
    
    The significance of Spearman's Rho can be determined through its p-value, with small
    p-values indicating that the observed correlation is unlikely to have occurred by chance.
    
    In credit risk modeling, Spearman's Rho is valuable for:
    1. Assessing whether higher predicted PDs correspond to higher observed default rates
    2. Providing a non-parametric measure of association that doesn't assume linearity
    3. Comparing the rank ordering ability of different models
    
    Spearman's Rho ranges from -1 (perfect negative monotonic association) to 1 (perfect
    positive monotonic association), with 0 indicating no association.
    
    References:
    ----------
    - Regulatory: Basel Committee on Banking Supervision (2005). "Studies on the Validation of Internal Rating Systems," 
      Working Paper No. 14, pp. 27-28.
    - Academic: Spearman, C. (1904). "The Proof and Measurement of Association between Two Things," 
      The American Journal of Psychology, 15(1), 72-101.
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="distribution", config=config, config_path=config_path, **kwargs)
        
    def _compute_raw(self, y_true, y_pred, **kwargs):
        rho, _ = stats.spearmanr(y_true, y_pred)
        return MetricResult(
            name=self.__class__.__name__,
            value=rho,
            details={}
        )

class KSDistPlot(BaseMetric):
    """
    Calculates cumulative distribution function of defaulted and non-defaulted populations.
    Shows maximum separation point, i.e. KS statistic.
    
    The KS Distribution Plot visualizes the cumulative distributions of scores for
    defaulters and non-defaulters, highlighting the point of maximum separation (the KS statistic).
    
    While not typically formulated as a formal hypothesis test in credit scoring,
    the visualization and KS statistic together allow for interpretation:
    - Overlapping distributions indicate poor discrimination
    - Widely separated distributions indicate strong discrimination
    - The point of maximum separation helps identify optimal cutoff thresholds
    
    In credit risk applications, the KS Distribution Plot provides several insights:
    1. The overall separation between good and bad populations across the score range
    2. The specific score value where maximum discrimination occurs
    3. Regions where the model has stronger or weaker separating power
    
    Unlike the ROC curve which plots sensitivity vs. (1-specificity), the KS plot shows
    the entire cumulative distributions, making it more intuitive for setting cutoffs
    based on population percentiles.
    
    References:
    ----------
    - Regulatory: Oesterreichische Nationalbank (OeNB) (2004). "Rating Models and Validation," 
      Guidelines on Credit Risk Management, pp. 80-81.
    - Academic: Mays, E. (2004). "Credit Scoring for Risk Managers: The Handbook for Lenders," 
      Thomson/South-Western.
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="distribution", config=config, config_path=config_path, **kwargs)
        
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
        return MetricResult(
                name=self.__class__.__name__, 
                value=float(ks_stat), 
                figure_data={
                    "x": thresholds.tolist(), #thresholds
                    "y": cdf_def.tolist(), #cdf_defaulted
                    "x_ref": thresholds.tolist(), #thresholds
                    "y_ref": cdf_nondef.tolist(), #cdf_non_defaulted
                    "actual_label": "CDF defaulted",
                    "ref_label": "CDF nondefaulted",
                    "title": f"KS Distribution Plot (KS = {ks_stat:.3f})",
                    "xlabel": "Score",
                    "ylabel": "Cumulative Proportion",
                    "n_obs": len(y_true),
                    "n_defaults": int(np.sum(y_true)),
                    "ks_threshold": float(ks_threshold) #ks_threshold - it is not used after changes?
                }
            )

class ScoreHistogram(BaseMetric):
    """
    Calculates 2 histograms of scores, one per each value of y_true,
    i.e. histogram of scores for defaulted and non-defaulted population.
    
    The Score Histogram provides a visual representation of the distribution of
    scores/predicted PDs for defaulters and non-defaulters, allowing for assessment
    of separation between these populations.
    
    While not a hypothesis test, the visualization provides valuable insights:
    - Well-separated histograms indicate good discriminatory power
    - Overlapping histograms suggest poor discrimination
    - The shape of distributions can reveal model biases or limitations
    
    In credit risk applications, the Score Histogram helps:
    1. Visualize the degree of overlap between good and bad populations
    2. Identify potential score ranges where misclassification is more likely
    3. Assess whether the score distribution follows expected patterns
    4. Detect anomalies such as excessive clustering at certain score values
    
    A good discriminatory model will show clear separation between the two histograms,
    with defaulters concentrated in the higher PD/score region and non-defaulters
    in the lower PD/score region.
    
    References:
    ----------
    - Regulatory: Basel Committee on Banking Supervision (2005). "Studies on the Validation of Internal Rating Systems," 
      Working Paper No. 14, pp. 33-34.
    - Regulatory: European Banking Authority (2017). "Guidelines on PD estimation, LGD estimation and the treatment of defaulted exposures," 
      EBA/GL/2017/16, Section 5.3.4.
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="distribution", config=config, config_path=config_path, **kwargs)
        
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
        return MetricResult(
            name=self.__class__.__name__, 
            value=np.nan, 
            figure_data={
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
        )

class PDLiftPlot(BaseMetric):
    """
    Implements Lift Plotting.

    Shows the lift curve: for each quantile (e.g., decile) of the sorted population (by predicted score),
    computes the ratio of the observed default rate in that quantile to the overall default rate.
    The random model is a horizontal line at lift=1.
    
    While not a formal hypothesis test, the Lift Plot provides a clear way to assess:
    - How much better the model performs than random assignment (lift > 1)
    - Where in the score distribution the model provides the greatest improvement
    - The practical business value of using the model for targeting
    
    In credit risk applications, the Lift Plot helps answer questions like:
    1. "If we target the top 20% highest-risk customers, how many times more defaults
        will we capture compared to random selection?"
    2. "At what population percentile does the model's discriminatory power diminish?"
    3. "How does the default concentration vary across different score ranges?"
    
    A good model will show high lift in the highest-risk segments (left side of the plot),
    gradually decreasing toward 1.0 as more of the population is included.
    
    References:
    ----------
    - Academic: Berry, M.J., Linoff, G.S. (2004). "Data Mining Techniques: For Marketing, Sales, and Customer Relationship Management," 
      2nd Edition, Wiley Publishing.
    - Regulatory: Office of the Comptroller of the Currency (OCC) (2011). "Supervisory Guidance on Model Risk Management," 
      SR Letter 11-7, Appendix A.
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="distribution", config=config, config_path=config_path, **kwargs)
        
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
        return MetricResult(
            name=self.__class__.__name__, 
            value=np.nan, 
            figure_data={
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
        )
    
class PDGainPlot(BaseMetric):
    """
    Implements Gain Plotting.

    Shows cumulative percentage of capture bads (or goods) as you move down the sorted population (usually sorted by predicted PD/score).
    It goes from the worst scores to the best, e.g. shows:
    With top 20% population we caputure 76% defaults.
    
    It looks really similar to CAP.
    
    While not a formal hypothesis test, the Gain Plot provides key insights:
    - How efficiently the model identifies defaults compared to random selection
    - What percentage of all defaults can be captured by targeting a specific 
      percentage of the population
    - The incremental value of expanding coverage beyond certain thresholds
    
    In credit risk applications, the Gain Chart helps:
    1. Set cutoffs based on business objectives (e.g., "capture at least 80% of defaults")
    2. Quantify the efficiency of risk screening at different population coverage levels
    3. Compare multiple models based on their default capture rates at specified coverage points
    
    The ideal model would capture 100% of defaults in a small portion of the population,
    while a random model would show a diagonal line (e.g., targeting 20% of population
    randomly would capture 20% of defaults).
    
    References:
    ----------
    - Regulatory: Basel Committee on Banking Supervision (2005). "Studies on the Validation of Internal Rating Systems," 
      Working Paper No. 14, pp. 28-29.
    - Academic: Sobehart, J., Keenan, S., Stein, R. (2000). "Benchmarking Quantitative Default Risk Models: A Validation Methodology," 
      Moody's Rating Methodology.
    """
    def __init__(self, model_name: str, config=None, config_path=None, **kwargs):
        super().__init__(model_name, metric_type="distribution", config=config, config_path=config_path, **kwargs)
        
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
        return MetricResult(
            name=self.__class__.__name__, 
            value=capture_20, 
            figure_data={
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
        )
