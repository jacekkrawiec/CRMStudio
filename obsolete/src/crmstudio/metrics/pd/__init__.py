"""
Probability of Default (PD) metrics for model validation.

This package contains metrics for evaluating PD model performance, organized by:
- discrimination.py: Metrics for assessing model's ability to distinguish between good and bad cases
- calibration.py: Metrics for assessing accuracy of probability predictions, heterogeneity testing, and concentration analysis
- stability.py: Metrics for assessing model stability over time and rating transitions
"""

from .discrimination import (
    AUC, AUCDelta, ROCCurve, PietraIndex, KSStat, Gini, GiniCI, 
    CAPCurve, CIER, KLDistance, InformationValue, KendallsTau, 
    SpearmansRho, KSDistPlot, ScoreHistogram, PDLiftPlot, PDGainPlot
)

from .calibration import (
    HosmerLemeshow, CalibrationCurve, BrierScore, BinomialTest, 
    NormalTest, JeffreysTest, ExpectedCalibrationError,
    RatingHeterogeneityTest, HerfindahlIndex
)

from .stability import (
    PSI, CSI, TemporalDriftDetection, MigrationAnalysis, RatingStabilityAnalysis
)

# Export all classes
__all__ = [
    # Discrimination metrics
    'AUC', 'AUCDelta', 'ROCCurve', 'PietraIndex', 'KSStat', 'Gini', 'GiniCI',
    'CAPCurve', 'CIER', 'KLDistance', 'InformationValue', 'KendallsTau',
    'SpearmansRho', 'KSDistPlot', 'ScoreHistogram', 'PDLiftPlot', 'PDGainPlot',
    
    # Calibration metrics
    'HosmerLemeshow', 'CalibrationCurve', 'BrierScore', 'ExpectedCalibrationError',
    'BinomialTest', 'NormalTest', 'JeffreysTest', 'RatingHeterogeneityTest',
    'HerfindahlIndex',

    # Stability metrics
    'PSI', 'CSI', 'TemporalDriftDetection', 'MigrationAnalysis', 'RatingStabilityAnalysis'
]
