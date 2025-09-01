"""
Probability of Default (PD) metrics for model validation.

This package contains metrics for evaluating PD model performance, organized by:
- discrimination.py: Metrics for assessing model's ability to distinguish between good and bad cases
- calibration.py: Metrics for assessing accuracy of probability predictions
- stability.py (future): Metrics for assessing model stability over time
"""

from .discrimination import (
    AUC, AUCDelta, ROCCurve, PietraIndex, KSStat, Gini, GiniCI, 
    CAPCurve, CIER, KLDistance, InformationValue, KendallsTau, 
    SpearmansRho, KSDistPlot, ScoreHistogram, PDLiftPlot, PDGainPlot
)

from .calibration import (
    HosmerLemeshow, CalibrationCurve, BrierScore,
    ExpectedCalibrationError, PDCalibrationStats
)

# Will be imported in the future when implemented
# from .stability import (
#     PSI, CSI, TemporalDriftDetection
# )

# Export all classes
__all__ = [
    # Discrimination metrics
    'AUC', 'AUCDelta', 'ROCCurve', 'PietraIndex', 'KSStat', 'Gini', 'GiniCI',
    'CAPCurve', 'CIER', 'KLDistance', 'InformationValue', 'KendallsTau',
    'SpearmansRho', 'KSDistPlot', 'ScoreHistogram', 'PDLiftPlot', 'PDGainPlot',
    
    # Calibration metrics
    'HosmerLemeshow', 'CalibrationCurve', 'BrierScore',
    'ExpectedCalibrationError', 'PDCalibrationStats',
    
    # Stability metrics - to be added in future
    # 'PSI', 'CSI', 'TemporalDriftDetection'
]
