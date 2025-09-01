# crmstudio/src/crmstudio/metrics/__init__.py

"""
Metrics package for credit risk model validation.

This package provides a collection of metrics for validating credit risk models,
with a focus on PD, LGD, and EAD/CCF metrics.
"""

# Import from the reorganized PD metrics structure
from .pd import (
    # Discrimination metrics
    AUC, AUCDelta, ROCCurve, PietraIndex, KSStat, Gini, GiniCI, 
    CAPCurve, CIER, KLDistance, InformationValue, KendallsTau, 
    SpearmansRho, KSDistPlot, ScoreHistogram, PDLiftPlot, PDGainPlot,
    
    # Calibration metrics
    HosmerLemeshow, CalibrationCurve, BrierScore,
    ExpectedCalibrationError, PDCalibrationStats
)

# Import from other metrics modules
# from .lgd_metrics import (
#     # Future LGD metrics would be imported here
# )

# from .stability_metrics import (
#     # Future generic stability metrics would be imported here
# )

METRIC_REGISTRY = {
    # Discrimination metrics
    "AUC": AUC,
    "AUCDelta": AUCDelta,
    "ROCCurve": ROCCurve,
    "PietraIndex": PietraIndex,
    "KSStat": KSStat,
    "Gini": Gini,
    "GiniCI": GiniCI,
    "CAPCurve": CAPCurve,
    "CIER": CIER,
    "KLDistance": KLDistance,
    "InformationValue": InformationValue,
    "KendallsTau": KendallsTau,
    "SpearmansRho": SpearmansRho,
    "KSDistPlot": KSDistPlot,
    "ScoreHistogram": ScoreHistogram,
    "PDLiftPlot": PDLiftPlot,
    "PDGainPlot": PDGainPlot,
    
    # Calibration metrics
    "HosmerLemeshow": HosmerLemeshow,
    "CalibrationCurve": CalibrationCurve,
    "BrierScore": BrierScore,
    "ExpectedCalibrationError": ExpectedCalibrationError,
    "PDCalibrationStats": PDCalibrationStats,
}