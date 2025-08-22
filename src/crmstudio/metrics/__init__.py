# crmstudio/src/crmstudio/metrics/__init__.py

from .pd_metrics import AUC

METRIC_REGISTRY = {
    "AUC": AUC,
}