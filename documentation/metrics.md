## Sources
- [ ] OeNB: https://www.oenb.at/dam/jcr:1db13877-21a0-40f8-b46c-d8448f162794/rating_models_tcm16-22933.pdf
- [ ] ECB_MV_INSTRUCTION 
- [ ] BIS: https://www.bis.org/publ/bcbs_wp14.pdf
- [ ] EBA: (https://www.eba.europa.eu/sites/default/files/documents/10180/2033363/6b062012-45d6-4655-af04-801d26493ed0/Guidelines%20on%20PD%20and%20LGD%20estimation%20%28EBA-GL-2017-16%29.pdf)
- [ ] MATLAB: https://www.mathworks.com/help/risk/regression.modelcalibration.html

## Data Quality
- [ ] Representativeness (Source: EBA)
- [ ] Stability (data drift)
- [ ] Outlier detection
- [ ] Completeness - e.g. number of missings
- [ ] Uniqueness - No of duplicates


## PD

# Calibration
- [x] Jeffreys test (source: ECB_MV_INSTRUCTION)
- [x] Brier Score (Source: OeNB)
- [x] Reliability Diagrams (Source: OeNB)
- [x] Normal Calibration Test (Source: OeNB)
- [x] Binomial Calibration Test (Source: OeNB)
- [x] Hosmer-Lameshow
 
# Discrimination
- [x] AUC (source: ECB_MV_INSTRUCTION)
- [x] Current AUC vs. AUC at inital validation/development (source: ECB_MV_INSTRUCTION)
- [x] ROC Curve (Source: OeNB)
- [x] Pietra Index (Source: OeNB)
- [x] KS_stat (Source: OeNB)
- [x] CAP Curve/Powercurve (Source: OeNB)
- [x] Gini (Source: OeNB)
- [x] Gini/AUC Confidence intervals (Source: OeNB)
- [x] ROC Curve (Source: OeNB)
- [x] Conditional Information Entropy Ratio (Source: OeNB)
- [ ] Kullback-Leibler distance (Source: BIS)
- [x] Information Value (Source: BIS)
- [x] Kendall's Tau/SommersD (Source: BIS)
- [x] Accuracy Ratio
- [x] Spearman's Rho

# Stability
 - [ ] Customer migrations (MWB) (source: ECB_MV_INSTRUCTION)
 - [ ] Stability of the migration matrix (source: ECB_MV_INSTRUCTION)
 - [ ] Concentration in rating grades (source: ECB_MV_INSTRUCTION)
 - [ ] PSI
 - [ ] Characteristic Stability Index
 - [ ] Jensen-Shannon divergence
 - [ ] ANOVA/Kruskal-Wallis test

## LGD performing

# Calibration
- [ ] T-test (source: ECB_MV_INSTRUCTION)
- [ ] R_squared (source: MATLAB)
- [ ] RMSE (source: MATLAB)
- [ ] Correlation (source: MATLAB)
- [ ] Sample mean error (source: MATLAB)
- [ ] Scatter PLOT (scource: MATLAB)
- [ ] MSE
- [ ] U-Mann Whitney test

# Discrimination
- [ ] gAUC (source: ECB_MV_INSTRUCTION)
- [ ] current gAUC vs. gAUC at initial validation/development (source: ECB_MV_INSTRUCTION)
- [ ] Spearman's Rho
- [ ] Valume Under the ROC Surface (VUS)
- [ ] Lorenz Curve
- [ ] Somers'D
- [ ] Kendall's Tau

# Stability
 - [ ] PSI (source: ECB_MV_INSTRUCTION)

## Expected Loss Best Estimate
# Calibration
- [ ] t-test (source: ECB_MV_INSTRUCTION)

## LGD in-default
# Calibration
- [ ] t-test (source: ECB_MV_INSTRUCTION)

## CCF

# Calibration
- [ ] t-test (CCF) (source: ECB_MV_INSTRUCTION)
- [ ] t-test (EAD) (source: ECB_MV_INSTRUCTION)

# Discrimination
- [ ] gAUC (source: ECB_MV_INSTRUCTION)
- [ ] current gAUC vs. gAUC at initial validation/development (source: ECB_MV_INSTRUCTION)

# Stability
- [ ] PSI (source: ECB_MV_INSTRUCTION)


## Copilot context

- Priorities: We're aiming to cover whole credit risk models' monitoring/validation toolkit, with all tests, reporting and viz
- Use cases: I'd like it to be flexible, base scenario is regulatory reporting (and minimum tests' coverege must be aligned with regulatory req) but ultimately I want to be also a useful business tool
- Reg requirments: We start with AIRB requirements, in the European context, so I want to be aligned with ECB/EBA requirements, but also consider local specificity from British/Polish regulators.
- Data volumnes/frequency: I don't expect dataset larger then several million rows; frequency up to monthly 
- Integration: this will be used for automated reporting via batch processing, i.e. all portfolios monitored at once with respective reports generated. No specific output format is needed, I'd prefer LaTeX

## Code refactoring
- We have too much inconsistency in the way data is created and transferred to transformators/plotters; we should keep one format of data returned by metrics
- We should have single plotting mechanism