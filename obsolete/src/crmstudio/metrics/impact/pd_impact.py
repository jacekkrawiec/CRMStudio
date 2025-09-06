"""This module implements several metrics for PD impact analysis."""
from typing import Optional, Dict, Any, List, Union
import numpy as np
import math

class PDImpact:
    """This class implements several metrics for PD impact analysis."""
    def __init__(self, 
                lgd: float = 0.45,  # Regulatory LGD default value
                m: float = 2.5,     # Regulatory maturity default value
                correlation_formula: str = 'basel',  # Use Basel correlation formula
                conf_level: float = 0.999,  # 99.9% confidence level for Basel
                scaling_factor: float = 1.06, # Basel scaling factor
                asset_class: str = 'mortgage',  # Default asset class
                sme_threshold: float = 50e6,  # SME threshold in EUR
                config: Optional[Dict] = None  # Additional configuration options
            ):
        """
        Initialize the PDImpact calculator with Basel parameters.
        
        Parameters
        ----------
        lgd : float
            Loss Given Default value (as a decimal)
        m : float
            Effective maturity in years
        correlation_formula : str
            Formula to use for asset correlation ('basel', 'asrf', 'custom')
        scaling_factor : float
            Regulatory scaling factor (1.06 for Basel)
        conf_level : float
            Confidence level for VaR calculation (0.999 for Basel)
        asset_class : str
            Asset class ('corporate', 'retail', 'sme', etc.)
        sme_threshold : float
            Threshold for SME treatment in EUR
        config : Dict, optional
            Additional configuration parameters
        """
        self.lgd = lgd
        self.m = m
        self.correlation_formula = correlation_formula
        self.conf_level = conf_level
        self.scaling_factor = scaling_factor
        self.asset_class = asset_class
        self.sme_threshold = sme_threshold
        self.config = config or {}
        
        # Normal inverse of confidence level (N^-1(0.999) ≈ 3.09)
        self.normal_inverse_conf = np.sqrt(2) * math.erfinv(2 * self.conf_level - 1)
    
    def calculate_risk_weight(self, pd_values: Union[float, np.ndarray], 
                            exposure_values: Optional[np.ndarray] = None,
                            lgd: Optional[float] = None, 
                            maturity: Optional[float] = None, 
                            asset_class: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate risk weights using the Basel IRB formula.
        
        Parameters
        ----------
        pd_values : float or np.ndarray
            Single PD value or array of PD values (as decimals)
        exposure_values : np.ndarray, optional
            Array of exposure values (EAD)
        lgd : float, optional
            LGD value to override default
        maturity : float, optional
            Maturity value to override default
        asset_class : str, optional
            Asset class to override default
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'risk_weights': Array of risk weights
            - 'rwa': Total RWA (if exposures provided)
            - 'avg_risk_weight': Exposure-weighted average risk weight
            - 'capital_requirement': Required capital
        """
        # Use provided parameters or defaults
        lgd_value = lgd if lgd is not None else self.lgd
        maturity_value = maturity if maturity is not None else self.m
        asset_class_value = asset_class if asset_class is not None else self.asset_class
        
        # Convert single PD value to array if needed
        if isinstance(pd_values, (float, int)):
            pd_array = np.array([pd_values])
        else:
            pd_array = np.array(pd_values)
        
        # Ensure PD is within regulatory bounds (0.03% to 100%)
        pd_array = np.clip(pd_array, 0.0003, 1.0)
        
        # Initialize arrays for results
        risk_weights = np.zeros_like(pd_array, dtype=float)
        capital_requirements = np.zeros_like(pd_array, dtype=float)
        
        # Calculate risk weight for each PD value
        for i, pd_value in enumerate(pd_array):
            # Step 1: Calculate correlation
            correlation = self._calculate_correlation(pd_value, asset_class_value)
            
            # Step 2: Calculate capital requirement
            capital_req = self._calculate_capital_requirement(pd_value, lgd_value, correlation)
            
            # Step 3: Apply maturity adjustment for non-retail exposures
            if asset_class_value not in ['mortgage', 'retail', 'qrr', 'other_retail']:
                maturity_adjustment = self._calculate_maturity_adjustment(pd_value, maturity_value)
                capital_req *= maturity_adjustment
            
            # Step 4: Apply scaling factor
            capital_req *= self.scaling_factor
            
            # Step 5: Convert to risk weight (capital * 12.5)
            risk_weight = capital_req * 12.5
            
            # Store results
            risk_weights[i] = risk_weight
            capital_requirements[i] = capital_req
        
        # Prepare results dictionary
        results = {
            'risk_weights': risk_weights,
            'capital_requirements': capital_requirements
        }
        
        # Calculate exposure-weighted metrics if exposures are provided
        if exposure_values is not None:
            if len(exposure_values) != len(pd_array):
                raise ValueError("Length of exposure_values must match length of pd_values")
            
            exposure_array = np.array(exposure_values)
            
            # Calculate RWA
            rwa_array = risk_weights * exposure_array
            total_rwa = np.sum(rwa_array)
            
            # Calculate exposure-weighted average risk weight
            total_exposure = np.sum(exposure_array)
            avg_risk_weight = total_rwa / total_exposure if total_exposure > 0 else 0
            
            # Add to results
            results.update({
                'rwa': total_rwa,
                'rwa_by_exposure': rwa_array,
                'avg_risk_weight': avg_risk_weight,
                'total_capital_requirement': total_rwa / 12.5
            })
        
        # For a single PD value, simplify the output
        if isinstance(pd_values, (float, int)):
            results['risk_weight'] = results['risk_weights'][0]
            results['capital_requirement'] = results['capital_requirements'][0]
        
        return results
    
    def calculate_deficiency_impact(self, current_pd: np.ndarray, 
                                   corrected_pd: np.ndarray,
                                   exposures: np.ndarray,
                                   deficiency_type: str = 'calibration',
                                   **kwargs) -> Dict[str, Any]:
        """
        Calculate the impact of model deficiencies on RWA.
        
        Parameters
        ----------
        current_pd : np.ndarray
            Current model PD predictions
        corrected_pd : np.ndarray
            PD values after correction of deficiency
        exposures : np.ndarray
            Exposure values (EAD)
        deficiency_type : str
            Type of deficiency ('calibration', 'discrimination', 'both')
        **kwargs : dict
            Additional parameters for specific deficiency calculations
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'current_rwa': RWA with current model
            - 'corrected_rwa': RWA with corrected model
            - 'rwa_difference': Absolute RWA difference
            - 'rwa_difference_pct': Percentage difference in RWA
            - 'capital_impact': Impact on capital requirements
            - 'detailed_impacts': Breakdown by rating grade/segment if available
        """
        return self
    
    def calculate_model_change_impact(self, current_model_pd: np.ndarray,
                                     new_model_pd: np.ndarray,
                                     exposures: np.ndarray,
                                     segment_info: Optional[np.ndarray] = None,
                                     **kwargs) -> Dict[str, Any]:
        """
        Calculate the impact of changing from current to new model.
        
        Parameters
        ----------
        current_model_pd : np.ndarray
            Current model PD predictions
        new_model_pd : np.ndarray
            New model PD predictions
        exposures : np.ndarray
            Exposure values (EAD)
        segment_info : np.ndarray, optional
            Segment identifiers for segmented impact analysis
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'current_rwa': RWA with current model
            - 'new_rwa': RWA with new model
            - 'rwa_difference': Absolute RWA difference
            - 'rwa_difference_pct': Percentage difference in RWA
            - 'capital_impact': Impact on capital requirements
            - 'migration_matrix': Rating migration analysis (if ratings provided)
            - 'segment_impacts': Breakdown by segments (if segments provided)
        """

    def benchmark_impact(self, current_pd: np.ndarray,
                    benchmark_pd: np.ndarray,
                    exposures: np.ndarray,
                    benchmark_name: str = 'reference',
                    **kwargs) -> Dict[str, Any]:
        """
        Compare current model RWA to benchmark model RWA.
        
        Parameters
        ----------
        current_pd : np.ndarray
            Current model PD predictions
        benchmark_pd : np.ndarray
            Benchmark model PD predictions
        exposures : np.ndarray
            Exposure values (EAD)
        benchmark_name : str
            Name of the benchmark for reporting
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'current_rwa': RWA with current model
            - 'benchmark_rwa': RWA with benchmark model
            - 'rwa_difference': Absolute RWA difference
            - 'rwa_difference_pct': Percentage difference in RWA
            - 'capital_impact': Impact on capital requirements
            - 'conservatism_index': Measure of relative conservatism vs benchmark
        """
        return self
    
    def sensitivity_analysis(self, base_pd: np.ndarray,
                            exposures: np.ndarray,
                            pd_shifts: List[float] = [-0.5, -0.25, 0, 0.25, 0.5],
                            shift_type: str = 'relative',
                            parameters_to_vary: List[str] = ['pd'],
                            **kwargs) -> Dict[str, Any]:
        """
        Perform sensitivity analysis of RWA to changes in PD and other parameters.
        
        Parameters
        ----------
        base_pd : np.ndarray
            Base PD values
        exposures : np.ndarray
            Exposure values (EAD)
        pd_shifts : List[float]
            List of shifts to apply to PD
        shift_type : str
            Type of shift ('relative', 'absolute', 'multiplier')
        parameters_to_vary : List[str]
            Parameters to include in sensitivity analysis ('pd', 'lgd', 'maturity', 'correlation')
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'baseline': Baseline RWA
            - 'scenarios': Dictionary of scenario results
            - 'pd_elasticity': Elasticity of RWA to PD changes
            - 'other_elasticities': Elasticities for other parameters
            - 'stress_results': Results of stress scenarios if defined
        """
        return self
    
    def _calculate_correlation(self, pd_value: float, asset_class: str) -> float:
        """
        Calculate asset correlation based on PD and asset class using Basel formula.
        
        Parameters
        ----------
        pd_value : float
            Probability of Default value
        asset_class : str
            Asset class ('corporate', 'retail', 'sme', 'mortgage', etc.)
        
        Returns
        -------
        float
            Asset correlation value
        """
        # Different correlation formulas based on asset class
        if asset_class in ['corporate', 'bank', 'sovereign']:
            # Formula for Corporate, Bank and Sovereign exposures
            correlation = (0.12 * (1 - math.exp(-50 * pd_value)) / (1 - math.exp(-50))) + \
                          (0.24 * (1 - (1 - math.exp(-50 * pd_value)) / (1 - math.exp(-50))))
                          
            # Apply SME adjustment if specified in config
            if asset_class == 'corporate' and self.config.get('is_sme', False):
                annual_sales = self.config.get('annual_sales', 5e6)  # Default 5M EUR
                sales_adjustment = min(max(annual_sales, 5e6), 50e6)
                sme_adjustment = 0.04 * (1 - (sales_adjustment - 5e6) / 45e6)
                correlation -= sme_adjustment
                
        elif asset_class == 'mortgage':
            # Fixed correlation for residential mortgages
            correlation = 0.15
            
        elif asset_class == 'qrr':
            # Fixed correlation for Qualifying Revolving Retail
            correlation = 0.04
            
        elif asset_class in ['retail', 'other_retail']:
            # Formula for Other Retail exposures
            correlation = (0.03 * (1 - math.exp(-35 * pd_value)) / (1 - math.exp(-35))) + \
                          (0.16 * (1 - (1 - math.exp(-35 * pd_value)) / (1 - math.exp(-35))))
                          
        else:
            # Default to corporate formula
            correlation = (0.12 * (1 - math.exp(-50 * pd_value)) / (1 - math.exp(-50))) + \
                          (0.24 * (1 - (1 - math.exp(-50 * pd_value)) / (1 - math.exp(-50))))
        
        return correlation
        
    def _calculate_capital_requirement(self, pd: float, lgd: float, correlation: float) -> float:
        """
        Calculate capital requirement for a single PD value using Basel formula.
        
        Parameters
        ----------
        pd : float
            Probability of Default value
        lgd : float
            Loss Given Default value
        correlation : float
            Asset correlation value
        
        Returns
        -------
        float
            Capital requirement as a percentage of exposure
        """
        # Basel IRB formula for capital requirement
        # K = LGD * N[(1 - R)^-0.5 * G(PD) + (R / (1 - R))^0.5 * G(0.999)] - LGD * PD
        
        # Compute inverse normal of PD
        normal_inverse_pd = np.sqrt(2) * math.erfinv(2 * pd - 1) if pd < 1 else 1000
        
        # Capital formula components
        sqrt_correlation = math.sqrt(correlation)
        sqrt_one_minus_correlation = math.sqrt(1 - correlation)
        
        # Main formula
        capital_req = lgd * (
            self._normal_cdf(
                (normal_inverse_pd * sqrt_one_minus_correlation + 
                 self.normal_inverse_conf * sqrt_correlation) / 
                sqrt_one_minus_correlation
            ) - pd
        )
        
        # Cap at LGD * PD for defaulted exposures
        if pd >= 1.0:
            capital_req = 0  # For defaulted exposures, capital requirement is handled separately
        
        return max(0, capital_req)  # Capital requirement cannot be negative
    
    def _calculate_maturity_adjustment(self, pd: float, maturity: float) -> float:
        """
        Calculate maturity adjustment factor for non-retail exposures.
        
        Parameters
        ----------
        pd : float
            Probability of Default value
        maturity : float
            Effective maturity in years
        
        Returns
        -------
        float
            Maturity adjustment factor
        """
        # Basel maturity adjustment formula
        b = (0.11852 - 0.05478 * math.log(pd))**2
        
        # Calculate adjustment
        adjustment = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)
        
        return adjustment
    
    def _normal_cdf(self, x: float) -> float:
        """
        Calculate the cumulative distribution function of the standard normal distribution.
        
        Parameters
        ----------
        x : float
            Input value
        
        Returns
        -------
        float
            CDF value
        """
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        
    def _format_detailed_report(self, impact_results: Dict[str, Any]) -> str:
        """Format detailed impact report as formatted text or HTML."""
        
    def _apply_pd_adjustment(self, pd_values: np.ndarray, 
                            adjustment_type: str,
                            adjustment_params: Dict[str, Any]) -> np.ndarray:
        """Apply structured adjustments to PD values."""
    