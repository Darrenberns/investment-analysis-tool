"""
Discounted Cash Flow (DCF) Calculation Engine

This module provides financial calculation functions for investment analysis,
including NPV, IRR, payback period, and other DCF metrics.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime
import uuid
import numpy as np
from scipy.optimize import newton


@dataclass
class CashFlow:
    """Represents a single cash flow in a specific period."""
    period: int  # 0 for initial investment, 1-10 for years
    amount: float
    description: Optional[str] = None

    def __post_init__(self):
        """Validate cash flow data."""
        if self.period < 0:
            raise ValueError("Period must be non-negative")
        if not isinstance(self.amount, (int, float, Decimal)):
            raise TypeError("Amount must be numeric")


@dataclass
class Investment:
    """Represents an investment with its cash flows."""
    name: str
    cash_flows: List[CashFlow]
    discount_rate: float  # Annual discount rate (e.g., 0.10 for 10%)
    description: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate investment data."""
        if not self.cash_flows:
            raise ValueError("Investment must have at least one cash flow")
        if not 0 <= self.discount_rate <= 1:
            raise ValueError("Discount rate must be between 0 and 1")
        
        # Sort cash flows by period
        self.cash_flows.sort(key=lambda cf: cf.period)
        
        # Ensure datetime objects (in case loaded from dict with strings)
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.modified_at, str):
            self.modified_at = datetime.fromisoformat(self.modified_at)
    
    def get_cash_flow_array(self) -> np.ndarray:
        """Convert cash flows to numpy array for calculations."""
        max_period = max(cf.period for cf in self.cash_flows)
        cash_flow_array = np.zeros(max_period + 1)
        
        for cf in self.cash_flows:
            cash_flow_array[cf.period] = cf.amount
        
        return cash_flow_array
    
    def to_dict(self) -> Dict:
        """
        Convert Investment to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of the investment
        """
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'discount_rate': self.discount_rate,
            'created_at': self.created_at.isoformat(),
            'modified_at': self.modified_at.isoformat(),
            'cash_flows': [
                {
                    'period': cf.period,
                    'amount': cf.amount,
                    'description': cf.description
                }
                for cf in self.cash_flows
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Investment':
        """
        Create Investment from dictionary (deserialization).
        
        Args:
            data: Dictionary containing investment data
            
        Returns:
            Investment object
        """
        # Reconstruct cash flows
        cash_flows = [
            CashFlow(
                period=cf_data['period'],
                amount=cf_data['amount'],
                description=cf_data.get('description')
            )
            for cf_data in data['cash_flows']
        ]
        
        # Create investment
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            name=data['name'],
            description=data.get('description'),
            discount_rate=data['discount_rate'],
            cash_flows=cash_flows,
            created_at=data.get('created_at', datetime.now().isoformat()),
            modified_at=data.get('modified_at', datetime.now().isoformat())
        )
    
    def update(self, **kwargs):
        """
        Update investment fields.
        
        Args:
            **kwargs: Fields to update (name, description, discount_rate, cash_flows)
        """
        if 'name' in kwargs:
            self.name = kwargs['name']
        if 'description' in kwargs:
            self.description = kwargs['description']
        if 'discount_rate' in kwargs:
            self.discount_rate = kwargs['discount_rate']
        if 'cash_flows' in kwargs:
            self.cash_flows = kwargs['cash_flows']
            self.cash_flows.sort(key=lambda cf: cf.period)
        
        # Update modified timestamp
        self.modified_at = datetime.now()


class DCFCalculator:
    """
    Main calculation engine for Discounted Cash Flow analysis.
    """
    
    @staticmethod
    def calculate_present_value(future_value: float, discount_rate: float, period: int) -> float:
        """
        Calculate present value of a future cash flow.
        
        PV = FV / (1 + r)^n
        
        Args:
            future_value: The future cash flow amount
            discount_rate: Annual discount rate (decimal)
            period: Number of periods in the future
            
        Returns:
            Present value of the future cash flow
        """
        if period == 0:
            return future_value
        return future_value / ((1 + discount_rate) ** period)
    
    @staticmethod
    def calculate_npv(investment: Investment) -> float:
        """
        Calculate Net Present Value (NPV) of an investment.
        
        NPV = ÃŽÂ£(CFt / (1 + r)^t) where t = 0 to n
        
        Args:
            investment: Investment object with cash flows and discount rate
            
        Returns:
            Net Present Value
        """
        npv = 0.0
        for cf in investment.cash_flows:
            pv = DCFCalculator.calculate_present_value(
                cf.amount, 
                investment.discount_rate, 
                cf.period
            )
            npv += pv
        return npv
    
    @staticmethod
    def calculate_npv_breakdown(investment: Investment) -> List[Dict]:
        """
        Calculate NPV with period-by-period breakdown.
        
        Args:
            investment: Investment object
            
        Returns:
            List of dictionaries containing period, cash flow, discount factor, and PV
        """
        breakdown = []
        cumulative_pv = 0.0
        
        for cf in investment.cash_flows:
            discount_factor = 1 / ((1 + investment.discount_rate) ** cf.period)
            pv = cf.amount * discount_factor
            cumulative_pv += pv
            
            breakdown.append({
                'period': cf.period,
                'cash_flow': cf.amount,
                'discount_rate': investment.discount_rate,
                'discount_factor': discount_factor,
                'present_value': pv,
                'cumulative_pv': cumulative_pv,
                'description': cf.description
            })
        
        return breakdown
    
    @staticmethod
    def calculate_irr(investment: Investment, initial_guess: float = 0.1) -> Optional[float]:
        """
        Calculate Internal Rate of Return (IRR).
        
        IRR is the discount rate that makes NPV = 0
        
        Args:
            investment: Investment object
            initial_guess: Starting point for IRR calculation
            
        Returns:
            IRR as a decimal, or None if calculation fails
        """
        cash_flows = investment.get_cash_flow_array()
        
        # Define NPV function for optimization
        def npv_function(rate):
            return np.sum(cash_flows / ((1 + rate) ** np.arange(len(cash_flows))))
        
        try:
            # Use Newton's method to find the root (where NPV = 0)
            irr = newton(npv_function, initial_guess, maxiter=1000)
            
            # Verify the result makes sense
            if -1 < irr < 10:  # Reasonable range for IRR
                return irr
            return None
        except (RuntimeError, OverflowError):
            # Fallback to numpy's IRR if Newton's method fails
            try:
                return float(np.irr(cash_flows))
            except:
                return None
    
    @staticmethod
    def calculate_payback_period(investment: Investment) -> Optional[float]:
        """
        Calculate payback period (when cumulative cash flows turn positive).
        
        Args:
            investment: Investment object
            
        Returns:
            Payback period in years, or None if investment never pays back
        """
        cumulative = 0.0
        previous_cumulative = 0.0
        
        for cf in investment.cash_flows:
            previous_cumulative = cumulative
            cumulative += cf.amount
            
            if cumulative >= 0 and previous_cumulative < 0:
                # Interpolate to find exact payback period
                fraction = abs(previous_cumulative) / cf.amount
                return cf.period - 1 + fraction
        
        # If still negative, no payback
        if cumulative < 0:
            return None
        
        # If positive from start, payback is immediate
        return 0.0
    
    @staticmethod
    def calculate_discounted_payback(investment: Investment) -> Optional[float]:
        """
        Calculate discounted payback period using present values.
        
        Args:
            investment: Investment object
            
        Returns:
            Discounted payback period in years, or None if never pays back
        """
        cumulative_pv = 0.0
        previous_cumulative_pv = 0.0
        
        for cf in investment.cash_flows:
            previous_cumulative_pv = cumulative_pv
            pv = DCFCalculator.calculate_present_value(
                cf.amount,
                investment.discount_rate,
                cf.period
            )
            cumulative_pv += pv
            
            if cumulative_pv >= 0 and previous_cumulative_pv < 0:
                fraction = abs(previous_cumulative_pv) / pv
                return cf.period - 1 + fraction
        
        if cumulative_pv < 0:
            return None
        
        return 0.0
    
    @staticmethod
    def calculate_profitability_index(investment: Investment) -> Optional[float]:
        """
        Calculate Profitability Index (PI).
        
        PI = PV of future cash flows / Initial Investment
        
        Args:
            investment: Investment object
            
        Returns:
            Profitability Index, or None if no initial investment
        """
        # Get initial investment (period 0, typically negative)
        initial_investment = next(
            (cf.amount for cf in investment.cash_flows if cf.period == 0),
            None
        )
        
        if initial_investment is None or initial_investment >= 0:
            return None
        
        # Calculate PV of future cash flows (excluding period 0)
        future_pv = sum(
            DCFCalculator.calculate_present_value(cf.amount, investment.discount_rate, cf.period)
            for cf in investment.cash_flows if cf.period > 0
        )
        
        # PI = PV of future cash flows / absolute value of initial investment
        return future_pv / abs(initial_investment)
    
    @staticmethod
    def calculate_all_metrics(investment: Investment) -> Dict:
        """
        Calculate all key DCF metrics for an investment.
        
        Args:
            investment: Investment object
            
        Returns:
            Dictionary containing all calculated metrics
        """
        npv = DCFCalculator.calculate_npv(investment)
        irr = DCFCalculator.calculate_irr(investment)
        payback = DCFCalculator.calculate_payback_period(investment)
        discounted_payback = DCFCalculator.calculate_discounted_payback(investment)
        pi = DCFCalculator.calculate_profitability_index(investment)
        breakdown = DCFCalculator.calculate_npv_breakdown(investment)
        
        return {
            'investment_name': investment.name,
            'discount_rate': investment.discount_rate,
            'npv': npv,
            'irr': irr,
            'payback_period': payback,
            'discounted_payback_period': discounted_payback,
            'profitability_index': pi,
            'breakdown': breakdown,
            'recommendation': 'Accept' if npv > 0 else 'Reject'
        }


class InvestmentComparator:
    """
    Compare multiple investments and rank them.
    """
    
    @staticmethod
    def compare_investments(investments: List[Investment], 
                          ranking_metric: str = 'npv') -> List[Dict]:
        """
        Compare multiple investments and rank them.
        
        Args:
            investments: List of Investment objects
            ranking_metric: Metric to rank by ('npv', 'irr', 'pi')
            
        Returns:
            List of investment metrics sorted by ranking metric
        """
        results = []
        
        for investment in investments:
            metrics = DCFCalculator.calculate_all_metrics(investment)
            results.append(metrics)
        
        # Sort by ranking metric (descending)
        metric_key = {
            'npv': 'npv',
            'irr': 'irr',
            'pi': 'profitability_index'
        }.get(ranking_metric, 'npv')
        
        results.sort(
            key=lambda x: x[metric_key] if x[metric_key] is not None else float('-inf'),
            reverse=True
        )
        
        # Add rank
        for idx, result in enumerate(results, 1):
            result['rank'] = idx
        
        return results
    
    @staticmethod
    def create_comparison_matrix(investments: List[Investment]) -> Dict:
        """
        Create a comparison matrix showing all investments side-by-side.
        
        Args:
            investments: List of Investment objects
            
        Returns:
            Dictionary with comparison data
        """
        matrix = {
            'investments': [],
            'metrics': ['NPV', 'IRR', 'Payback Period', 'Discounted Payback', 'Profitability Index'],
            'data': []
        }
        
        for investment in investments:
            metrics = DCFCalculator.calculate_all_metrics(investment)
            matrix['investments'].append(investment.name)
            matrix['data'].append({
                'name': investment.name,
                'NPV': metrics['npv'],
                'IRR': metrics['irr'],
                'Payback Period': metrics['payback_period'],
                'Discounted Payback': metrics['discounted_payback_period'],
                'Profitability Index': metrics['profitability_index'],
                'Recommendation': metrics['recommendation']
            })
        
        return matrix


# Utility functions for validation and formatting

def validate_cash_flows(cash_flows: List[Dict]) -> Tuple[bool, Optional[str]]:
    """
    Validate cash flow data structure.
    
    Args:
        cash_flows: List of dictionaries with 'period' and 'amount' keys
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not cash_flows:
        return False, "Cash flows cannot be empty"
    
    periods = set()
    for cf in cash_flows:
        if 'period' not in cf or 'amount' not in cf:
            return False, "Each cash flow must have 'period' and 'amount'"
        
        if cf['period'] in periods:
            return False, f"Duplicate period: {cf['period']}"
        
        periods.add(cf['period'])
        
        if cf['period'] < 0:
            return False, "Periods must be non-negative"
    
    return True, None


def format_currency(amount: float, currency_symbol: str = '$') -> str:
    """Format number as currency."""
    return f"{currency_symbol}{amount:,.2f}"


def format_percentage(rate: float) -> str:
    """Format decimal as percentage."""
    return f"{rate * 100:.2f}%"
