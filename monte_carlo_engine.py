"""
Monte Carlo Simulation Engine for Probabilistic DCF Analysis (Phase 4)

This module extends the basic DCF calculator with probabilistic analysis
using Monte Carlo simulation to account for uncertainty in cash flows.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import scipy.stats as stats


class DistributionType(Enum):
    """Supported probability distributions for cash flows."""
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    TRIANGULAR = "triangular"
    UNIFORM = "uniform"
    BETA = "beta"


@dataclass
class ProbabilisticCashFlow:
    """
    Cash flow with probability distribution instead of fixed value.
    """
    period: int
    distribution_type: DistributionType
    parameters: Dict  # Distribution-specific parameters
    description: Optional[str] = None
    
    def sample(self, size: int = 1) -> np.ndarray:
        """
        Generate random samples from the distribution.
        
        Args:
            size: Number of samples to generate
            
        Returns:
            Array of sampled values
        """
        if self.distribution_type == DistributionType.NORMAL:
            # Parameters: mean, std_dev
            return np.random.normal(
                self.parameters['mean'],
                self.parameters['std_dev'],
                size
            )
        
        elif self.distribution_type == DistributionType.LOGNORMAL:
            # Parameters: mean, std_dev (of underlying normal)
            return np.random.lognormal(
                self.parameters['mean'],
                self.parameters['std_dev'],
                size
            )
        
        elif self.distribution_type == DistributionType.TRIANGULAR:
            # Parameters: min, mode (most likely), max
            return np.random.triangular(
                self.parameters['min'],
                self.parameters['mode'],
                self.parameters['max'],
                size
            )
        
        elif self.distribution_type == DistributionType.UNIFORM:
            # Parameters: min, max
            return np.random.uniform(
                self.parameters['min'],
                self.parameters['max'],
                size
            )
        
        elif self.distribution_type == DistributionType.BETA:
            # Parameters: alpha, beta, min, max (for scaling)
            beta_samples = np.random.beta(
                self.parameters['alpha'],
                self.parameters['beta'],
                size
            )
            # Scale from [0,1] to [min, max]
            return (self.parameters['min'] + 
                   beta_samples * (self.parameters['max'] - self.parameters['min']))


@dataclass
class ProbabilisticInvestment:
    """Investment with probabilistic cash flows."""
    name: str
    initial_investment: float  # Usually deterministic
    probabilistic_cash_flows: List[ProbabilisticCashFlow]
    discount_rate: float
    discount_rate_uncertainty: Optional[Dict] = None  # Optional uncertainty in discount rate


@dataclass
class MonteCarloResults:
    """Results from Monte Carlo simulation."""
    investment_name: str
    num_simulations: int
    
    # NPV statistics
    mean_npv: float
    median_npv: float
    std_npv: float
    min_npv: float
    max_npv: float
    
    # Percentiles
    npv_5th_percentile: float
    npv_25th_percentile: float
    npv_75th_percentile: float
    npv_95th_percentile: float
    
    # Risk metrics
    probability_positive_npv: float
    value_at_risk_95: float  # 95% VaR
    conditional_var_95: float  # Expected loss in worst 5% of cases
    
    # Full distribution
    npv_distribution: np.ndarray
    
    # IRR statistics (if calculable)
    mean_irr: Optional[float] = None
    median_irr: Optional[float] = None
    std_irr: Optional[float] = None


class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for probabilistic DCF analysis.
    """
    
    @staticmethod
    def run_simulation(
        investment: ProbabilisticInvestment,
        num_simulations: int = 10000,
        random_seed: Optional[int] = None
    ) -> MonteCarloResults:
        """
        Run Monte Carlo simulation for an investment.
        
        Args:
            investment: Probabilistic investment to analyze
            num_simulations: Number of simulation runs
            random_seed: Seed for reproducibility
            
        Returns:
            MonteCarloResults object with statistics
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize arrays to store results
        npv_results = np.zeros(num_simulations)
        irr_results = []
        
        # Run simulations
        for i in range(num_simulations):
            # Sample cash flows for this simulation
            sampled_cash_flows = np.zeros(
                max(cf.period for cf in investment.probabilistic_cash_flows) + 1
            )
            
            # Initial investment (usually deterministic)
            sampled_cash_flows[0] = investment.initial_investment
            
            # Sample each period's cash flow
            for cf in investment.probabilistic_cash_flows:
                sampled_cash_flows[cf.period] = cf.sample(size=1)[0]
            
            # Sample discount rate if uncertain
            if investment.discount_rate_uncertainty:
                discount_rate = np.random.normal(
                    investment.discount_rate,
                    investment.discount_rate_uncertainty['std_dev']
                )
                discount_rate = max(0.01, min(discount_rate, 0.99))  # Bound it
            else:
                discount_rate = investment.discount_rate
            
            # Calculate NPV for this simulation
            npv = MonteCarloSimulator._calculate_npv(
                sampled_cash_flows,
                discount_rate
            )
            npv_results[i] = npv
            
            # Try to calculate IRR (may not always be possible)
            try:
                irr = np.irr(sampled_cash_flows)
                if -1 < irr < 10:  # Reasonable range
                    irr_results.append(irr)
            except:
                pass
        
        # Calculate statistics
        return MonteCarloResults(
            investment_name=investment.name,
            num_simulations=num_simulations,
            mean_npv=float(np.mean(npv_results)),
            median_npv=float(np.median(npv_results)),
            std_npv=float(np.std(npv_results)),
            min_npv=float(np.min(npv_results)),
            max_npv=float(np.max(npv_results)),
            npv_5th_percentile=float(np.percentile(npv_results, 5)),
            npv_25th_percentile=float(np.percentile(npv_results, 25)),
            npv_75th_percentile=float(np.percentile(npv_results, 75)),
            npv_95th_percentile=float(np.percentile(npv_results, 95)),
            probability_positive_npv=float(np.mean(npv_results > 0)),
            value_at_risk_95=float(np.percentile(npv_results, 5)),
            conditional_var_95=float(np.mean(npv_results[npv_results <= np.percentile(npv_results, 5)])),
            npv_distribution=npv_results,
            mean_irr=float(np.mean(irr_results)) if irr_results else None,
            median_irr=float(np.median(irr_results)) if irr_results else None,
            std_irr=float(np.std(irr_results)) if irr_results else None
        )
    
    @staticmethod
    def _calculate_npv(cash_flows: np.ndarray, discount_rate: float) -> float:
        """Calculate NPV for a single simulation run."""
        periods = np.arange(len(cash_flows))
        discount_factors = 1 / ((1 + discount_rate) ** periods)
        return float(np.sum(cash_flows * discount_factors))
    
    @staticmethod
    def compare_investments(
        investments: List[ProbabilisticInvestment],
        num_simulations: int = 10000
    ) -> List[MonteCarloResults]:
        """
        Compare multiple probabilistic investments.
        
        Args:
            investments: List of probabilistic investments
            num_simulations: Number of simulations per investment
            
        Returns:
            List of results, sorted by mean NPV
        """
        results = []
        
        for investment in investments:
            result = MonteCarloSimulator.run_simulation(
                investment,
                num_simulations
            )
            results.append(result)
        
        # Sort by mean NPV
        results.sort(key=lambda r: r.mean_npv, reverse=True)
        
        return results
    
    @staticmethod
    def calculate_stochastic_dominance(
        results_a: MonteCarloResults,
        results_b: MonteCarloResults
    ) -> Dict:
        """
        Check if one investment stochastically dominates another.
        
        First-order stochastic dominance: A dominates B if for all outcomes,
        the probability of A being at least x is >= the probability of B being at least x.
        
        Args:
            results_a: Results for investment A
            results_b: Results for investment B
            
        Returns:
            Dictionary with dominance information
        """
        # Create empirical CDFs
        sorted_a = np.sort(results_a.npv_distribution)
        sorted_b = np.sort(results_b.npv_distribution)
        
        # Check first-order stochastic dominance
        # A dominates B if CDF_A(x) <= CDF_B(x) for all x
        test_points = np.linspace(
            min(sorted_a[0], sorted_b[0]),
            max(sorted_a[-1], sorted_b[-1]),
            1000
        )
        
        cdf_a = np.searchsorted(sorted_a, test_points) / len(sorted_a)
        cdf_b = np.searchsorted(sorted_b, test_points) / len(sorted_b)
        
        a_dominates_b = np.all(cdf_a <= cdf_b)
        b_dominates_a = np.all(cdf_b <= cdf_a)
        
        return {
            'a_dominates_b': a_dominates_b,
            'b_dominates_a': b_dominates_a,
            'a_mean_higher': results_a.mean_npv > results_b.mean_npv,
            'a_less_risky': results_a.std_npv < results_b.std_npv,
            'sharpe_ratio_a': (results_a.mean_npv / results_a.std_npv 
                              if results_a.std_npv > 0 else 0),
            'sharpe_ratio_b': (results_b.mean_npv / results_b.std_npv 
                              if results_b.std_npv > 0 else 0)
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_monte_carlo_simulation():
    """Example: Monte Carlo simulation with uncertain cash flows."""
    
    print("="*80)
    print("MONTE CARLO SIMULATION EXAMPLE")
    print("="*80)
    
    # Define an investment with uncertain cash flows
    # Year 1-3: Normal distribution (mean Â± std dev)
    # Year 4-5: Triangular distribution (pessimistic, most likely, optimistic)
    
    investment = ProbabilisticInvestment(
        name="Uncertain Software Project",
        initial_investment=-500000,
        probabilistic_cash_flows=[
            ProbabilisticCashFlow(
                period=1,
                distribution_type=DistributionType.NORMAL,
                parameters={'mean': 120000, 'std_dev': 30000},
                description="Year 1: High uncertainty"
            ),
            ProbabilisticCashFlow(
                period=2,
                distribution_type=DistributionType.NORMAL,
                parameters={'mean': 180000, 'std_dev': 40000},
                description="Year 2: Growing market"
            ),
            ProbabilisticCashFlow(
                period=3,
                distribution_type=DistributionType.TRIANGULAR,
                parameters={'min': 150000, 'mode': 200000, 'max': 280000},
                description="Year 3: Market established"
            ),
            ProbabilisticCashFlow(
                period=4,
                distribution_type=DistributionType.TRIANGULAR,
                parameters={'min': 180000, 'mode': 250000, 'max': 350000},
                description="Year 4: Mature phase"
            ),
            ProbabilisticCashFlow(
                period=5,
                distribution_type=DistributionType.TRIANGULAR,
                parameters={'min': 200000, 'mode': 280000, 'max': 400000},
                description="Year 5: Peak revenue"
            ),
        ],
        discount_rate=0.10,
        discount_rate_uncertainty={'std_dev': 0.02}  # 2% uncertainty in discount rate
    )
    
    # Run simulation
    print(f"\nRunning 10,000 Monte Carlo simulations for: {investment.name}\n")
    results = MonteCarloSimulator.run_simulation(investment, num_simulations=10000)
    
    # Display results
    print(f"Investment: {results.investment_name}")
    print(f"Number of Simulations: {results.num_simulations:,}")
    print(f"\nNPV Statistics:")
    print(f"  Mean NPV:               ${results.mean_npv:,.2f}")
    print(f"  Median NPV:             ${results.median_npv:,.2f}")
    print(f"  Standard Deviation:     ${results.std_npv:,.2f}")
    print(f"  Minimum NPV:            ${results.min_npv:,.2f}")
    print(f"  Maximum NPV:            ${results.max_npv:,.2f}")
    
    print(f"\nPercentiles:")
    print(f"  5th Percentile:         ${results.npv_5th_percentile:,.2f}")
    print(f"  25th Percentile (Q1):   ${results.npv_25th_percentile:,.2f}")
    print(f"  75th Percentile (Q3):   ${results.npv_75th_percentile:,.2f}")
    print(f"  95th Percentile:        ${results.npv_95th_percentile:,.2f}")
    
    print(f"\nRisk Metrics:")
    print(f"  Probability of Positive NPV: {results.probability_positive_npv*100:.1f}%")
    print(f"  Value at Risk (95%):         ${results.value_at_risk_95:,.2f}")
    print(f"  Conditional VaR (95%):       ${results.conditional_var_95:,.2f}")
    
    if results.mean_irr:
        print(f"\nIRR Statistics:")
        print(f"  Mean IRR:               {results.mean_irr*100:.2f}%")
        print(f"  Median IRR:             {results.median_irr*100:.2f}%")
    
    print(f"\nInterpretation:")
    if results.probability_positive_npv > 0.8:
        print(f"  âœ“ Strong investment: {results.probability_positive_npv*100:.0f}% chance of positive NPV")
    elif results.probability_positive_npv > 0.6:
        print(f"  ~ Moderate investment: {results.probability_positive_npv*100:.0f}% chance of positive NPV")
    else:
        print(f"  âœ— Risky investment: Only {results.probability_positive_npv*100:.0f}% chance of positive NPV")
    
    # Calculate coefficient of variation (risk per unit of return)
    cv = results.std_npv / results.mean_npv if results.mean_npv != 0 else float('inf')
    print(f"  Coefficient of Variation: {cv:.2f} (lower is better)")


def example_compare_risky_investments():
    """Compare a safe vs risky investment using Monte Carlo."""
    
    print("\n" + "="*80)
    print("COMPARING SAFE VS RISKY INVESTMENTS")
    print("="*80)
    
    # Safe investment: Low returns, low variance
    safe_investment = ProbabilisticInvestment(
        name="Safe Bond Investment",
        initial_investment=-400000,
        probabilistic_cash_flows=[
            ProbabilisticCashFlow(
                period=i,
                distribution_type=DistributionType.NORMAL,
                parameters={'mean': 95000, 'std_dev': 5000},
                description=f"Year {i}: Predictable bonds"
            )
            for i in range(1, 6)
        ],
        discount_rate=0.08
    )
    
    # Risky investment: High potential returns, high variance
    risky_investment = ProbabilisticInvestment(
        name="Risky Startup Investment",
        initial_investment=-400000,
        probabilistic_cash_flows=[
            ProbabilisticCashFlow(
                period=i,
                distribution_type=DistributionType.LOGNORMAL,
                parameters={'mean': 11.5, 'std_dev': 0.6},  # log-space parameters
                description=f"Year {i}: Volatile startup"
            )
            for i in range(1, 6)
        ],
        discount_rate=0.12  # Higher discount rate for risky investment
    )
    
    # Run simulations
    results = MonteCarloSimulator.compare_investments(
        [safe_investment, risky_investment],
        num_simulations=10000
    )
    
    print(f"\nComparison Results:\n")
    print(f"{'Metric':<35} {'Safe Investment':<25} {'Risky Investment':<25}")
    print("-" * 85)
    print(f"{'Mean NPV':<35} ${results[0].mean_npv:>20,.0f}  ${results[1].mean_npv:>20,.0f}")
    print(f"{'Standard Deviation':<35} ${results[0].std_npv:>20,.0f}  ${results[1].std_npv:>20,.0f}")
    print(f"{'Probability of Positive NPV':<35} {results[0].probability_positive_npv:>20.1%}  {results[1].probability_positive_npv:>20.1%}")
    print(f"{'Value at Risk (5%)':<35} ${results[0].value_at_risk_95:>20,.0f}  ${results[1].value_at_risk_95:>20,.0f}")
    
    # Calculate Sharpe-like ratio
    sharpe_safe = results[0].mean_npv / results[0].std_npv if results[0].std_npv > 0 else 0
    sharpe_risky = results[1].mean_npv / results[1].std_npv if results[1].std_npv > 0 else 0
    
    print(f"{'Risk-Adjusted Return Ratio':<35} {sharpe_safe:>24.2f}  {sharpe_risky:>24.2f}")
    
    # Stochastic dominance
    dominance = MonteCarloSimulator.calculate_stochastic_dominance(
        results[0], results[1]
    )
    
    print(f"\nStochastic Dominance Analysis:")
    if dominance['a_dominates_b']:
        print(f"  Safe investment stochastically dominates risky investment")
    elif dominance['b_dominates_a']:
        print(f"  Risky investment stochastically dominates safe investment")
    else:
        print(f"  No clear dominance - choice depends on risk preferences")
        print(f"  Safe: Higher Sharpe ratio = {dominance['sharpe_ratio_a']:.2f}")
        print(f"  Risky: Sharpe ratio = {dominance['sharpe_ratio_b']:.2f}")


if __name__ == "__main__":
    example_monte_carlo_simulation()
    example_compare_risky_investments()
