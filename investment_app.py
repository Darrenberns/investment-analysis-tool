"""
Investment Comparison Tool - Streamlit App
A professional tool for DCF analysis, investment comparison, and Monte Carlo simulation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add project directory to path to import our engines
project_path = Path("/mnt/project")
sys.path.insert(0, str(project_path))

from dcf_engine import (
    CashFlow, Investment, DCFCalculator, InvestmentComparator,
    format_currency, format_percentage
)
from monte_carlo_engine import (
    ProbabilisticCashFlow, ProbabilisticInvestment, 
    MonteCarloSimulator, DistributionType
)

# Page configuration
st.set_page_config(
    page_title="Investment Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .positive {
        color: #28a745;
        font-weight: bold;
    }
    .negative {
        color: #dc3545;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'investments' not in st.session_state:
    st.session_state.investments = []
if 'investment_counter' not in st.session_state:
    st.session_state.investment_counter = 0

def create_investment_from_form(name, description, initial_investment, discount_rate, cash_flows_df):
    """Create an Investment object from form inputs."""
    cash_flows = [
        CashFlow(period=0, amount=initial_investment, description="Initial Investment")
    ]
    
    for idx, row in cash_flows_df.iterrows():
        if pd.notna(row['Amount']) and row['Amount'] != 0:
            cash_flows.append(
                CashFlow(
                    period=int(row['Year']),
                    amount=float(row['Amount']),
                    description=row['Description'] if pd.notna(row['Description']) else f"Year {row['Year']}"
                )
            )
    
    return Investment(
        name=name,
        cash_flows=cash_flows,
        discount_rate=discount_rate
    )

def display_metrics_cards(metrics):
    """Display key metrics in card format."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        npv_class = "positive" if metrics['npv'] > 0 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin:0; color: #666;">Net Present Value</h4>
            <h2 class="{npv_class}" style="margin:0.5rem 0 0 0;">{format_currency(metrics['npv'])}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        irr_display = format_percentage(metrics['irr']) if metrics['irr'] else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin:0; color: #666;">Internal Rate of Return</h4>
            <h2 style="margin:0.5rem 0 0 0; color: #1f77b4;">{irr_display}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        pi_display = f"{metrics['profitability_index']:.2f}" if metrics['profitability_index'] else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin:0; color: #666;">Profitability Index</h4>
            <h2 style="margin:0.5rem 0 0 0; color: #9467bd;">{pi_display}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        rec_class = "positive" if metrics['recommendation'] == 'Accept' else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin:0; color: #666;">Recommendation</h4>
            <h2 class="{rec_class}" style="margin:0.5rem 0 0 0;">{metrics['recommendation']}</h2>
        </div>
        """, unsafe_allow_html=True)

def create_cash_flow_chart(breakdown):
    """Create interactive cash flow visualization."""
    df = pd.DataFrame(breakdown)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Cash Flows Over Time", "Cumulative Present Value"),
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5]
    )
    
    # Cash flows bar chart
    colors = ['red' if x < 0 else 'green' for x in df['cash_flow']]
    fig.add_trace(
        go.Bar(
            x=df['period'],
            y=df['cash_flow'],
            name='Cash Flow',
            marker_color=colors,
            text=df['cash_flow'].apply(lambda x: format_currency(x)),
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # Cumulative PV line chart
    fig.add_trace(
        go.Scatter(
            x=df['period'],
            y=df['cumulative_pv'],
            name='Cumulative PV',
            mode='lines+markers',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ),
        row=2, col=1
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_yaxes(title_text="Amount ($)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative PV ($)", row=2, col=1)
    
    fig.update_layout(
        height=700,
        showlegend=False,
        hovermode='x unified'
    )
    
    return fig

def create_sensitivity_chart(sensitivity_data):
    """Create sensitivity analysis chart."""
    df = pd.DataFrame(sensitivity_data)
    
    fig = go.Figure()
    
    # NPV line
    fig.add_trace(go.Scatter(
        x=df['discount_rate'] * 100,
        y=df['npv'],
        mode='lines+markers',
        name='NPV',
        line=dict(color='blue', width=3),
        marker=dict(size=10)
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red", 
                  annotation_text="Break-even", annotation_position="right")
    
    fig.update_layout(
        title="NPV Sensitivity to Discount Rate",
        xaxis_title="Discount Rate (%)",
        yaxis_title="Net Present Value ($)",
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_monte_carlo_histogram(results):
    """Create histogram of Monte Carlo simulation results."""
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=results.npv_distribution,
        nbinsx=50,
        name='NPV Distribution',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    # Add percentile lines
    fig.add_vline(x=results.npv_5th_percentile, line_dash="dash", line_color="red",
                  annotation_text="5th %ile", annotation_position="top")
    fig.add_vline(x=results.median_npv, line_dash="solid", line_color="green",
                  annotation_text="Median", annotation_position="top")
    fig.add_vline(x=results.npv_95th_percentile, line_dash="dash", line_color="red",
                  annotation_text="95th %ile", annotation_position="top")
    fig.add_vline(x=0, line_dash="dot", line_color="black",
                  annotation_text="Zero NPV", annotation_position="bottom")
    
    fig.update_layout(
        title=f"Monte Carlo Simulation Results ({results.num_simulations:,} simulations)",
        xaxis_title="Net Present Value ($)",
        yaxis_title="Frequency",
        height=500,
        showlegend=False
    )
    
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown('<h1 class="main-header">📊 Investment Analysis Tool</h1>', unsafe_allow_html=True)
    st.markdown("**Professional DCF Analysis | Investment Comparison | Monte Carlo Simulation**")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a section:",
        ["🏠 Home", "➕ New Investment", "📊 Analyze Investment", 
         "⚖️ Compare Investments", "🎲 Monte Carlo Simulation", 
         "📈 Sensitivity Analysis", "💾 Manage Investments"]
    )
    
    # ========================================================================
    # HOME PAGE
    # ========================================================================
    if page == "🏠 Home":
        st.header("Welcome to the Investment Analysis Tool")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 What This Tool Does")
            st.markdown("""
            This professional investment analysis tool helps you:
            
            - **Evaluate Investments**: Calculate NPV, IRR, Payback Period, and more
            - **Compare Opportunities**: Rank multiple investments side-by-side
            - **Assess Risk**: Run Monte Carlo simulations with uncertainty
            - **Test Scenarios**: Perform sensitivity analysis on key assumptions
            
            Built on proven Discounted Cash Flow (DCF) methodology used by Fortune 500 companies.
            """)
            
            st.subheader("🚀 Quick Start")
            st.markdown("""
            1. **Create an Investment** - Add your project details and cash flows
            2. **Analyze** - View comprehensive metrics and visualizations
            3. **Compare** - Rank multiple opportunities
            4. **Test Uncertainty** - Run Monte Carlo simulations
            """)
        
        with col2:
            st.subheader("📊 Current Portfolio")
            if st.session_state.investments:
                st.metric("Total Investments", len(st.session_state.investments))
                
                # Quick summary table
                summary_data = []
                for inv_data in st.session_state.investments:
                    inv = inv_data['investment']
                    metrics = DCFCalculator.calculate_all_metrics(inv)
                    summary_data.append({
                        'Name': inv.name,
                        'NPV': metrics['npv'],
                        'IRR': metrics['irr'] if metrics['irr'] else 0,
                        'Recommendation': metrics['recommendation']
                    })
                
                df = pd.DataFrame(summary_data)
                st.dataframe(
                    df.style.format({
                        'NPV': '${:,.0f}',
                        'IRR': '{:.2%}'
                    }),
                    use_container_width=True
                )
            else:
                st.info("👋 No investments yet. Click '➕ New Investment' to get started!")
        
        # Key Metrics Explanation
        with st.expander("📚 Understanding the Metrics"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **Net Present Value (NPV)**  
                The value created by an investment in today's dollars. Accept if NPV > 0.
                
                **Internal Rate of Return (IRR)**  
                The annualized rate of return. Compare to your hurdle rate.
                
                **Profitability Index (PI)**  
                Return per dollar invested. PI > 1 indicates value creation.
                """)
            with col2:
                st.markdown("""
                **Payback Period**  
                Time to recover initial investment (undiscounted).
                
                **Discounted Payback**  
                Time to recover initial investment (time-value adjusted).
                
                **Value at Risk (VaR)**  
                Worst-case scenario in 95% of simulations.
                """)
    
    # ========================================================================
    # NEW INVESTMENT PAGE
    # ========================================================================
    elif page == "➕ New Investment":
        st.header("Create New Investment")
        
        with st.form("investment_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Investment Name*", placeholder="e.g., New Product Launch")
                description = st.text_area("Description", placeholder="Brief description of the investment opportunity")
            
            with col2:
                initial_investment = st.number_input(
                    "Initial Investment ($)*",
                    value=-100000.0,
                    step=10000.0,
                    help="Enter as negative value (e.g., -500000)"
                )
                discount_rate = st.slider(
                    "Discount Rate (%)*",
                    min_value=1.0,
                    max_value=30.0,
                    value=10.0,
                    step=0.5,
                    help="Annual discount rate (cost of capital)"
                ) / 100
            
            st.subheader("Cash Flows")
            st.markdown("Enter the expected cash flows for each year:")
            
            # Number of years selector
            num_years = st.slider("Number of Years", min_value=1, max_value=10, value=5)
            
            # Create cash flow input table
            cash_flow_data = {
                'Year': list(range(1, num_years + 1)),
                'Amount': [0.0] * num_years,
                'Description': [''] * num_years
            }
            
            df_editor = st.data_editor(
                pd.DataFrame(cash_flow_data),
                use_container_width=True,
                num_rows="fixed",
                column_config={
                    "Year": st.column_config.NumberColumn(
                        "Year",
                        disabled=True,
                        width="small"
                    ),
                    "Amount": st.column_config.NumberColumn(
                        "Amount ($)",
                        format="$%.2f",
                        width="medium"
                    ),
                    "Description": st.column_config.TextColumn(
                        "Description",
                        width="large"
                    )
                }
            )
            
            submitted = st.form_submit_button("Create Investment", use_container_width=True)
            
            if submitted:
                if not name:
                    st.error("Please enter an investment name")
                elif initial_investment >= 0:
                    st.error("Initial investment must be negative (it's a cost)")
                elif df_editor['Amount'].sum() == 0:
                    st.error("Please enter at least one cash flow")
                else:
                    try:
                        # Create investment
                        investment = create_investment_from_form(
                            name, description, initial_investment, 
                            discount_rate, df_editor
                        )
                        
                        # Calculate metrics
                        metrics = DCFCalculator.calculate_all_metrics(investment)
                        
                        # Store in session state
                        st.session_state.investments.append({
                            'id': st.session_state.investment_counter,
                            'investment': investment,
                            'metrics': metrics,
                            'description': description
                        })
                        st.session_state.investment_counter += 1
                        
                        st.success(f"✅ Investment '{name}' created successfully!")
                        st.balloons()
                        
                        # Show quick preview
                        st.subheader("Quick Preview")
                        display_metrics_cards(metrics)
                        
                    except Exception as e:
                        st.error(f"Error creating investment: {str(e)}")
    
    # ========================================================================
    # ANALYZE INVESTMENT PAGE
    # ========================================================================
    elif page == "📊 Analyze Investment":
        st.header("Analyze Investment")
        
        if not st.session_state.investments:
            st.warning("No investments available. Create one first!")
            return
        
        # Select investment
        investment_names = [inv['investment'].name for inv in st.session_state.investments]
        selected_name = st.selectbox("Select Investment to Analyze", investment_names)
        
        # Find selected investment
        selected_inv_data = next(
            (inv for inv in st.session_state.investments 
             if inv['investment'].name == selected_name),
            None
        )
        
        if selected_inv_data:
            investment = selected_inv_data['investment']
            metrics = selected_inv_data['metrics']
            
            # Display description if available
            if selected_inv_data['description']:
                st.info(f"**Description:** {selected_inv_data['description']}")
            
            # Key Metrics
            st.subheader("Key Metrics")
            display_metrics_cards(metrics)
            
            # Additional metrics in columns
            st.subheader("Additional Details")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Payback Period",
                    f"{metrics['payback_period']:.2f} years" if metrics['payback_period'] else "Never"
                )
            
            with col2:
                st.metric(
                    "Discounted Payback",
                    f"{metrics['discounted_payback_period']:.2f} years" if metrics['discounted_payback_period'] else "Never"
                )
            
            with col3:
                st.metric(
                    "Discount Rate",
                    format_percentage(metrics['discount_rate'])
                )
            
            # Visualizations
            st.subheader("Cash Flow Analysis")
            fig = create_cash_flow_chart(metrics['breakdown'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed breakdown table
            with st.expander("📋 View Detailed Cash Flow Breakdown"):
                df = pd.DataFrame(metrics['breakdown'])
                st.dataframe(
                    df.style.format({
                        'cash_flow': '${:,.2f}',
                        'discount_factor': '{:.6f}',
                        'present_value': '${:,.2f}',
                        'cumulative_pv': '${:,.2f}',
                        'discount_rate': '{:.2%}'
                    }),
                    use_container_width=True
                )
    
    # ========================================================================
    # COMPARE INVESTMENTS PAGE
    # ========================================================================
    elif page == "⚖️ Compare Investments":
        st.header("Compare Investments")
        
        if len(st.session_state.investments) < 2:
            st.warning("You need at least 2 investments to compare. Create more investments first!")
            return
        
        # Select ranking metric
        ranking_metric = st.selectbox(
            "Rank By",
            ["NPV", "IRR", "Profitability Index"],
            help="Choose which metric to use for ranking"
        )
        
        metric_map = {
            "NPV": "npv",
            "IRR": "irr",
            "Profitability Index": "pi"
        }
        
        # Get all investments for comparison
        investments = [inv['investment'] for inv in st.session_state.investments]
        comparison = InvestmentComparator.compare_investments(
            investments,
            ranking_metric=metric_map[ranking_metric]
        )
        
        # Display ranking
        st.subheader(f"Investment Rankings (by {ranking_metric})")
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame([{
            'Rank': item['rank'],
            'Investment': item['investment_name'],
            'NPV': item['npv'],
            'IRR': item['irr'] if item['irr'] else 0,
            'Profitability Index': item['profitability_index'] if item['profitability_index'] else 0,
            'Payback (years)': item['payback_period'] if item['payback_period'] else float('inf'),
            'Recommendation': item['recommendation']
        } for item in comparison])
        
        # Style the dataframe
        def highlight_recommendation(val):
            color = 'background-color: #d4edda' if val == 'Accept' else 'background-color: #f8d7da'
            return color
        
        styled_df = comparison_df.style.format({
            'NPV': '${:,.0f}',
            'IRR': '{:.2%}',
            'Profitability Index': '{:.2f}',
            'Payback (years)': '{:.2f}'
        }).applymap(highlight_recommendation, subset=['Recommendation'])
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Comparison charts
        st.subheader("Visual Comparison")
        
        tab1, tab2, tab3 = st.tabs(["NPV Comparison", "IRR Comparison", "Multi-Metric"])
        
        with tab1:
            fig = px.bar(
                comparison_df,
                x='Investment',
                y='NPV',
                color='NPV',
                color_continuous_scale=['red', 'yellow', 'green'],
                title="NPV Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = px.bar(
                comparison_df,
                x='Investment',
                y='IRR',
                color='IRR',
                color_continuous_scale='Blues',
                title="IRR Comparison"
            )
            fig.update_yaxes(tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Radar chart for multi-metric comparison
            metrics_for_radar = comparison_df[['Investment', 'NPV', 'IRR', 'Profitability Index']].copy()
            
            # Normalize metrics to 0-100 scale for radar chart
            for col in ['NPV', 'IRR', 'Profitability Index']:
                max_val = metrics_for_radar[col].max()
                min_val = metrics_for_radar[col].min()
                if max_val != min_val:
                    metrics_for_radar[col] = ((metrics_for_radar[col] - min_val) / (max_val - min_val)) * 100
            
            fig = go.Figure()
            
            for idx, row in metrics_for_radar.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row['NPV'], row['IRR'], row['Profitability Index']],
                    theta=['NPV', 'IRR', 'Profitability Index'],
                    fill='toself',
                    name=row['Investment']
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                title="Multi-Metric Comparison (Normalized)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # MONTE CARLO SIMULATION PAGE
    # ========================================================================
    elif page == "🎲 Monte Carlo Simulation":
        st.header("Monte Carlo Simulation")
        st.markdown("**Model uncertainty in cash flows using probability distributions**")
        
        if not st.session_state.investments:
            st.warning("No investments available. Create one first!")
            return
        
        # Select base investment
        investment_names = [inv['investment'].name for inv in st.session_state.investments]
        selected_name = st.selectbox("Select Base Investment", investment_names)
        
        selected_inv_data = next(
            (inv for inv in st.session_state.investments 
             if inv['investment'].name == selected_name),
            None
        )
        
        if selected_inv_data:
            investment = selected_inv_data['investment']
            
            st.subheader("Configure Uncertainty")
            st.markdown("Define probability distributions for each year's cash flows:")
            
            with st.form("monte_carlo_form"):
                # Get cash flows (excluding initial investment)
                future_cash_flows = [cf for cf in investment.cash_flows if cf.period > 0]
                
                # Distribution type selector
                dist_type = st.selectbox(
                    "Distribution Type",
                    ["Normal", "Triangular", "Uniform"],
                    help="Select how cash flows vary"
                )
                
                # Uncertainty level
                uncertainty = st.slider(
                    "Uncertainty Level (%)",
                    min_value=5,
                    max_value=50,
                    value=20,
                    step=5,
                    help="Standard deviation as % of base value"
                ) / 100
                
                # Number of simulations
                num_sims = st.selectbox(
                    "Number of Simulations",
                    [1000, 5000, 10000, 25000],
                    index=2
                )
                
                # Discount rate uncertainty
                discount_uncertainty = st.checkbox(
                    "Include Discount Rate Uncertainty",
                    value=False,
                    help="Add uncertainty to the discount rate"
                )
                
                discount_std = 0.02
                if discount_uncertainty:
                    discount_std = st.slider(
                        "Discount Rate Std Dev",
                        min_value=0.01,
                        max_value=0.05,
                        value=0.02,
                        step=0.01,
                        format="%.2f"
                    )
                
                run_simulation = st.form_submit_button("Run Simulation", use_container_width=True)
                
                if run_simulation:
                    with st.spinner(f"Running {num_sims:,} simulations..."):
                        # Create probabilistic cash flows
                        prob_cash_flows = []
                        
                        for cf in future_cash_flows:
                            if dist_type == "Normal":
                                prob_cash_flows.append(ProbabilisticCashFlow(
                                    period=cf.period,
                                    distribution_type=DistributionType.NORMAL,
                                    parameters={
                                        'mean': cf.amount,
                                        'std_dev': abs(cf.amount * uncertainty)
                                    },
                                    description=cf.description
                                ))
                            elif dist_type == "Triangular":
                                prob_cash_flows.append(ProbabilisticCashFlow(
                                    period=cf.period,
                                    distribution_type=DistributionType.TRIANGULAR,
                                    parameters={
                                        'min': cf.amount * (1 - uncertainty),
                                        'mode': cf.amount,
                                        'max': cf.amount * (1 + uncertainty)
                                    },
                                    description=cf.description
                                ))
                            else:  # Uniform
                                prob_cash_flows.append(ProbabilisticCashFlow(
                                    period=cf.period,
                                    distribution_type=DistributionType.UNIFORM,
                                    parameters={
                                        'min': cf.amount * (1 - uncertainty),
                                        'max': cf.amount * (1 + uncertainty)
                                    },
                                    description=cf.description
                                ))
                        
                        # Create probabilistic investment
                        initial_inv = next(cf.amount for cf in investment.cash_flows if cf.period == 0)
                        prob_investment = ProbabilisticInvestment(
                            name=investment.name,
                            initial_investment=initial_inv,
                            probabilistic_cash_flows=prob_cash_flows,
                            discount_rate=investment.discount_rate,
                            discount_rate_uncertainty={'std_dev': discount_std} if discount_uncertainty else None
                        )
                        
                        # Run simulation
                        results = MonteCarloSimulator.run_simulation(
                            prob_investment,
                            num_simulations=num_sims,
                            random_seed=42
                        )
                        
                        # Display results
                        st.success("✅ Simulation Complete!")
                        
                        # Key statistics
                        st.subheader("Statistical Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Mean NPV", format_currency(results.mean_npv))
                            st.metric("Median NPV", format_currency(results.median_npv))
                        
                        with col2:
                            st.metric("Std Deviation", format_currency(results.std_npv))
                            st.metric("Min NPV", format_currency(results.min_npv))
                        
                        with col3:
                            st.metric("Max NPV", format_currency(results.max_npv))
                            st.metric("5th Percentile", format_currency(results.npv_5th_percentile))
                        
                        with col4:
                            prob_positive = results.probability_positive_npv * 100
                            st.metric("Probability NPV > 0", f"{prob_positive:.1f}%")
                            st.metric("Value at Risk (95%)", format_currency(results.value_at_risk_95))
                        
                        # Histogram
                        st.subheader("NPV Distribution")
                        fig = create_monte_carlo_histogram(results)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Percentiles table
                        with st.expander("📊 View Percentiles"):
                            percentiles = [5, 10, 25, 50, 75, 90, 95]
                            percentile_values = [
                                np.percentile(results.npv_distribution, p) 
                                for p in percentiles
                            ]
                            
                            perc_df = pd.DataFrame({
                                'Percentile': [f"{p}th" for p in percentiles],
                                'NPV': percentile_values
                            })
                            
                            st.dataframe(
                                perc_df.style.format({'NPV': '${:,.0f}'}),
                                use_container_width=True
                            )
                        
                        # Risk interpretation
                        st.subheader("Risk Assessment")
                        
                        if prob_positive > 80:
                            st.success(f"✅ **Low Risk**: {prob_positive:.1f}% probability of positive NPV")
                        elif prob_positive > 60:
                            st.warning(f"⚠️ **Moderate Risk**: {prob_positive:.1f}% probability of positive NPV")
                        else:
                            st.error(f"❌ **High Risk**: Only {prob_positive:.1f}% probability of positive NPV")
                        
                        # Coefficient of variation
                        cv = results.std_npv / results.mean_npv if results.mean_npv != 0 else float('inf')
                        st.info(f"**Coefficient of Variation**: {cv:.2f} (Lower is better - measures risk per unit of return)")
    
    # ========================================================================
    # SENSITIVITY ANALYSIS PAGE
    # ========================================================================
    elif page == "📈 Sensitivity Analysis":
        st.header("Sensitivity Analysis")
        st.markdown("**Analyze how NPV changes with different discount rates**")
        
        if not st.session_state.investments:
            st.warning("No investments available. Create one first!")
            return
        
        # Select investment
        investment_names = [inv['investment'].name for inv in st.session_state.investments]
        selected_name = st.selectbox("Select Investment", investment_names)
        
        selected_inv_data = next(
            (inv for inv in st.session_state.investments 
             if inv['investment'].name == selected_name),
            None
        )
        
        if selected_inv_data:
            investment = selected_inv_data['investment']
            
            col1, col2 = st.columns(2)
            
            with col1:
                min_rate = st.slider(
                    "Minimum Discount Rate (%)",
                    min_value=1,
                    max_value=20,
                    value=5
                ) / 100
            
            with col2:
                max_rate = st.slider(
                    "Maximum Discount Rate (%)",
                    min_value=10,
                    max_value=50,
                    value=25
                ) / 100
            
            num_points = st.slider("Number of Data Points", min_value=5, max_value=20, value=10)
            
            if st.button("Run Sensitivity Analysis", use_container_width=True):
                # Generate discount rates
                discount_rates = np.linspace(min_rate, max_rate, num_points)
                
                # Calculate NPV for each rate
                sensitivity_data = []
                for rate in discount_rates:
                    temp_investment = Investment(
                        name=investment.name,
                        cash_flows=investment.cash_flows.copy(),
                        discount_rate=rate
                    )
                    npv = DCFCalculator.calculate_npv(temp_investment)
                    irr = DCFCalculator.calculate_irr(temp_investment)
                    
                    sensitivity_data.append({
                        'discount_rate': rate,
                        'npv': npv,
                        'irr': irr,
                        'decision': 'Accept' if npv > 0 else 'Reject'
                    })
                
                # Display chart
                st.subheader("Sensitivity Chart")
                fig = create_sensitivity_chart(sensitivity_data)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display table
                st.subheader("Sensitivity Table")
                df = pd.DataFrame(sensitivity_data)
                
                def highlight_decision(val):
                    return 'background-color: #d4edda' if val == 'Accept' else 'background-color: #f8d7da'
                
                styled_df = df.style.format({
                    'discount_rate': '{:.2%}',
                    'npv': '${:,.0f}',
                    'irr': '{:.2%}'
                }).applymap(highlight_decision, subset=['decision'])
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Key insights
                st.subheader("Key Insights")
                
                # Find break-even rate (where NPV = 0)
                npv_values = df['npv'].values
                rates = df['discount_rate'].values
                
                if (npv_values > 0).any() and (npv_values < 0).any():
                    # Find approximate break-even
                    sign_changes = np.where(np.diff(np.sign(npv_values)))[0]
                    if len(sign_changes) > 0:
                        idx = sign_changes[0]
                        breakeven_rate = (rates[idx] + rates[idx + 1]) / 2
                        st.info(f"**Break-even Discount Rate**: ≈ {breakeven_rate:.2%}")
                        st.markdown(f"The investment is acceptable when discount rate < {breakeven_rate:.2%}")
    
    # ========================================================================
    # MANAGE INVESTMENTS PAGE
    # ========================================================================
    elif page == "💾 Manage Investments":
        st.header("Manage Investments")
        
        if not st.session_state.investments:
            st.info("No investments to manage yet.")
            return
        
        st.subheader("All Investments")
        
        # Create summary table
        for idx, inv_data in enumerate(st.session_state.investments):
            investment = inv_data['investment']
            metrics = inv_data['metrics']
            
            with st.expander(f"📁 {investment.name}", expanded=False):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**Description:** {inv_data['description']}")
                    st.markdown(f"**Discount Rate:** {format_percentage(investment.discount_rate)}")
                    st.markdown(f"**NPV:** {format_currency(metrics['npv'])}")
                    st.markdown(f"**IRR:** {format_percentage(metrics['irr']) if metrics['irr'] else 'N/A'}")
                
                with col2:
                    st.markdown(f"**Initial Investment:** {format_currency(investment.cash_flows[0].amount)}")
                    st.markdown(f"**Time Horizon:** {max(cf.period for cf in investment.cash_flows)} years")
                
                with col3:
                    if st.button("🗑️ Delete", key=f"delete_{idx}"):
                        st.session_state.investments.pop(idx)
                        st.rerun()
        
        # Bulk actions
        st.subheader("Bulk Actions")
        if st.button("🗑️ Clear All Investments"):
            if st.checkbox("Yes, I'm sure I want to delete all investments"):
                st.session_state.investments = []
                st.session_state.investment_counter = 0
                st.success("All investments deleted!")
                st.rerun()

if __name__ == "__main__":
    main()
