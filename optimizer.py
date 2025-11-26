import pandas as pd
import numpy as np
from itertools import combinations
from snowflake.snowpark.context import get_active_session
import streamlit as st
import random

# === GLOBAL CONSTANTS ===
INTERVAL_ORDER_11 = ['Jan-Feb', 'Feb-Mar', 'Mar-Apr', 'Apr-May', 'May-Jun',
                     'Jun-Jul', 'Jul-Aug', 'Aug-Sep', 'Sep-Oct', 'Oct-Nov', 'Nov-Dec']

st.set_page_config(layout="wide", page_title="PRF Grid Optimizer")

# =============================================================================
# === 1. CACHED DATA-LOADING FUNCTIONS (Aligned with Study App) ===
# =============================================================================

@st.cache_data(ttl=3600)
def load_distinct_grids(_session):
    """Fetches the list of all available Grid IDs from the data."""
    query = """
        SELECT DISTINCT GRID_ID 
        FROM CAPITAL_MARKETS_SANDBOX.PUBLIC.RAIN_INDEX_PLATINUM_ENHANCED
        ORDER BY GRID_ID
    """
    df = _session.sql(query).to_pandas()
    return df['GRID_ID'].tolist()

@st.cache_data(ttl=3600)
def load_all_indices(_session, grid_id):
    """Fetches all historical rainfall data for a single grid."""
    all_indices_query = f"""
        SELECT 
            YEAR, INTERVAL_NAME, INDEX_VALUE, INTERVAL_CODE, INTERVAL_MAPPING_TS_TEXT
        FROM RAIN_INDEX_PLATINUM_ENHANCED 
        WHERE GRID_ID = {grid_id}
        ORDER BY YEAR, INTERVAL_CODE
    """
    df = _session.sql(all_indices_query).to_pandas()
    df['INDEX_VALUE'] = pd.to_numeric(df['INDEX_VALUE'], errors='coerce')
    df = df.dropna(subset=['INDEX_VALUE'])
    return df

@st.cache_data(ttl=3600)
def load_county_base_value(_session, grid_id):
    """Fetches the average county base value for the grid (aligned with study app)."""
    base_value_query = f"""
        SELECT AVG(COUNTY_BASE_VALUE) 
        FROM COUNTY_BASE_VALUES_PLATINUM 
        WHERE SUB_COUNTY_CODE = {grid_id}
    """
    return float(_session.sql(base_value_query).to_pandas().iloc[0, 0])

@st.cache_data(ttl=3600)
def get_current_rate_year(_session):
    """Finds the most recent year in the premium rates table."""
    return int(_session.sql("SELECT MAX(YEAR) FROM PRF_PREMIUM_RATES").to_pandas().iloc[0, 0])

@st.cache_data(ttl=3600)
def load_premium_rates(_session, grid_id, use, coverage_levels_list, year):
    """Fetches premium rates for all specified coverage levels."""
    all_premiums = {}
    for cov_level in coverage_levels_list:
        cov_string = f"{cov_level:.0%}"
        premium_query = f"""
            SELECT INDEX_INTERVAL_NAME, PREMIUMRATE 
            FROM PRF_PREMIUM_RATES 
            WHERE GRID_ID = {grid_id}
              AND INTENDED_USE = '{use}'
              AND COVERAGE_LEVEL = '{cov_string}'
              AND YEAR = {year}
        """
        prem_df = _session.sql(premium_query).to_pandas()
        prem_df['PREMIUMRATE'] = pd.to_numeric(prem_df['PREMIUMRATE'], errors='coerce')
        all_premiums[cov_level] = prem_df.set_index('INDEX_INTERVAL_NAME')['PREMIUMRATE'].to_dict()
    return all_premiums

@st.cache_data(ttl=3600)
def load_subsidies(_session, plan_code, coverage_levels_list):
    """Fetches subsidy percentages for all specified coverage levels."""
    all_subsidies = {}
    for cov_level in coverage_levels_list:
        subsidy_query = f"""
            SELECT SUBSIDY_PERCENT 
            FROM SUBSIDYPERCENT_YTD_PLATINUM 
            WHERE INSURANCE_PLAN_CODE = {plan_code}
              AND COVERAGE_LEVEL_PERCENT = {cov_level}
            LIMIT 1
        """
        all_subsidies[cov_level] = float(_session.sql(subsidy_query).to_pandas().iloc[0, 0])
    return all_subsidies

# =============================================================================
# === 2. HELPER FUNCTIONS ===
# =============================================================================

def is_adjacent(interval1, interval2):
    """Check if two intervals are adjacent (excluding Nov-Dec/Jan-Feb wrap which is allowed)."""
    try:
        idx1 = INTERVAL_ORDER_11.index(interval1)
        idx2 = INTERVAL_ORDER_11.index(interval2)
    except ValueError:
        return False
    
    diff = abs(idx1 - idx2)
    # Adjacent if diff == 1, but NOT the wrap-around (diff == 10 means Nov-Dec and Jan-Feb)
    return diff == 1

def has_adjacent_intervals(intervals_list):
    """Check if any intervals in the list are adjacent (excluding Nov-Dec/Jan-Feb wrap)."""
    for i in range(len(intervals_list)):
        for j in range(i + 1, len(intervals_list)):
            if is_adjacent(intervals_list[i], intervals_list[j]):
                return True
    return False

def generate_random_valid_allocation(min_intervals, max_intervals):
    """
    Generate a random valid allocation using Monte Carlo approach.
    Rules: 10-50% per interval (or 0%), no adjacent intervals, total = 100%.
    """
    num_intervals = random.randint(min_intervals, max_intervals)
    
    # Build list of valid non-adjacent interval combinations
    available = list(range(11))
    selected_indices = []
    
    attempts = 0
    while len(selected_indices) < num_intervals and attempts < 100:
        attempts += 1
        if not available:
            break
        
        idx = random.choice(available)
        selected_indices.append(idx)
        
        # Remove this index and adjacent ones from available
        to_remove = [idx]
        if idx > 0:
            to_remove.append(idx - 1)
        if idx < 10:
            to_remove.append(idx + 1)
        
        available = [i for i in available if i not in to_remove]
    
    if len(selected_indices) < min_intervals:
        return None
    
    selected_intervals = [INTERVAL_ORDER_11[i] for i in selected_indices]
    
    # Generate random percentages (10-50% each, sum to 100%)
    # Use Dirichlet-like approach then clamp
    weights = []
    remaining = 100
    
    for i in range(len(selected_intervals) - 1):
        # Each interval needs at least 10%, max 50%
        intervals_left = len(selected_intervals) - i
        min_for_rest = 10 * (intervals_left - 1)
        max_available = remaining - min_for_rest
        
        min_pct = 10
        max_pct = min(50, max_available)
        
        if min_pct > max_pct:
            return None
        
        pct = random.randint(min_pct, max_pct)
        weights.append(pct)
        remaining -= pct
    
    # Last interval gets the remainder
    if remaining < 10 or remaining > 50:
        return None
    
    weights.append(remaining)
    
    # Build allocation dict
    allocation = {interval: 0.0 for interval in INTERVAL_ORDER_11}
    for i, interval in enumerate(selected_intervals):
        allocation[interval] = weights[i] / 100.0
    
    return allocation

def generate_systematic_allocations(intervals_list, num_intervals):
    """Generate systematic allocation patterns for a given interval combination."""
    allocations = []
    
    if num_intervals == 2:
        allocations.append({intervals_list[0]: 0.50, intervals_list[1]: 0.50})
    
    elif num_intervals == 3:
        splits = [
            (34, 33, 33),
            (50, 25, 25),
            (40, 30, 30),
            (50, 30, 20),
            (40, 40, 20),
        ]
        for s in splits:
            allocations.append({intervals_list[i]: s[i] / 100.0 for i in range(3)})
    
    elif num_intervals == 4:
        splits = [
            (25, 25, 25, 25),
            (50, 20, 20, 10),
            (40, 20, 20, 20),
            (30, 30, 20, 20),
            (50, 20, 15, 15),
            (40, 25, 20, 15),
            (35, 25, 25, 15),
        ]
        for s in splits:
            allocations.append({intervals_list[i]: s[i] / 100.0 for i in range(4)})
    
    elif num_intervals == 5:
        splits = [
            (20, 20, 20, 20, 20),
            (50, 15, 15, 10, 10),
            (30, 20, 20, 15, 15),
            (40, 15, 15, 15, 15),
            (25, 25, 20, 15, 15),
            (35, 20, 15, 15, 15),
            (30, 25, 20, 15, 10),
        ]
        for s in splits:
            allocations.append({intervals_list[i]: s[i] / 100.0 for i in range(5)})
    
    elif num_intervals == 6:
        splits = [
            (17, 17, 17, 17, 16, 16),
            (50, 10, 10, 10, 10, 10),
            (30, 15, 15, 15, 15, 10),
            (25, 20, 15, 15, 15, 10),
            (20, 20, 15, 15, 15, 15),
            (25, 20, 20, 15, 10, 10),
            (30, 20, 15, 15, 10, 10),
        ]
        for s in splits:
            allocations.append({intervals_list[i]: s[i] / 100.0 for i in range(6)})
    
    return allocations

# =============================================================================
# === 3. OPTIMIZATION ENGINE ===
# =============================================================================

def run_optimization(
    _session, grid_id, start_year, end_year, plan_code, prod_factor,
    acres, use, coverage_levels, objective, min_intervals, max_intervals, 
    num_monte_carlo
):
    """
    Run optimization for a single grid using Monte Carlo + systematic search.
    Returns: (results_df, yearly_details_dict, rate_year, strategy_count)
    """
    # 1. Load data
    county_base_value = load_county_base_value(_session, grid_id)
    all_indices_df = load_all_indices(_session, grid_id)
    all_indices_df = all_indices_df[
        (all_indices_df['YEAR'] >= start_year) & 
        (all_indices_df['YEAR'] <= end_year)
    ]
    
    current_rate_year = get_current_rate_year(_session)
    all_premiums = load_premium_rates(_session, grid_id, use, coverage_levels, current_rate_year)
    all_subsidies = load_subsidies(_session, plan_code, coverage_levels)
    
    # 2. Score intervals by average shortage (for systematic search)
    interval_scores = {}
    for interval in INTERVAL_ORDER_11:
        interval_data = all_indices_df[all_indices_df['INTERVAL_NAME'] == interval]['INDEX_VALUE']
        avg_shortage = (100 - interval_data).mean() if len(interval_data) > 0 else 0
        interval_scores[interval] = avg_shortage
    
    sorted_intervals = sorted(interval_scores.items(), key=lambda x: x[1], reverse=True)
    top_intervals = [x[0] for x in sorted_intervals[:8]]  # Top 8 driest
    
    # 3. Generate candidate strategies
    candidates = []
    
    # 3a. Systematic combinations of top intervals
    for num_intervals in range(min_intervals, max_intervals + 1):
        for combo in combinations(top_intervals, num_intervals):
            combo_list = list(combo)
            
            # Skip if has adjacent intervals
            if has_adjacent_intervals(combo_list):
                continue
            
            candidates.extend(generate_systematic_allocations(combo_list, num_intervals))
    
    # 3b. Monte Carlo random allocations
    for _ in range(num_monte_carlo):
        alloc = generate_random_valid_allocation(min_intervals, max_intervals)
        if alloc is not None:
            candidates.append(alloc)
    
    # Deduplicate
    unique_candidates = []
    seen = set()
    for candidate in candidates:
        # Round to avoid floating point issues
        key = tuple(sorted((k, round(v, 3)) for k, v in candidate.items() if v > 0))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)
    
    # 4. Evaluate all candidates
    results = []
    yearly_details = {}  # Store yearly breakdown for best strategies
    
    def calculate_roi_for_strategy(allocation, coverage_level):
        """Calculate multi-year ROI for a given allocation and coverage level."""
        subsidy = all_subsidies[coverage_level]
        premiums = all_premiums[coverage_level]
        dollar_protection = county_base_value * coverage_level * prod_factor
        total_protection = dollar_protection * acres
        
        year_rois = []
        year_details = []
        total_indemnity_all_years = 0
        total_producer_premium_all_years = 0
        
        for year in range(start_year, end_year + 1):
            year_data = all_indices_df[all_indices_df['YEAR'] == year]
            if year_data.empty:
                continue
            
            total_indemnity, total_producer_premium = 0, 0
            
            # Validate allocation
            if abs(sum(allocation.values()) - 1.0) > 0.01:
                return None, None
            
            for interval, pct in allocation.items():
                if pct == 0:
                    continue
                if pct > 0.51:
                    return None, None
                
                index_row = year_data[year_data['INTERVAL_NAME'] == interval]
                index_value = float(index_row['INDEX_VALUE'].iloc[0]) if not index_row.empty else 100
                
                premium_rate = premiums.get(interval, 0)
                interval_protection = total_protection * pct
                total_premium = interval_protection * premium_rate
                producer_premium = total_premium - (total_premium * subsidy)
                
                trigger = coverage_level * 100
                shortfall_pct = max(0, (trigger - index_value) / trigger)
                indemnity = shortfall_pct * interval_protection
                
                total_indemnity += indemnity
                total_producer_premium += producer_premium
            
            year_roi = (total_indemnity - total_producer_premium) / total_producer_premium if total_producer_premium > 0 else 0
            year_rois.append(year_roi)
            
            year_details.append({
                'Year': year,
                'Total Indemnity': total_indemnity,
                'Producer Premium': total_producer_premium,
                'Net Return': total_indemnity - total_producer_premium,
                'ROI': year_roi
            })
            
            total_indemnity_all_years += total_indemnity
            total_producer_premium_all_years += total_producer_premium
        
        if len(year_rois) == 0:
            return None, None
        
        year_rois_array = np.array(year_rois)
        
        average_roi = year_rois_array.mean()
        cumulative_roi = (total_indemnity_all_years - total_producer_premium_all_years) / total_producer_premium_all_years if total_producer_premium_all_years > 0 else 0
        std_dev = year_rois_array.std()
        risk_adj_ret = cumulative_roi / std_dev if std_dev > 0 else 0
        
        metrics = {
            'average_roi': average_roi,
            'median_roi': np.median(year_rois_array),
            'cumulative_roi': cumulative_roi,
            'profitable_pct': (year_rois_array > 0).sum() / len(year_rois_array),
            'std_dev': std_dev,
            'min_roi': year_rois_array.min(),
            'max_roi': year_rois_array.max(),
            'risk_adj_ret': risk_adj_ret,
            'total_indemnity': total_indemnity_all_years,
            'total_premium': total_producer_premium_all_years
        }
        
        return metrics, year_details
    
    for coverage_level in coverage_levels:
        for allocation in unique_candidates:
            metrics, year_detail = calculate_roi_for_strategy(allocation, coverage_level)
            if metrics is not None:
                result_entry = {
                    'coverage_level': coverage_level,
                    'allocation': allocation,
                    **metrics
                }
                results.append(result_entry)
                
                # Store yearly details keyed by (coverage, allocation tuple)
                alloc_key = tuple(sorted((k, round(v, 3)) for k, v in allocation.items() if v > 0))
                yearly_details[(coverage_level, alloc_key)] = year_detail
    
    if len(results) == 0:
        return pd.DataFrame(), {}, current_rate_year, len(unique_candidates)
    
    results_df = pd.DataFrame(results).sort_values(objective, ascending=False)
    
    return results_df, yearly_details, current_rate_year, len(unique_candidates)

# =============================================================================
# === 4. MAIN APP ===
# =============================================================================

def main():
    st.title("PRF Grid Optimizer")
    st.caption("Monte Carlo optimization across multiple grids")
    
    session = get_active_session()
    
    # Load valid grids
    try:
        valid_grids = load_distinct_grids(session)
    except Exception as e:
        st.error(f"Could not load Grid ID list: {e}")
        st.stop()
    
    # === SIDEBAR: Common Parameters ===
    st.sidebar.header("Common Parameters")
    
    # Initialize session state
    if 'productivity_factor' not in st.session_state:
        st.session_state.productivity_factor = 1.0
    if 'total_insured_acres' not in st.session_state:
        st.session_state.total_insured_acres = 1000
    if 'intended_use' not in st.session_state:
        st.session_state.intended_use = 'Grazing'
    if 'insurance_plan_code' not in st.session_state:
        st.session_state.insurance_plan_code = 13
    
    # Productivity Factor
    prod_options = list(range(60, 151))
    prod_options_formatted = [f"{x}%" for x in prod_options]
    try:
        current_prod_index = prod_options.index(int(st.session_state.productivity_factor * 100))
    except ValueError:
        current_prod_index = 40
    selected_prod_str = st.sidebar.selectbox(
        "Productivity Factor", options=prod_options_formatted, index=current_prod_index,
        key="sidebar_prod_factor"
    )
    productivity_factor = int(selected_prod_str.replace('%', '')) / 100.0
    
    total_insured_acres = st.sidebar.number_input(
        "Total Insured Acres", value=st.session_state.total_insured_acres, step=10,
        key="sidebar_acres"
    )
    
    intended_use = st.sidebar.selectbox(
        "Intended Use", ['Grazing', 'Haying'],
        index=0 if st.session_state.intended_use == 'Grazing' else 1,
        key="sidebar_use"
    )
    
    plan_code = st.sidebar.number_input(
        "Insurance Plan Code",
        value=st.session_state.insurance_plan_code,
        disabled=True
    )
    
    st.sidebar.divider()
    st.sidebar.caption("*Uses current year rates")
    
    # === MAIN AREA: Optimizer Settings ===
    st.subheader("Optimization Settings")
    
    # Grid Selection
    selected_grids = st.multiselect(
        "Select Grids to Optimize",
        options=valid_grids,
        default=[valid_grids[0]] if valid_grids else [],
        help="Select one or more grids. Each will be optimized independently."
    )
    
    if not selected_grids:
        st.warning("Select at least one grid to continue.")
        st.stop()
    
    # Year Range
    col1, col2 = st.columns(2)
    start_year = col1.selectbox("Start Year", list(range(1948, 2026)), index=50, key="opt_start")
    end_year = col2.selectbox("End Year", list(range(1948, 2026)), index=76, key="opt_end")
    
    # Coverage Levels
    coverage_levels = st.multiselect(
        "Coverage Levels to Test",
        [0.70, 0.75, 0.80, 0.85, 0.90],
        default=[0.70, 0.75, 0.80, 0.85, 0.90],
        format_func=lambda x: f"{x:.0%}",
        key="opt_coverage"
    )
    
    if not coverage_levels:
        st.error("Select at least one coverage level.")
        st.stop()
    
    # Objective
    objective = st.selectbox(
        "Optimization Objective",
        ['cumulative_roi', 'median_roi', 'profitable_pct', 'risk_adj_ret'],
        index=0,
        format_func=lambda x: {
            'cumulative_roi': 'Cumulative ROI',
            'median_roi': 'Median ROI',
            'profitable_pct': 'Win Rate',
            'risk_adj_ret': 'Risk-Adjusted Return'
        }.get(x, x),
        key="opt_objective"
    )
    
    # Interval Bounds
    st.markdown("**Interval Constraints**")
    col1, col2 = st.columns(2)
    min_intervals = col1.number_input("Min Intervals", min_value=2, max_value=6, value=4, step=1)
    max_intervals = col2.number_input("Max Intervals", min_value=2, max_value=6, value=6, step=1)
    
    if min_intervals > max_intervals:
        st.error("Min intervals cannot exceed max intervals.")
        st.stop()
    
    # Search Depth (Monte Carlo iterations)
    search_depth_map = {
        'Fast (500)': 500,
        'Standard (1000)': 1000,
        'Thorough (2500)': 2500,
        'Maximum (5000)': 5000
    }
    search_depth_key = st.select_slider(
        "Monte Carlo Iterations",
        options=list(search_depth_map.keys()),
        value='Standard (1000)',
        key="opt_depth"
    )
    num_monte_carlo = search_depth_map[search_depth_key]
    
    st.divider()
    
    # === RUN OPTIMIZATION ===
    if st.button("Run Optimization", type="primary", key="run_opt"):
        
        all_grid_results = {}
        
        progress_bar = st.progress(0, text="Starting optimization...")
        
        for idx, grid_id in enumerate(selected_grids):
            progress_bar.progress(
                (idx) / len(selected_grids),
                text=f"Optimizing Grid {grid_id}..."
            )
            
            try:
                results_df, yearly_details, rate_year, strategy_count = run_optimization(
                    session, grid_id, start_year, end_year, plan_code,
                    productivity_factor, total_insured_acres, intended_use,
                    coverage_levels, objective, min_intervals, max_intervals,
                    num_monte_carlo
                )
                
                all_grid_results[grid_id] = {
                    'results_df': results_df,
                    'yearly_details': yearly_details,
                    'rate_year': rate_year,
                    'strategy_count': strategy_count
                }
                
            except Exception as e:
                st.error(f"Error optimizing Grid {grid_id}: {e}")
                all_grid_results[grid_id] = {'error': str(e)}
        
        progress_bar.progress(1.0, text="Complete!")
        
        # Store results
        st.session_state.optimization_results = {
            'all_grid_results': all_grid_results,
            'selected_grids': selected_grids,
            'start_year': start_year,
            'end_year': end_year,
            'objective': objective,
            'coverage_levels': coverage_levels,
            'productivity_factor': productivity_factor,
            'total_insured_acres': total_insured_acres,
            'intended_use': intended_use
        }
    
    # === DISPLAY RESULTS ===
    if 'optimization_results' in st.session_state and st.session_state.optimization_results:
        r = st.session_state.optimization_results
        
        st.divider()
        st.header(f"Optimization Results ({r['start_year']}-{r['end_year']})")
        
        objective_display = {
            'cumulative_roi': 'Cumulative ROI',
            'median_roi': 'Median ROI',
            'profitable_pct': 'Win Rate',
            'risk_adj_ret': 'Risk-Adjusted Return'
        }
        
        st.caption(f"Optimized for: **{objective_display.get(r['objective'], r['objective'])}** | "
                   f"Productivity: {r['productivity_factor']:.0%} | "
                   f"Acres: {r['total_insured_acres']:,} | "
                   f"Use: {r['intended_use']}")
        
        # === GRID RANKING SUMMARY ===
        st.subheader("Grid Ranking Summary")
        
        ranking_data = []
        for grid_id in r['selected_grids']:
            grid_data = r['all_grid_results'].get(grid_id, {})
            
            if 'error' in grid_data:
                ranking_data.append({
                    'Grid': grid_id,
                    'Best Coverage': 'ERROR',
                    'Cumulative ROI': 0,
                    'Median ROI': 0,
                    'Win Rate': 0,
                    'Risk-Adj': 0,
                    'Strategies Tested': 0
                })
            elif grid_data.get('results_df') is not None and not grid_data['results_df'].empty:
                best = grid_data['results_df'].iloc[0]
                ranking_data.append({
                    'Grid': grid_id,
                    'Best Coverage': f"{best['coverage_level']:.0%}",
                    'Cumulative ROI': best['cumulative_roi'],
                    'Median ROI': best['median_roi'],
                    'Win Rate': best['profitable_pct'],
                    'Risk-Adj': best['risk_adj_ret'],
                    'Strategies Tested': grid_data['strategy_count']
                })
            else:
                ranking_data.append({
                    'Grid': grid_id,
                    'Best Coverage': 'N/A',
                    'Cumulative ROI': 0,
                    'Median ROI': 0,
                    'Win Rate': 0,
                    'Risk-Adj': 0,
                    'Strategies Tested': 0
                })
        
        ranking_df = pd.DataFrame(ranking_data)
        
        # Sort by the objective
        obj_col_map = {
            'cumulative_roi': 'Cumulative ROI',
            'median_roi': 'Median ROI',
            'profitable_pct': 'Win Rate',
            'risk_adj_ret': 'Risk-Adj'
        }
        sort_col = obj_col_map.get(r['objective'], 'Cumulative ROI')
        ranking_df = ranking_df.sort_values(sort_col, ascending=False)
        
        # Display ranking table
        st.dataframe(
            ranking_df.style.format({
                'Cumulative ROI': '{:.2%}',
                'Median ROI': '{:.2%}',
                'Win Rate': '{:.1%}',
                'Risk-Adj': '{:.2f}',
                'Strategies Tested': '{:,}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Download ranking
        csv_ranking = ranking_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Grid Ranking CSV",
            data=csv_ranking,
            file_name=f"grid_ranking_{r['start_year']}-{r['end_year']}.csv",
            mime="text/csv"
        )
        
        st.divider()
        
        # === DETAILED RESULTS PER GRID ===
        st.subheader("Best Strategy by Grid")
        
        for grid_id in r['selected_grids']:
            grid_data = r['all_grid_results'].get(grid_id, {})
            
            if 'error' in grid_data:
                st.error(f"Grid {grid_id}: {grid_data['error']}")
                continue
            
            results_df = grid_data.get('results_df')
            if results_df is None or results_df.empty:
                st.warning(f"Grid {grid_id}: No valid strategies found.")
                continue
            
            best = results_df.iloc[0]
            
            with st.container(border=True):
                st.markdown(f"### Grid {grid_id}")
                st.caption(f"Tested {grid_data['strategy_count']:,} strategies")
                
                # Metrics row
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Coverage Level", f"{best['coverage_level']:.0%}")
                c2.metric("Median ROI", f"{best['median_roi']:.2%}")
                c3.metric("Cumulative ROI", f"{best['cumulative_roi']:.2%}")
                c4.metric("Win Rate", f"{best['profitable_pct']:.1%}")
                c5.metric("Risk Adj Return", f"{best['risk_adj_ret']:.2f}")
                
                # Allocation display
                with st.expander("Show Allocation & Details"):
                    alloc_str = ", ".join([
                        f"{k}: {v*100:.0f}%" 
                        for k, v in sorted(best['allocation'].items(), key=lambda x: x[1], reverse=True) 
                        if v > 0
                    ])
                    st.markdown(f"**Allocation:** {alloc_str}")
                    st.text(f"Std Dev: {best['std_dev']:.2%} | Best Year: {best['max_roi']:.2%} | Worst Year: {best['min_roi']:.2%}")
                    st.text(f"Total Indemnity: ${best['total_indemnity']:,.0f} | Total Premium: ${best['total_premium']:,.0f}")
        
        st.divider()
        
        # === AUDITABLE YEARLY DETAIL ===
        st.subheader("📊 Yearly Performance Audit")
        st.caption("Year-by-year breakdown for the best strategy of each grid.")
        
        # Build master audit dataframe
        audit_rows = []
        
        for grid_id in r['selected_grids']:
            grid_data = r['all_grid_results'].get(grid_id, {})
            
            if 'error' in grid_data:
                continue
            
            results_df = grid_data.get('results_df')
            if results_df is None or results_df.empty:
                continue
            
            best = results_df.iloc[0]
            yearly_details = grid_data.get('yearly_details', {})
            
            # Get yearly details for best strategy
            alloc_key = tuple(sorted((k, round(v, 3)) for k, v in best['allocation'].items() if v > 0))
            year_data = yearly_details.get((best['coverage_level'], alloc_key), [])
            
            for yr in year_data:
                audit_rows.append({
                    'Grid': grid_id,
                    'Year': yr['Year'],
                    'Total Indemnity': yr['Total Indemnity'],
                    'Producer Premium': yr['Producer Premium'],
                    'Net Return': yr['Net Return'],
                    'ROI': yr['ROI']
                })
        
        if audit_rows:
            audit_df = pd.DataFrame(audit_rows)
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                filter_grids = st.multiselect(
                    "Filter by Grid:",
                    options=sorted(audit_df['Grid'].unique()),
                    default=sorted(audit_df['Grid'].unique()),
                    key="audit_grid_filter"
                )
            
            # Apply filter
            filtered_audit = audit_df[audit_df['Grid'].isin(filter_grids)]
            
            # Display
            st.dataframe(
                filtered_audit.style.format({
                    'Year': '{:.0f}',
                    'Total Indemnity': '${:,.0f}',
                    'Producer Premium': '${:,.0f}',
                    'Net Return': '${:,.0f}',
                    'ROI': '{:.2%}'
                }),
                use_container_width=True,
                height=400,
                hide_index=True
            )
            
            # Download audit
            csv_audit = filtered_audit.to_csv(index=False)
            st.download_button(
                label="📥 Download Yearly Audit CSV",
                data=csv_audit,
                file_name=f"yearly_audit_{r['start_year']}-{r['end_year']}.csv",
                mime="text/csv"
            )
            
            # Summary stats per grid
            st.markdown("**Summary by Grid:**")
            summary_data = []
            for grid_id in filter_grids:
                grid_audit = filtered_audit[filtered_audit['Grid'] == grid_id]
                total_indem = grid_audit['Total Indemnity'].sum()
                total_prem = grid_audit['Producer Premium'].sum()
                total_net = grid_audit['Net Return'].sum()
                cum_roi = total_net / total_prem if total_prem > 0 else 0
                
                summary_data.append({
                    'Grid': grid_id,
                    'Years': len(grid_audit),
                    'Total Indemnity': total_indem,
                    'Total Premium': total_prem,
                    'Net Return': total_net,
                    'Cumulative ROI': cum_roi
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(
                summary_df.style.format({
                    'Total Indemnity': '${:,.0f}',
                    'Total Premium': '${:,.0f}',
                    'Net Return': '${:,.0f}',
                    'Cumulative ROI': '{:.2%}'
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No yearly data available.")

if __name__ == "__main__":
    main()
