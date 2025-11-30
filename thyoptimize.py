import pandas as pd
import numpy as np
import random
from itertools import combinations
from snowflake.snowpark.context import get_active_session
import streamlit as st

# === GLOBAL CONSTANTS ===
INTERVAL_ORDER_11 = ['Jan-Feb', 'Feb-Mar', 'Mar-Apr', 'Apr-May', 'May-Jun',
                     'Jun-Jul', 'Jul-Aug', 'Aug-Sep', 'Sep-Oct', 'Oct-Nov', 'Nov-Dec']

st.set_page_config(layout="wide", page_title="PRF Grid Optimizer")

# === SESSION STATE INITIALIZATION ===
if 'expander_states' not in st.session_state:
    st.session_state.expander_states = {}

# =============================================================================
# === 1. CACHED DATA-LOADING FUNCTIONS (EXACT MATCH TO STUDY APP) ===
# =============================================================================

def extract_numeric_grid_id(grid_id):
    """Extract numeric grid ID from formatted string like '9128 (Kleberg - TX)'."""
    if isinstance(grid_id, str):
        return int(grid_id.split('(')[0].strip())
    return int(grid_id)

@st.cache_data(ttl=3600)
def load_distinct_grids(_session):
    """Fetches the list of all available Grid IDs from PRF_COUNTY_BASE_VALUES."""
    query = """
        SELECT DISTINCT GRID_ID 
        FROM CAPITAL_MARKETS_SANDBOX.PUBLIC.PRF_COUNTY_BASE_VALUES
        ORDER BY GRID_ID
    """
    df = _session.sql(query).to_pandas()
    return df['GRID_ID'].tolist()

@st.cache_data(ttl=3600)
def load_all_indices(_session, grid_id):
    """Fetches all historical rainfall data for a single grid."""
    numeric_id = extract_numeric_grid_id(grid_id)
    all_indices_query = f"""
        SELECT 
            YEAR, INTERVAL_NAME, INDEX_VALUE, INTERVAL_CODE, INTERVAL_MAPPING_TS_TEXT
        FROM RAIN_INDEX_PLATINUM_ENHANCED 
        WHERE GRID_ID = {numeric_id}
        ORDER BY YEAR, INTERVAL_CODE
    """
    df = _session.sql(all_indices_query).to_pandas()
    df['INDEX_VALUE'] = pd.to_numeric(df['INDEX_VALUE'], errors='coerce')
    df = df.dropna(subset=['INDEX_VALUE'])
    return df

@st.cache_data(ttl=3600)
def load_county_base_value(_session, grid_id):
    """Fetches the average county base value for the grid."""
    base_value_query = f"""
        SELECT AVG(COUNTY_BASE_VALUE) 
        FROM PRF_COUNTY_BASE_VALUES 
        WHERE GRID_ID = '{grid_id}'
    """
    return float(_session.sql(base_value_query).to_pandas().iloc[0, 0])

@st.cache_data(ttl=3600)
def get_current_rate_year(_session):
    """Finds the most recent year in the premium rates table."""
    return int(_session.sql("SELECT MAX(YEAR) FROM PRF_PREMIUM_RATES").to_pandas().iloc[0, 0])

@st.cache_data(ttl=3600)
def load_premium_rates(_session, grid_id, use, coverage_levels_list, year):
    """Fetches premium rates for all specified coverage levels."""
    numeric_id = extract_numeric_grid_id(grid_id)
    all_premiums = {}
    for cov_level in coverage_levels_list:
        cov_string = f"{cov_level:.0%}"
        premium_query = f"""
            SELECT INDEX_INTERVAL_NAME, PREMIUMRATE 
            FROM PRF_PREMIUM_RATES 
            WHERE GRID_ID = {numeric_id}
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
    """Check if two intervals are adjacent, with wrap-around"""
    try:
        idx1 = INTERVAL_ORDER_11.index(interval1)
        idx2 = INTERVAL_ORDER_11.index(interval2)
    except ValueError:
        return False
    
    diff = abs(idx1 - idx2)
    return diff == 1 or diff == (len(INTERVAL_ORDER_11) - 1)

def has_adjacent_intervals_in_list(intervals_list):
    """Check if any intervals in the list are adjacent (excluding Nov-Dec/Jan-Feb wrap)"""
    for i in range(len(intervals_list)):
        for j in range(i + 1, len(intervals_list)):
            interval1 = intervals_list[i]
            interval2 = intervals_list[j]
            
            if is_adjacent(interval1, interval2):
                # Allow Nov-Dec and Jan-Feb together (wrap-around exception)
                if {interval1, interval2} == {'Nov-Dec', 'Jan-Feb'}:
                    continue
                else:
                    return True
    return False

def generate_allocations(intervals_to_use, num_intervals):
    """Generate allocation percentages for N intervals (from study app)."""
    allocations = []

    if num_intervals == 2:
        allocations.append({intervals_to_use[0]: 0.50, intervals_to_use[1]: 0.50})

    elif num_intervals == 3:
        splits = [(0.34, 0.33, 0.33), (0.50, 0.25, 0.25)]
        for s in splits:
            allocations.append({intervals_to_use[i]: s[i] for i in range(3)})

    elif num_intervals == 4:
        splits = [(0.25, 0.25, 0.25, 0.25), (0.50, 0.20, 0.20, 0.10)]
        for s in splits:
            allocations.append({intervals_to_use[i]: s[i] for i in range(4)})

    elif num_intervals == 5:
        splits = [(0.20, 0.20, 0.20, 0.20, 0.20), (0.50, 0.125, 0.125, 0.125, 0.125)]
        for s in splits:
            allocations.append({intervals_to_use[i]: s[i] for i in range(5)})

    elif num_intervals == 6:
        splits = [(0.17, 0.17, 0.17, 0.17, 0.16, 0.16), (0.50, 0.10, 0.10, 0.10, 0.10, 0.10)]
        for s in splits:
            allocations.append({intervals_to_use[i]: s[i] for i in range(6)})

    return allocations


@st.cache_data
def build_audit_dataframe(year_data_tuple, grid_id, alloc_str, acres):
    """Cache the audit dataframe construction to avoid recomputation."""
    audit_rows = []
    for yd in year_data_tuple:
        audit_rows.append({
            'Grid': grid_id,
            'Year': yd[0],
            'Allocation': alloc_str,
            'Acres': acres,
            'Total Indemnity': f"${yd[1]:,.0f}",
            'Producer Premium': f"${yd[2]:,.0f}",
            'Net Return': f"${yd[3]:,.0f}",
            'Total ROI': f"{yd[4]:.2%}"
        })
    return pd.DataFrame(audit_rows)


@st.fragment
def render_audit_section(grid_id, strategy_num, detail_key, yearly_details, alloc_str, params):
    """Render the audit section as a fragment to prevent full page reruns."""
    if detail_key not in yearly_details:
        return

    checkbox_key = f"audit_check_{grid_id}_{strategy_num}"

    show_audit = st.checkbox(
        "Show Year-by-Year Audit",
        key=checkbox_key
    )

    if show_audit:
        year_data = yearly_details[detail_key]

        # Convert year_data to tuple for caching (lists aren't hashable)
        year_data_tuple = tuple(
            (yd['Year'], yd['Total Indemnity'], yd['Producer Premium'], yd['Net Return'], yd['ROI'])
            for yd in year_data
        )

        audit_df = build_audit_dataframe(
            year_data_tuple,
            grid_id,
            alloc_str,
            params['total_insured_acres']
        )

        st.dataframe(audit_df, use_container_width=True, hide_index=True)

        # Download button for this strategy's audit
        audit_csv = audit_df.to_csv(index=False)
        st.download_button(
            "Download Audit CSV",
            audit_csv,
            f"audit_{grid_id}_strategy{strategy_num}.csv",
            "text/csv",
            key=f"audit_dl_{grid_id}_{strategy_num}"
        )


# =============================================================================
# === 3. OPTIMIZATION ENGINE ===
# =============================================================================

def run_optimization(
    _session, grid_id, start_year, end_year, plan_code, prod_factor,
    acres, use, coverage_levels, objective, min_intervals, max_intervals, 
    num_iterations
):
    """
    Run optimization for a single grid using Monte Carlo sampling.
    Returns: (results_df, yearly_details_dict, rate_year, strategy_count)
    """
    # 1. Load data
    try:
        county_base_value = load_county_base_value(_session, grid_id)
    except Exception as e:
        st.error(f"Error loading county base value for {grid_id}: {e}")
        return pd.DataFrame(), {}, 0, 0
    
    all_indices_df = load_all_indices(_session, grid_id)
    all_indices_df = all_indices_df[
        (all_indices_df['YEAR'] >= start_year) & 
        (all_indices_df['YEAR'] <= end_year)
    ]
    
    if all_indices_df.empty:
        st.warning(f"No index data for grid {grid_id} in year range {start_year}-{end_year}")
        return pd.DataFrame(), {}, 0, 0
    
    current_rate_year = get_current_rate_year(_session)
    all_premiums = load_premium_rates(_session, grid_id, use, coverage_levels, current_rate_year)
    all_subsidies = load_subsidies(_session, plan_code, coverage_levels)
    
    # 2. Generate candidates via Monte Carlo sampling
    candidates = []
    seen = set()
    attempts = 0
    max_attempts = num_iterations * 10  # Prevent infinite loops
    
    while len(candidates) < num_iterations and attempts < max_attempts:
        attempts += 1
        
        # Random number of intervals
        num_intervals = random.randint(min_intervals, max_intervals)
        
        # Random selection of intervals
        combo_list = random.sample(INTERVAL_ORDER_11, num_intervals)
        
        # Skip if has adjacent intervals (excluding Nov-Dec/Jan-Feb wrap)
        if has_adjacent_intervals_in_list(combo_list):
            continue
        
        # Generate allocation patterns for this combination
        combo_allocations = generate_allocations(combo_list, num_intervals)
        
        for alloc in combo_allocations:
            key = tuple(sorted((k, round(v, 3)) for k, v in alloc.items() if v > 0))
            if key not in seen:
                seen.add(key)
                candidates.append(alloc)
    
    if len(candidates) == 0:
        st.warning(f"No valid candidates generated for grid {grid_id}")
        return pd.DataFrame(), {}, current_rate_year, 0
    
    # 3. Prepare data for vectorized evaluation
    # Pivot index data: rows = years, columns = intervals
    pivot_df = all_indices_df.pivot_table(
        index='YEAR', 
        columns='INTERVAL_NAME', 
        values='INDEX_VALUE',
        aggfunc='first'
    ).reindex(columns=INTERVAL_ORDER_11).fillna(100)  # Missing = no shortfall
    
    years = pivot_df.index.values
    if len(years) == 0:
        st.warning(f"No valid years for grid {grid_id}")
        return pd.DataFrame(), {}, current_rate_year, len(candidates)
    
    index_matrix = pivot_df.values  # Shape: (num_years, 11)
    
    # 4. Evaluate all candidates (vectorized)
    results = []
    yearly_details = {}
    
    for coverage_level in coverage_levels:
        subsidy = all_subsidies[coverage_level]
        premiums = all_premiums[coverage_level]
        dollar_protection = county_base_value * coverage_level * prod_factor
        total_protection = dollar_protection * acres
        trigger = coverage_level * 100
        
        # Build premium rate array for all intervals
        premium_rates = np.array([premiums.get(interval, 0) for interval in INTERVAL_ORDER_11])
        
        for allocation in candidates:
            # Validate allocation sums to ~100%
            alloc_sum = sum(allocation.values())
            if abs(alloc_sum - 1.0) > 0.02:
                continue
            
            # Build allocation array
            alloc_array = np.array([allocation.get(interval, 0) for interval in INTERVAL_ORDER_11])
            
            # Vectorized calculations across all years
            # interval_protection: shape (11,)
            interval_protection = total_protection * alloc_array
            
            # total_premium per interval: shape (11,)
            total_premium_per_interval = interval_protection * premium_rates
            producer_premium_per_interval = total_premium_per_interval * (1 - subsidy)
            
            # Producer premium per year (sum across intervals): shape (num_years,)
            yearly_producer_premium = np.sum(producer_premium_per_interval)  # Same each year
            yearly_producer_premium_arr = np.full(len(years), yearly_producer_premium)
            
            # Shortfall calculation: shape (num_years, 11)
            # Where index < trigger, shortfall = (trigger - index) / trigger
            shortfall_matrix = np.maximum(0, (trigger - index_matrix) / trigger)
            
            # Indemnity per interval per year: shape (num_years, 11)
            indemnity_matrix = shortfall_matrix * interval_protection
            
            # Total indemnity per year: shape (num_years,)
            yearly_indemnity = np.nansum(indemnity_matrix, axis=1)
            
            # ROI per year
            yearly_roi = np.where(
                yearly_producer_premium > 0,
                (yearly_indemnity - yearly_producer_premium) / yearly_producer_premium,
                0
            )
            
            # Aggregate metrics
            total_indemnity_all = np.nansum(yearly_indemnity)
            total_premium_all = yearly_producer_premium * len(years)
            
            if total_premium_all > 0:
                cumulative_roi = (total_indemnity_all - total_premium_all) / total_premium_all
            else:
                cumulative_roi = 0
            
            std_dev = np.nanstd(yearly_roi)
            risk_adj_ret = cumulative_roi / std_dev if std_dev > 0 else 0
            
            result_entry = {
                'coverage_level': coverage_level,
                'allocation': allocation,
                'average_roi': np.nanmean(yearly_roi),
                'median_roi': np.nanmedian(yearly_roi),
                'cumulative_roi': cumulative_roi,
                'profitable_pct': np.sum(yearly_roi > 0) / len(yearly_roi),
                'std_dev': std_dev,
                'min_roi': np.nanmin(yearly_roi),
                'max_roi': np.nanmax(yearly_roi),
                'risk_adj_ret': risk_adj_ret,
                'total_indemnity': total_indemnity_all,
                'total_premium': total_premium_all
            }
            results.append(result_entry)
            
            # Store yearly details (only if needed later)
            alloc_key = tuple(sorted((k, round(v, 3)) for k, v in allocation.items() if v > 0))
            year_details = [
                {
                    'Year': int(years[i]),
                    'Total Indemnity': yearly_indemnity[i],
                    'Producer Premium': yearly_producer_premium,
                    'Net Return': yearly_indemnity[i] - yearly_producer_premium,
                    'ROI': yearly_roi[i]
                }
                for i in range(len(years))
            ]
            yearly_details[(coverage_level, alloc_key)] = year_details
    
    if len(results) == 0:
        st.warning(f"No valid results for grid {grid_id}")
        return pd.DataFrame(), {}, current_rate_year, len(candidates)
    
    results_df = pd.DataFrame(results).sort_values(objective, ascending=False)
    
    return results_df, yearly_details, current_rate_year, len(candidates)

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
        st.sidebar.success(f"Loaded {len(valid_grids)} grids")
    except Exception as e:
        st.error(f"Could not load Grid ID list: {e}")
        st.stop()
    
    # === SIDEBAR: Common Parameters ===
    st.sidebar.header("Common Parameters")
    
    # === PRESETS ===
    PRESETS = {
        'None': {
            'grids': [],
            'productivity_factor': 100
        },
        'King Ranch Non Actives': {
            'grids': [
                '9131 (Kleberg - TX)',
                '8828 (Kleberg - TX)',
                '8830 (Kleberg - TX)',
                '8831 (Kleberg - TX)',
                '8528 (Brooks - TX)',
                '8529 (Brooks - TX)',
                '8829 (Kenedy - TX)',
                '7930 (Kenedy - TX)',
                '8231 (Kenedy - TX)',
                '7931 (Kenedy - TX)',
            ],
            'productivity_factor': 135
        }
    }
    
    preset_choice = st.sidebar.selectbox(
        "Preset",
        options=list(PRESETS.keys()),
        key="preset_choice"
    )
    
    # Get preset values
    preset = PRESETS[preset_choice]
    preset_prod = preset['productivity_factor']
    preset_grids = preset['grids']
    
    st.sidebar.divider()
    
    prod_options = list(range(60, 151))
    prod_options_formatted = [f"{x}%" for x in prod_options]
    prod_default_idx = preset_prod - 60  # Convert to index (60% = index 0)
    selected_prod_str = st.sidebar.selectbox(
        "Productivity Factor", 
        options=prod_options_formatted, 
        index=prod_default_idx,
        key="prod_factor"
    )
    productivity_factor = int(selected_prod_str.replace('%', '')) / 100.0
    
    total_insured_acres = st.sidebar.number_input(
        "Total Insured Acres", 
        value=1000, 
        step=100,
        key="acres"
    )
    
    intended_use = st.sidebar.selectbox(
        "Intended Use", 
        ['Grazing', 'Haying'],
        key="use"
    )
    
    plan_code = st.sidebar.number_input(
        "Insurance Plan Code", 
        value=13, 
        disabled=True
    )
    
    st.sidebar.divider()
    st.sidebar.caption("*2025 Rates used")
    
    # === MAIN CONTENT ===
    st.subheader("Grid Selection")
    
    # Filter preset grids to only those that exist in valid_grids
    default_grids = [g for g in preset_grids if g in valid_grids] if preset_grids else []
    if not default_grids and valid_grids:
        default_grids = [valid_grids[0]]
    
    selected_grids = st.multiselect(
        "Select Grids to Optimize",
        options=valid_grids,
        default=default_grids,
        max_selections=10
    )
    
    if not selected_grids:
        st.warning("Select at least one grid to optimize.")
        st.stop()
    
    st.info(f"Selected {len(selected_grids)} grid(s)")
    
    st.divider()
    
    # === OPTIMIZATION PARAMETERS ===
    st.subheader("Optimization Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_year = st.selectbox("Start Year", list(range(1948, 2025)), index=50)
        
        coverage_levels = st.multiselect(
            "Coverage Levels",
            [0.70, 0.75, 0.80, 0.85, 0.90],
            default=[0.75, 0.80, 0.85],
            format_func=lambda x: f"{x:.0%}"
        )
    
    with col2:
        end_year = st.selectbox("End Year", list(range(1948, 2025)), index=76)
        
        objective = st.selectbox(
            "Optimize For",
            ['cumulative_roi', 'risk_adj_ret', 'profitable_pct', 'median_roi'],
            format_func=lambda x: {
                'cumulative_roi': 'Cumulative ROI',
                'risk_adj_ret': 'Risk-Adjusted Return',
                'profitable_pct': 'Win Rate',
                'median_roi': 'Median ROI'
            }.get(x, x)
        )
    
    st.divider()
    
    # Interval constraints
    st.subheader("Interval Constraints")
    
    col1, col2 = st.columns(2)
    min_intervals = col1.number_input("Min Intervals", min_value=2, max_value=6, value=4)
    max_intervals = col2.number_input("Max Intervals", min_value=2, max_value=6, value=6)
    
    # Monte Carlo iterations
    iteration_options = {'Fast (500)': 500, 'Standard (1000)': 1000, 'Thorough (2500)': 2500, 'Maximum (5000)': 5000}
    mc_selection = st.select_slider(
        "Monte Carlo Iterations",
        options=list(iteration_options.keys()),
        value='Standard (1000)'
    )
    num_iterations = iteration_options[mc_selection]
    
    st.divider()
    
    # === RUN OPTIMIZATION ===
    if st.button("Run Optimization", type="primary", key="run_opt"):
        
        if not coverage_levels:
            st.error("Select at least one coverage level")
            st.stop()
        
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
                    num_iterations
                )
                
                st.write(f"**Grid {grid_id}:** {strategy_count} candidates tested, {len(results_df)} valid results")
                
                if not results_df.empty:
                    all_grid_results[grid_id] = {
                        'results_df': results_df,
                        'yearly_details': yearly_details,
                        'rate_year': rate_year,
                        'strategy_count': strategy_count
                    }
                
            except Exception as e:
                st.error(f"Error optimizing Grid {grid_id}: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        progress_bar.progress(1.0, text="Optimization complete!")
        
        st.session_state.optimization_results = all_grid_results
        st.session_state.optimization_params = {
            'start_year': start_year,
            'end_year': end_year,
            'objective': objective,
            'productivity_factor': productivity_factor,
            'total_insured_acres': total_insured_acres,
            'intended_use': intended_use
        }
    
    # === DISPLAY RESULTS ===
    if 'optimization_results' in st.session_state and st.session_state.optimization_results:
        st.divider()
        st.header("Optimization Results")
        
        results = st.session_state.optimization_results
        params = st.session_state.optimization_params
        
        st.caption(f"Period: {params['start_year']}-{params['end_year']} | "
                   f"Productivity: {params['productivity_factor']:.0%} | "
                   f"Use: {params['intended_use']} | "
                   f"Acres: {params['total_insured_acres']:,}")
        
        # Summary table
        summary_data = []
        for grid_id, data in results.items():
            if 'results_df' in data and not data['results_df'].empty:
                best = data['results_df'].iloc[0]
                
                alloc_str = ", ".join([
                    f"{k}: {v*100:.0f}%" 
                    for k, v in sorted(best['allocation'].items(), key=lambda x: x[1], reverse=True) 
                    if v > 0
                ])
                
                summary_data.append({
                    'Grid': grid_id,
                    'Coverage': f"{best['coverage_level']:.0%}",
                    'Allocation': alloc_str,
                    'Cumulative ROI': f"{best['cumulative_roi']:.1%}",
                    'Risk-Adj Return': f"{best['risk_adj_ret']:.2f}",
                    'Win Rate': f"{best['profitable_pct']:.0%}",
                    'Strategies Tested': data['strategy_count']
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            csv = summary_df.to_csv(index=False)
            st.download_button(
                "Download Results CSV",
                csv,
                f"optimization_results_{params['start_year']}-{params['end_year']}.csv",
                "text/csv"
            )
        else:
            st.warning("No valid optimization results found.")
        
        # Detailed view per grid
        st.divider()
        st.subheader("Detailed Results by Grid")
        
        for grid_id, data in results.items():
            if 'results_df' not in data or data['results_df'].empty:
                continue

            # Track expander state in session_state to persist across reruns
            expander_key = f"expander_{grid_id}"
            if expander_key not in st.session_state.expander_states:
                st.session_state.expander_states[expander_key] = False

            with st.expander(
                f"Grid {grid_id} - Top 5 Strategies",
                expanded=st.session_state.expander_states.get(expander_key, False)
            ):
                # Update expander state when opened
                st.session_state.expander_states[expander_key] = True

                top5 = data['results_df'].head(5)
                yearly_details = data.get('yearly_details', {})

                for idx, row in top5.iterrows():
                    strategy_num = top5.index.get_loc(idx) + 1
                    st.markdown(f"**Strategy {strategy_num}** - Coverage: {row['coverage_level']:.0%}")

                    alloc_str = ", ".join([
                        f"{k}: {v*100:.0f}%"
                        for k, v in sorted(row['allocation'].items(), key=lambda x: x[1], reverse=True)
                        if v > 0
                    ])
                    st.caption(f"Allocation: {alloc_str}")

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Cumulative ROI", f"{row['cumulative_roi']:.1%}")
                    col2.metric("Risk-Adj Return", f"{row['risk_adj_ret']:.2f}")
                    col3.metric("Win Rate", f"{row['profitable_pct']:.0%}")
                    col4.metric("Median ROI", f"{row['median_roi']:.1%}")

                    # === AUDIT TABLE (using fragment to prevent full page rerun) ===
                    alloc_key = tuple(sorted((k, round(v, 3)) for k, v in row['allocation'].items() if v > 0))
                    detail_key = (row['coverage_level'], alloc_key)

                    # Render audit section as a fragment (isolated rerun)
                    render_audit_section(
                        grid_id,
                        strategy_num,
                        detail_key,
                        yearly_details,
                        alloc_str,
                        params
                    )

                    st.divider()

if __name__ == "__main__":
    main()