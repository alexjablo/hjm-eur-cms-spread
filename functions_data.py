import matplotlib.pyplot as plt
def set_academic_style():
    """
    Configures Matplotlib global parameters to produce publication-quality,
    LaTeX-compatible figures for a finance/mathematics thesis.
    """
    # 1. Reset to clean baseline
    plt.style.use('seaborn-v0_8-whitegrid') 
    
    # 2. Typography & Rendering
    # If you have a working local LaTeX installation, uncomment the line below 
    # for genuine Computer Modern fonts:
    # plt.rcParams['text.usetex'] = True
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif']
    
    # 3. Explicit Font Sizes (Hierarchical)
    plt.rcParams['font.size'] = 10          # Base text size
    plt.rcParams['axes.titlesize'] = 12     # Chart title
    plt.rcParams['axes.labelsize'] = 10     # X and Y axis labels
    plt.rcParams['xtick.labelsize'] = 9     # Axis tick marks
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9     # Legend text
    
    # 4. Grid and Border Aesthetics
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.color'] = '#888888'
    
    # Remove top and right borders (spines)
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True
    
    # 5. Line and Marker Weights
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 4
    
    # 6. Academic Color Palette (Muted, professional corporate tones)
    # Slate Blue, Muted Crimson, Sage Green, Dark Orange, Dark Grey
    academic_colors = ['#1a365d', '#9b2c2c', '#2f855a', '#c05621', '#4a5568']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=academic_colors)
    
    # 7. Layout Optimization
    plt.rcParams['figure.autolayout'] = True

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
def calculate_maturity(spot, tenor_str):

    tenor_str = str(tenor_str).upper().strip()
    
    # Parse Bloomberg conventions
    if 'WK' in tenor_str:
        weeks = int(tenor_str.replace('WK', ''))
        mat = spot + timedelta(weeks=weeks)
    elif 'MO' in tenor_str:
        months = int(tenor_str.replace('MO', ''))
        mat = spot + relativedelta(months=months)
    elif 'YR' in tenor_str:
        years = int(tenor_str.replace('YR', ''))
        mat = spot + relativedelta(years=years)
    else:
        return spot # Fallback if format is unrecognized
    
    # Weekend Adjustment (Move to Monday)
    if mat.weekday() == 5:    # Saturday
        mat += timedelta(days=2)
    elif mat.weekday() == 6:  # Sunday
        mat += timedelta(days=1)
        
    return mat

import numpy as np
import pandas as pd
def bootstrap_ois_curve_act360(df_input, spot_date):
    """
    Bootstraps a discount curve using exact ACT/360 calendar logic.
    Expects df_input to contain: 'Maturity_Date', 'T', and 'Mid'.
    """
    bootstrapped_T = []
    bootstrapped_DF = []
    
    for index, row in df_input.iterrows():
        T = row['T']
        R = row['Mid'] / 100.0
        mat_date = row['Maturity_Date']
        
        # 1. Short End (<= 1 Year)
        if T <= 1.02:   # Using 1.02 to safely capture the 1Y tenor
            # Single payment at maturity
            D = 1.0 / (1.0 + R * T) 
            
        # 2. Long End (> 1 Year)
        else:
            coupon_times = []
            deltas = []
            
            prev_date = spot_date
            current_k = 1
            
            # Generate exact intermediate annual cashflow dates
            while True:
                # Step forward exactly 1 calendar year at a time
                next_coupon_date = spot_date + relativedelta(years=current_k)
                
                # Weekend Adjustment
                if next_coupon_date.weekday() == 5:    # Saturday
                    next_coupon_date += timedelta(days=2)
                elif next_coupon_date.weekday() == 6:  # Sunday
                    next_coupon_date += timedelta(days=1)
                    
                # Stop generating intermediate coupons if we hit or pass maturity
                if next_coupon_date >= mat_date:
                    break
                    
                # Calculate exact ACT/360 model times and period deltas
                T_k = (next_coupon_date - spot_date).days / 360.0
                delta_k = (next_coupon_date - prev_date).days / 360.0
                
                coupon_times.append(T_k)
                deltas.append(delta_k)
                
                prev_date = next_coupon_date
                current_k += 1
            
            # Calculate the final period delta up to the maturity date
            delta_final = (mat_date - prev_date).days / 360.0
            
            # If there are intermediate coupons, interpolate their discount factors
            if len(coupon_times) > 0:
                coupon_times = np.array(coupon_times)
                deltas = np.array(deltas)
                
                # Prepare known arrays for interpolation (pre-pending T=0, DF=1)
                known_T = np.insert(np.array(bootstrapped_T), 0, 0.0)
                known_DF = np.insert(np.array(bootstrapped_DF), 0, 1.0)
                
                # Log-linear interpolation for intermediate discount factors
                interp_logD = np.interp(coupon_times, known_T, np.log(known_DF))
                D_coupons = np.exp(interp_logD)
                
                # Present Value of known intermediate coupons
                annuity_known = np.sum(R * deltas * D_coupons)
            else:
                annuity_known = 0.0
                
            # Solve for final Discount Factor: D(0, T)
            D = (1.0 - annuity_known) / (1.0 + R * delta_final)
            
        bootstrapped_T.append(T)
        bootstrapped_DF.append(D)
        
    # Compile final results into a DataFrame
    results = pd.DataFrame({
        'T': bootstrapped_T,
        'Discount_Factor': bootstrapped_DF
    })
    
    # Calculate continuously compounded Zero Rate (ACT/365) for conventional viewing
    results['Zero_Rate'] = (-np.log(results['Discount_Factor']) / results['T'])
    
    return results

from scipy.interpolate import PchipInterpolator
def forward_curve(df_bootstrapped, max_T=50.0, num_points=10000):
    """
    Takes the bootstrapped pillar nodes and generates a highly granular 
    curve of discount factors and instantaneous forward rates using PCHIP.
    """
    # 1. Extract nodes and prepend T=0, DF=1
    T_nodes = np.concatenate([[0.0], df_bootstrapped['T'].values])
    DF_nodes = np.concatenate([[1.0], df_bootstrapped['Discount_Factor'].values])
    
    # 2. Build the PCHIP interpolator on log(DF)
    pchip = PchipInterpolator(T_nodes, np.log(DF_nodes))
    
    # 3. Create the fine grid
    T_fine = np.linspace(0.001, max_T, num_points)
    
    # 4. Calculate continuous discount factors and forwards
    logD_fine = pchip(T_fine)
    D_fine = np.exp(logD_fine)
    
    pchip_deriv = pchip.derivative()
    f_fine = -pchip_deriv(T_fine)
    
    # 5. Return as a clean DataFrame
    return pd.DataFrame({
        'T': T_fine,
        'Discount_Factor': D_fine,
        'Forward_Rate': f_fine
    })

def plot_ois_curves(pillar_df, fine_curve_df):
    """
    Plots the Discount, Zero Rate, and Forward Rate curves side-by-side.
    Calculates zero rates on the fly to ensure consistency.
    """
    # 1. Safely calculate continuous Zero Rates (%) for both dataframes
    # We use np.maximum to avoid division by zero at T=0
    fine_curve_df['Zero_Rate_%'] = -np.log(fine_curve_df['Discount_Factor']) / np.maximum(fine_curve_df['T'], 1e-6) * 100
    fine_curve_df['Forward_Rate_%'] = fine_curve_df['Forward_Rate'] * 100
    pillar_df['Zero_Rate_%'] = -np.log(pillar_df['Discount_Factor']) / np.maximum(pillar_df['T'], 1e-6) * 100

    # 2. Setup the figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=500)
    fig.suptitle("€STR OIS Curve — 15 January 2026", fontsize=14, fontweight='bold')

    # 3. Define the configuration for each subplot
    plot_configs = [
        {
            "ax": axes[0], "y_fine": "Discount_Factor", "y_pillar": "Discount_Factor",
            "color": "#185FA5", "title": "Discount Curve", "ylabel": "Discount factor D(0,T)",
            "plot_scatter": True
        },
        {
            "ax": axes[1], "y_fine": "Zero_Rate_%", "y_pillar": "Zero_Rate_%",
            "color": "#0F6E56", "title": "Zero Rate Curve", "ylabel": "Zero rate (%)",
            "plot_scatter": True
        },
        {
            "ax": axes[2], "y_fine": "Forward_Rate_%", "y_pillar": None,
            "color": "#993C1D", "title": "Forward Curve f(0,T)", "ylabel": "Instantaneous forward rate f(0,T) (%)",
            "plot_scatter": False, "ylim": (1, 4)
        }
    ]

    # 4. Loop through configurations to build the plots dynamically
    for config in plot_configs:
        ax = config["ax"]
        
        # Plot the smooth interpolated curve
        ax.plot(fine_curve_df['T'], fine_curve_df[config['y_fine']], color=config["color"], lw=1.8)
        
        # Plot the bootstrap pillar nodes if applicable (Forward rates usually don't have exact pillar matches)
        if config["plot_scatter"]:
            ax.scatter(pillar_df['T'], pillar_df[config['y_pillar']], color=config["color"], s=25, zorder=5)
        
        # Apply standard formatting
        ax.set_title(config["title"])
        ax.set_xlabel("Maturity (years)")
        ax.set_ylabel(config["ylabel"])
        ax.grid(True, alpha=0.3)
        
        # Apply custom Y-limits if specified
        if "ylim" in config:
            ax.set_ylim(config["ylim"])

    # 5. Finalize and show
    plt.tight_layout()
    plt.show()

def generate_forward_rates_annuity_grid_act360(pillar_df, spot_date, expiries, tenors):
    """
    Computes Forward Swap Rates and Annuities using ACT/360 exact day counts.
    """
    # 1. Curve Reconstruction Setup
    bootstrapped_T = pillar_df['T'].values
    bootstrapped_DF = pillar_df['Discount_Factor'].values
    
    T_nodes = np.concatenate([[0.0], bootstrapped_T])
    DF_nodes = np.concatenate([[1.0], bootstrapped_DF])
    local_pchip = PchipInterpolator(T_nodes, np.log(DF_nodes))
    
    def discount(T):
        return np.exp(local_pchip(T))

    # 2. Define standard Bloomberg string inputs
    grid_expiries = expiries
    grid_tenors = tenors

    rates_matrix = pd.DataFrame(index=grid_expiries, columns=grid_tenors)
    annuity_matrix = pd.DataFrame(index=grid_expiries, columns=grid_tenors)
    detailed_results = {}

    # 3. Main Calculation Loop
    for exp_lbl in grid_expiries:
        
        # Calculate exact option expiry date and ACT/360 fraction
        T_alpha_date = calculate_maturity(spot_date, exp_lbl)
        T_alpha = (T_alpha_date - spot_date).days / 360.0
        
        for ten_lbl in grid_tenors:
            years = int(ten_lbl.replace('YR', ''))
            
            T_coupons = []
            tau_coupons = [] # Exact delta terms for the annuity
            
            prev_date = T_alpha_date
            
            # Generate annual swap cashflows iteratively
            for k in range(1, years + 1):
                # Dynamically generate the tenor string (e.g., '1YR', '2YR') to use our robust function
                curr_date = calculate_maturity(T_alpha_date, f"{k}YR")
                
                # Model time for interpolation (from spot date)
                T_k = (curr_date - spot_date).days / 360.0
                T_coupons.append(T_k)
                
                # Day count fraction for this specific coupon (ACT/360)
                tau_k = (curr_date - prev_date).days / 360.0
                tau_coupons.append(tau_k)
                
                prev_date = curr_date
                
            T_beta = T_coupons[-1]
            
            # Vectorize lists for math operations
            T_coupons = np.array(T_coupons)
            tau_coupons = np.array(tau_coupons)
            
            # Evaluate discount factors on exact model times
            D_coupons = discount(T_coupons)
            D_alpha = discount(T_alpha)
            D_beta = discount(T_beta)
            
            # Calculate Annuity and Forward Rate
            A = np.sum(tau_coupons * D_coupons)
            S = (D_alpha - D_beta) / A
            
            # Populate matrices
            rates_matrix.loc[exp_lbl, ten_lbl] = np.round(S, 6)
            annuity_matrix.loc[exp_lbl, ten_lbl] = np.round(A, 6)
            
            # Store exact times and weights for HJM calibration
            detailed_results[(exp_lbl, ten_lbl)] = {
                'T_alpha': np.round(T_alpha, 6),
                'T_beta': np.round(T_beta, 6),
                'T_coupons': T_coupons,                  # Array of model times for cashflows
                'tau_coupons': np.round(tau_coupons, 6), # Exact ACT/360 year fractions
                'D_coupons': np.round(D_coupons, 6),
                'D_alpha': np.round(D_alpha, 6),
                'D_beta': np.round(D_beta, 6),
                'S': np.round(S, 6),
                'A': np.round(A, 6)
            }

    return rates_matrix.astype(float), annuity_matrix.astype(float), detailed_results

def compute_model_implied_volatility(kappa, sigma, 
                                     T_alpha, T_beta, T_coupons, tau_coupons, 
                                     D_alpha, D_beta, D_coupons, S, A):
    """
    Computes the model-implied Bachelier normal volatility for a single swaption
    under a 3-factor Markovian HJM model.
    
    Parameters:
    -----------
    kappa       : np.array of shape (3,) - Mean reversion parameters
    sigma       : np.array of shape (3,) - Volatility parameters
    T_alpha     : float - Option expiry in exact model time
    T_beta      : float - Underlying swap maturity in exact model time
    T_coupons   : np.array of shape (N,) - Continuous model times of the underlying cashflows
    tau_coupons : np.array of shape (N,) - Exact ACT/360 day count fractions for each cashflow
    D_alpha     : float - Discount factor at expiry D(0, T_alpha)
    D_beta      : float - Discount factor at maturity D(0, T_beta)
    D_coupons   : np.array of shape (N,) - Discount factors at each coupon date
    S           : float - Forward swap rate S(0; T_alpha, T_beta)
    A           : float - Forward annuity A(0; T_alpha, T_beta)
    
    Returns:
    --------
    Sigma_model : float - Model-implied normal volatility
    """
    
    # Ensure inputs are numpy arrays for vectorization
    kappa = np.asarray(kappa)
    sigma = np.asarray(sigma)
    T_coupons = np.asarray(T_coupons)
    tau_coupons = np.asarray(tau_coupons)
    
    # 1. Compute Bond Loadings B_i(0, T) evaluated at t=0
    # Shape of B_alpha and B_beta will be (3,)
    B_alpha = -(1.0 - np.exp(-kappa * T_alpha)) / kappa
    B_beta  = -(1.0 - np.exp(-kappa * T_beta)) / kappa
    
    # Shape of B_coupons will be (N coupons, 3 factors)
    B_coupons = -(1.0 - np.exp(-kappa * T_coupons[:, None])) / kappa 
    
    # 2. Compute the summation term: sum_j [tau_j * B_i(0, T_j) * D(0, T_j)]
    # We broadcast tau_coupons and D_coupons to (N, 1) so they multiply correctly across the 3 factors
    sum_term = np.sum(tau_coupons[:, None] * D_coupons[:, None] * B_coupons, axis=0)
    
    # 3. Compute the drift-frozen factor loading Lambda_tilde_i 
    # Shape will be (3,) -> one loading per factor
    numerator = (B_alpha * D_alpha) - (B_beta * D_beta) - (S * sum_term)
    Lambda_tilde = sigma * (numerator / A)
    
    # 4. Compute Model Implied Volatility 
    # Sigma_model = sqrt( sum( Lambda_tilde_i^2 ) )
    Sigma_model = np.sqrt(np.sum(Lambda_tilde**2))
    
    return Sigma_model

def extract_vega_weights(detailed_results, market_vols):
    """
    Extracts the normalized Bachelier Vega weights for every cell in the calibration grid.
    Returns a DataFrame perfectly formatted for a seaborn heatmap.
    """
    vega_df = pd.DataFrame(index=market_vols.index, columns=market_vols.columns, dtype=float)
    
    for exp_lbl in market_vols.index:
        for ten_lbl in market_vols.columns:
            cell_data = detailed_results[(exp_lbl, ten_lbl)]
            # Bachelier ATM Vega = A * sqrt(T_alpha) / sqrt(2*pi)
            raw_vega = (cell_data['A'] * np.sqrt(cell_data['T_alpha'])) / np.sqrt(2 * np.pi)
            vega_df.loc[exp_lbl, ten_lbl] = raw_vega
            
    # Least_squares minimizes SUM OF SQUARED residuals, so we square Vegas before normalizing
    vega_squared = vega_df ** 2
    
    total_squared_vega = vega_squared.values.sum()
    normalized_vega_weights = vega_squared / total_squared_vega
    
    return normalized_vega_weights

def hjm_objective_function(params, detailed_results, market_vols, use_vega=False):
    """
    Calculates the 1D vector of residuals for the optimizer.
    Dynamically handles n factors based on the length of params.
    """
    n_factors = len(params) // 2
    kappa = params[:n_factors]
    sigma = params[n_factors:]
    
    residuals = []
    
    for exp_lbl in market_vols.index:
        for ten_lbl in market_vols.columns:
            cell_data = detailed_results[(exp_lbl, ten_lbl)]
            market_vol = market_vols.loc[exp_lbl, ten_lbl]
            
            model_vol = compute_model_implied_volatility(
                kappa=kappa, sigma=sigma,
                T_alpha=cell_data['T_alpha'], T_beta=cell_data['T_beta'],
                T_coupons=cell_data['T_coupons'], tau_coupons=cell_data['tau_coupons'],
                D_alpha=cell_data['D_alpha'], D_beta=cell_data['D_beta'],
                D_coupons=cell_data['D_coupons'], S=cell_data['S'], A=cell_data['A']
            )
            
            vol_error = model_vol - market_vol
            
            if use_vega:
                vega = (cell_data['A'] * np.sqrt(cell_data['T_alpha'])) / np.sqrt(2 * np.pi)
                residuals.append(vega * vol_error)
            else:
                residuals.append(vol_error)
            
    return np.array(residuals)

from scipy.optimize import least_squares
from scipy.stats import qmc
def calibration_1_lhs(bounds, detailed_results, market_target_vols, 
                         num_initial_guesses=50, use_vega=False):
    """
    Phase 1: Latin Hypercube spatial search to find the global minimum neighborhood.
    Returns the best guess, its error, and the full history for notebook analysis.
    """
    weight_type = "Vega Weights" if use_vega else "Equal Weights"
    
    lower_bounds = np.array(bounds[0])
    upper_bounds = np.array(bounds[1])
    n_params = len(lower_bounds)
    n_factors = n_params // 2
    
    print(f"--- Starting {n_factors}-Factor Calibration ({weight_type}) ---")
    print(f"Phase 1: Coarse Search ({num_initial_guesses} Latin Hypercube points)")
    
    # 1. Generate uniform sample points
    sampler = qmc.LatinHypercube(d=n_params, seed=42)
    sample_points = sampler.random(n=num_initial_guesses)
    scaled_guesses = qmc.scale(sample_points, lower_bounds, upper_bounds)
    
    best_initial_guess = None
    lowest_initial_error = np.inf
    
    # Track the entire search grid for deeper notebook analysis
    search_history = []
    
    # 2. Evaluate unoptimized error across the grid
    for guess in scaled_guesses:
        residuals = hjm_objective_function(guess, detailed_results, market_target_vols, use_vega)
        current_error = np.sum(residuals**2)
        
        search_history.append({'guess': guess, 'sse': current_error})
        
        if current_error < lowest_initial_error:
            lowest_initial_error = current_error
            best_initial_guess = guess
            
    print(f"Best initial guess found with SSE: {lowest_initial_error:.6e}")
    
    return best_initial_guess, lowest_initial_error, search_history
def calibration_2_trf(best_initial_guess, bounds, detailed_results, 
                             market_target_vols, use_vega=False):
    """
    Phase 2: Trust-Region Reflective (trf) local optimization for exact precision.
    Takes the best guess from Phase 1 and refines it via gradient descent.
    """
    n_params = len(best_initial_guess)
    n_factors = n_params // 2
    
    print("Phase 2: Gradient-Based Precision Refinement")
    
    # 3. Run least_squares using the best guess
    opt_result = least_squares(
        hjm_objective_function, 
        x0=best_initial_guess, 
        bounds=bounds,
        args=(detailed_results, market_target_vols, use_vega),
        method='trf', 
        ftol=1e-6, xtol=1e-6,
        max_nfev=1500
    )
    
    if not opt_result.success:
        raise RuntimeError(f"Calibration failed: {opt_result.message}")
        
    print(f"Refined SSE: {np.sum(opt_result.fun**2):.6e}")
    print("--- Calibration Successful! ---")
    
    # Extract final parameters dynamically
    calibrated_params = opt_result.x
    kappa = calibrated_params[:n_factors]
    sigma = calibrated_params[n_factors:]
    
    # --- PRINT FINAL VALUES ---
    print("\nFinal Calibrated Parameters:")
    for i in range(n_factors):
        print(f"  Factor {i+1} | Kappa: {kappa[i]:.4f} | Sigma: {sigma[i]*10000:.2f} bps")
    print("----------------------------------------\n")
    
    # 4. Generate the mirror object
    model_vols_decimal = pd.DataFrame(index=market_target_vols.index, columns=market_target_vols.columns)
    for exp_lbl in market_target_vols.index:
        for ten_lbl in market_target_vols.columns:
            cell_data = detailed_results[(exp_lbl, ten_lbl)]
            vol = compute_model_implied_volatility(
                kappa=kappa, sigma=sigma,
                T_alpha=cell_data['T_alpha'], T_beta=cell_data['T_beta'],
                T_coupons=cell_data['T_coupons'], tau_coupons=cell_data['tau_coupons'],
                D_alpha=cell_data['D_alpha'], D_beta=cell_data['D_beta'],
                D_coupons=cell_data['D_coupons'], S=cell_data['S'], A=cell_data['A']
            )
            model_vols_decimal.loc[exp_lbl, ten_lbl] = vol
            
    return opt_result, calibrated_params, model_vols_decimal.astype(float)

def plot_calibration_convergence(lhs_history, opt_result):
    """
    Visualizes the convergence from the Phase 1 LHS coarse search 
    to the Phase 2 TRF precision refinement.
    """
    # Extract data
    initial_sses = np.array([item['sse'] for item in lhs_history])
    best_initial_sse = np.min(initial_sses)
    refined_sse = np.sum(opt_result.fun**2)
    
    # Setup plot using your academic style settings
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=500)
    fig.suptitle("Calibration Convergence: Phase 1 (LHS) vs Phase 2 (TRF)", fontsize=14, fontweight='bold')
    
    # Corporate academic colors from your palette
    color_lhs = '#1a365d' # Slate blue
    color_trf = '#9b2c2c' # Muted crimson
    
    # ==========================================================
    # LEFT PANEL: Phase 1 Search Landscape
    # ==========================================================
    axes[0].scatter(range(len(initial_sses)), initial_sses, color=color_lhs, alpha=0.5, edgecolor='none', label='LHS Sample Points')
    axes[0].axhline(best_initial_sse, color='black', linestyle='--', lw=1.5, label=f'Best LHS: {best_initial_sse:.2e}')
    axes[0].axhline(refined_sse, color=color_trf, linestyle='-', linewidth=2, label=f'Final TRF: {refined_sse:.2e}')
    
    # Use log scale because unoptimized parameters can yield massive SSEs
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Latin Hypercube Sample Index')
    axes[0].set_ylabel('Sum of Squared Errors (Log Scale)')
    axes[0].set_title('Phase 1: Global Search Landscape')
    axes[0].legend(frameon=True, facecolor='white', edgecolor='none')
    
    # ==========================================================
    # RIGHT PANEL: Phase 2 Precision Refinement
    # ==========================================================
    labels = ['Best Phase 1\n(LHS Initial Guess)', 'Final Phase 2\n(TRF Refinement)']
    values = [best_initial_sse, refined_sse]
    
    axes[1].bar(labels, values, color=[color_lhs, color_trf], width=0.4, alpha=0.85)
    axes[1].set_ylabel('Sum of Squared Errors')
    axes[1].set_title('Phase 2: Local Gradient Refinement')
    
    # Format Y-axis to scientific notation for clean readability
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    # Annotate bars with the exact SSE values
    for i, v in enumerate(values):
        axes[1].text(i, v + (max(values)*0.02), f"{v:.2e}", ha='center', va='bottom', fontweight='bold')
    
    # Format borders and grid
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
    plt.tight_layout()
    plt.show()

import seaborn as sns
def plot_calibration_error(market_target_vols, model_vols_equal, 
                                     model_vols_vega, expiry_prefix):
    """
    Computes pricing errors in basis points, prints summary statistics, 
    and generates a full suite of academic plots: MAE/RMSE comparisons, 
    side-by-side heatmaps, and dynamic term structure grids.
    """
    # 1. Convert all matrices to Basis Points (bps)
    market_bps = market_target_vols * 10000.0
    equal_bps  = model_vols_equal.astype(float) * 10000.0
    vega_bps   = model_vols_vega.astype(float) * 10000.0
    
    # 2. Calculate Residuals (Model - Market)
    err_equal = equal_bps - market_bps
    err_vega  = vega_bps - market_bps
    
    # 3. Compute error metrics
    mae_equal = err_equal.abs().mean()
    mae_vega  = err_vega.abs().mean()
    rmse_equal = np.sqrt((err_equal**2).mean())
    rmse_vega  = np.sqrt((err_vega**2).mean())

    total_mae_equal = mae_equal.mean()
    total_mae_vega  = mae_vega.mean()
    total_rmse_equal = rmse_equal.mean()
    total_rmse_vega  = rmse_vega.mean()
    
    error_metrics_summary = pd.DataFrame({
        'Equal MAE': mae_equal,
        'Vega MAE':  mae_vega,
        'Equal RMSE': rmse_equal,
        'Vega RMSE':  rmse_vega
    }).round(2)
    
    # --- PRINT TERMINAL SUMMARY ---
    print("="*60)
    print(f" PERFORMANCE SUMMARY: {expiry_prefix.upper()}")
    print("="*60)
    print(error_metrics_summary)
    print("-" * 60)
    print(f"Global Surface Mean | Equal MAE: {total_mae_equal:.2f} bps | Vega MAE: {total_mae_vega:.2f} bps")
    print(f"Global Surface Mean | Equal RMSE: {total_rmse_equal:.2f} bps | Vega RMSE: {total_rmse_vega:.2f} bps")
    print("="*60 + "\n")
    
    # Academic Colors
    color_eq = '#9b2c2c'  # Muted Crimson
    color_vg = '#1a365d'  # Slate Blue
    color_mkt = 'black'
    
    # ============================================================
    # PLOT 1: BAR CHARTS (Fused from plot_calibration_error)
    # ============================================================
    tenors = err_equal.columns
    x = np.arange(len(tenors))
    width = 0.35
    
    fig_bar, axes_bar = plt.subplots(1, 2, figsize=(14, 6), dpi=500)
    
    # MAE Panel
    axes_bar[0].bar(x - width/2, mae_equal, width, label='Equal $w_{ij}$', color=color_eq, alpha=0.85)
    axes_bar[0].bar(x + width/2, mae_vega, width, label='Vega $w_{ij}$', color=color_vg, alpha=0.85)
    axes_bar[0].axhline(y=total_mae_equal, color=color_eq, linestyle='--', lw=1.5, label=f'Eq Avg. ({total_mae_equal:.1f} bp)')
    axes_bar[0].axhline(y=total_mae_vega, color=color_vg, linestyle=':', lw=2.0, label=f'Vg Avg. ({total_mae_vega:.1f} bp)')
    axes_bar[0].set_ylabel('Mean Absolute Error (bps)')
    axes_bar[0].set_title('MAE by Underlying Swap Tenor', fontweight='bold')
    
    # RMSE Panel
    axes_bar[1].bar(x - width/2, rmse_equal, width, label='Equal $w_{ij}$', color=color_eq, alpha=0.85)
    axes_bar[1].bar(x + width/2, rmse_vega, width, label='Vega $w_{ij}$', color=color_vg, alpha=0.85)
    axes_bar[1].axhline(y=total_rmse_equal, color=color_eq, linestyle='--', lw=1.5, label=f'Eq Avg. ({total_rmse_equal:.1f} bp)')
    axes_bar[1].axhline(y=total_rmse_vega, color=color_vg, linestyle=':', lw=2.0, label=f'Vg Avg. ({total_rmse_vega:.1f} bp)')
    axes_bar[1].set_ylabel('Root Mean Squared Error (bps)')
    axes_bar[1].set_title('RMSE by Underlying Swap Tenor', fontweight='bold')
    
    # Shared formatting for bar charts
    for ax in axes_bar:
        ax.set_xticks(x)
        ax.set_xticklabels(tenors)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.legend(frameon=True, facecolor='white', edgecolor='none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, max(rmse_vega.max(), rmse_equal.max()) * 1.25)
        
    plt.suptitle(f"Aggregate Errors: {expiry_prefix}", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    # plt.savefig(f"{expiry_prefix}_bar_errors.pdf", format="pdf", dpi=500, bbox_inches='tight')
    plt.show()

    # ============================================================
    # PLOT 2: SIDE-BY-SIDE RESIDUAL HEATMAPS
    # ============================================================
    vmin = min(err_equal.min().min(), err_vega.min().min())
    vmax = max(err_equal.max().max(), err_vega.max().max())
    
    fig_heat, axes_heat = plt.subplots(1, 2, figsize=(18, 7), dpi=500)
    fig_heat.tight_layout(rect=[0, 0, 0.90, 1])
    cbar_ax = fig_heat.add_axes([0.92, 0.12, 0.015, 0.72])
    
    sns.heatmap(err_equal, annot=True, fmt=".1f", cmap="RdBu_r", center=0, 
                vmin=vmin, vmax=vmax, ax=axes_heat[0], cbar=False, annot_kws={"size": 8})
    axes_heat[0].set_title("Residuals: Equal Weights (Model - Market in bps)", fontweight='bold')
    axes_heat[0].set_xlabel("Underlying Swap Tenor")
    axes_heat[0].set_ylabel("Option Expiry")
    
    sns.heatmap(err_vega, annot=True, fmt=".1f", cmap="RdBu_r", center=0, 
                vmin=vmin, vmax=vmax, ax=axes_heat[1], cbar_ax=cbar_ax,
                cbar_kws={'label': 'Pricing Error (bps)'}, annot_kws={"size": 8})
    axes_heat[1].set_title("Residuals: Vega Weights (Model - Market in bps)", fontweight='bold')
    axes_heat[1].set_xlabel("Underlying Swap Tenor")
    axes_heat[1].set_ylabel("") 
    
    plt.subplots_adjust(wspace=0.12, right=0.91)
    # plt.savefig(f"{export_prefix}_heatmap.pdf", format="pdf", dpi=500, bbox_inches='tight')
    plt.show()
    
    # ============================================================
    # PLOT 3: DYNAMIC TERM STRUCTURE FITS
    # ============================================================
    n_tenors = len(tenors)
    n_cols = 3
    n_rows = int(np.ceil(n_tenors / n_cols))
    
    fig_ts, axes_ts = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows), sharex=True, dpi=500)
    axes_ts = axes_ts.flatten()
    expiries_x = market_target_vols.index
    
    for i, ten_lbl in enumerate(tenors):
        ax = axes_ts[i]
        
        ax.plot(expiries_x, market_bps[ten_lbl], 'o-', label='Market Target', color=color_mkt, lw=1.5, markersize=4)
        ax.plot(expiries_x, equal_bps[ten_lbl], 's--', label='Model (Equal)', color=color_eq, lw=1.2, markersize=3)
        ax.plot(expiries_x, vega_bps[ten_lbl], '^--', label='Model (Vega)', color=color_vg, lw=1.2, markersize=3)
        
        ax.set_title(f"Term Structure: {ten_lbl} Tenor", fontsize=11, fontweight='bold')
        ax.set_ylabel("Normal Vol (bps)", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if i == 0:
            ax.legend(frameon=True, facecolor='white', edgecolor='none', loc='best')
            
    # Clean up empty subplots
    for j in range(i + 1, len(axes_ts)):
        axes_ts[j].set_visible(False)
        
    for ax in axes_ts[-n_cols:]:
        ax.set_xlabel("Option Expiry", fontsize=11)
        ax.tick_params(axis='x', labelrotation=45)
        
    plt.suptitle(f"Model Fit Robustness: {expiry_prefix}", fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    # plt.savefig(f"{expiry_prefix}_term_structure.pdf", format="pdf", dpi=500, bbox_inches='tight')
    plt.show()
    
    return error_metrics_summary

def run_hjm_monte_carlo(kappa, sigma, T_expiry,
                        T_10Y_cashflows, tau_10Y,
                        T_2Y_cashflows, tau_2Y,
                        P_0_func, f_0_func, strike,
                        num_paths=100000, num_steps=100,
                        discount_method='exact', seed=42):
    """
    Prices a 10Y-2Y CMS Spread Option.
    discount_method: 'exact' (affine formula) or 'trapezoidal' (path integral)
    """
    rng = np.random.default_rng(seed)

    kappa = np.asarray(kappa, dtype=float).reshape(3, 1)
    sigma = np.asarray(sigma, dtype=float).reshape(3, 1)

    dt = T_expiry / num_steps

    def calc_Y(t):
        return (sigma**2 / (2.0 * kappa)) * (1.0 - np.exp(-2.0 * kappa * t))

    def calc_Psi(t):
        return (sigma**2 / (2.0 * kappa**2)) * (1.0 - np.exp(-kappa * t))**2

    exp_k_dt  = np.exp(-kappa * dt)
    Y_dt      = calc_Y(dt)
    std_X_dt  = np.sqrt(Y_dt)

    # State variables
    X = np.zeros((3, num_paths))
    
    # Trapezoidal integration tracking
    integral_r = np.zeros(num_paths)
    if discount_method == 'trapezoidal':
        r_curr = f_0_func(0.0) + 0.0 # X and Psi are 0 at t=0

    # Phase 1: Time-stepping to T_expiry
    for k in range(num_steps):
        t_next = (k + 1) * dt
        Z = rng.standard_normal((3, num_paths))
        X = exp_k_dt * X + std_X_dt * Z
        
        if discount_method == 'trapezoidal':
            Psi_next = calc_Psi(t_next)
            r_next = f_0_func(t_next) + np.sum(X + Psi_next, axis=0)
            integral_r += 0.5 * (r_curr + r_next) * dt
            r_curr = r_next

    # Phase 2: Stochastic Discount Factor at T_expiry
    Y_T   = calc_Y(T_expiry)
    Psi_T = calc_Psi(T_expiry)
    
    if discount_method == 'exact':
        P_0_T = P_0_func(T_expiry)
        A_0T = -(1.0 - np.exp(-kappa * T_expiry)) / kappa
        exponent_discount = np.sum(A_0T * (X + Psi_T) - 0.5 * (A_0T**2) * Y_T, axis=0)
        stochastic_discount = P_0_T * np.exp(exponent_discount)
    elif discount_method == 'trapezoidal':
        stochastic_discount = np.exp(-integral_r)
    else:
        raise ValueError("discount_method must be 'exact' or 'trapezoidal'")

    # Phase 3: Affine bond price reconstruction at T_expiry
    def reconstruct_bond(T_k):
        B = -(1.0 - np.exp(-kappa * (T_k - T_expiry))) / kappa
        exponent = np.sum(B * (X + Psi_T) - 0.5 * (B**2) * Y_T, axis=0)
        return (P_0_func(T_k) / P_0_func(T_expiry)) * np.exp(exponent)

    # 10Y Swap Reconstruction
    P_10Y = np.array([reconstruct_bond(Tk) for Tk in T_10Y_cashflows])
    A_10Y = np.sum(tau_10Y[:, None] * P_10Y, axis=0)
    S_10Y = (1.0 - P_10Y[-1]) / A_10Y

    # 2Y Swap Reconstruction
    P_2Y = np.array([reconstruct_bond(Tk) for Tk in T_2Y_cashflows])
    A_2Y = np.sum(tau_2Y[:, None] * P_2Y, axis=0)
    S_2Y = (1.0 - P_2Y[-1]) / A_2Y

    # Phase 4: Payoff and aggregation
    spreads            = S_10Y - S_2Y
    payoffs            = np.maximum(spreads - strike, 0.0)
    discounted_payoffs = payoffs * stochastic_discount

    mc_price   = float(np.mean(discounted_payoffs))
    mc_std_err = float(np.std(discounted_payoffs) / np.sqrt(num_paths))

    return mc_price, mc_std_err, spreads

def run_hjm_monte_carlo_paths(kappa, sigma, T_expiry,
                              T_10Y_cashflows, tau_10Y,
                              T_2Y_cashflows, tau_2Y,
                              P_0_func, f_0_func, strike,
                              num_paths=100000, num_steps=100,
                              discount_method='trapezoidal', seed=42):
    """
    Prices a 10Y-2Y CMS Spread Option and returns the full time-series 
    paths of the forward spread to analyze dynamic variance expansion.
    """
    rng = np.random.default_rng(seed)

    kappa = np.asarray(kappa, dtype=float).reshape(3, 1)
    sigma = np.asarray(sigma, dtype=float).reshape(3, 1)

    dt = T_expiry / num_steps

    def calc_Y(t):
        return (sigma**2 / (2.0 * kappa)) * (1.0 - np.exp(-2.0 * kappa * t))

    def calc_Psi(t):
        return (sigma**2 / (2.0 * kappa**2)) * (1.0 - np.exp(-kappa * t))**2

    exp_k_dt  = np.exp(-kappa * dt)
    Y_dt      = calc_Y(dt)
    std_X_dt  = np.sqrt(Y_dt)

    # State variables
    X = np.zeros((3, num_paths))
    
    # NEW: Matrix to store the forward spread at each time step
    spread_paths = np.zeros((num_paths, num_steps))
    
    # Trapezoidal integration tracking
    integral_r = np.zeros(num_paths)
    if discount_method == 'trapezoidal':
        r_curr = f_0_func(0.0) + 0.0 

    # Phase 1: Time-stepping to T_expiry & Path Tracking
    for k in range(num_steps):
        t_next = (k + 1) * dt
        Z = rng.standard_normal((3, num_paths))
        X = exp_k_dt * X + std_X_dt * Z
        
        # --- DYNAMIC FORWARD SWAP RECONSTRUCTION ---
        Y_t = calc_Y(t_next)
        Psi_t = calc_Psi(t_next)
        P_0_t_next = P_0_func(t_next)
        
        def reconstruct_bond_t(T_k):
            """Reconstructs bond price P(t_next, T_k) given current state X"""
            B = -(1.0 - np.exp(-kappa * (T_k - t_next))) / kappa
            exponent = np.sum(B * (X + Psi_t) - 0.5 * (B**2) * Y_t, axis=0)
            return (P_0_func(T_k) / P_0_t_next) * np.exp(exponent)

        # Numerator requires the bond maturing at the option expiry
        P_T_exp_t = reconstruct_bond_t(T_expiry)
        
        # 10Y Forward Swap at t_next
        P_10Y_t = np.array([reconstruct_bond_t(Tk) for Tk in T_10Y_cashflows])
        A_10Y_t = np.sum(tau_10Y[:, None] * P_10Y_t, axis=0)
        S_10Y_t = (P_T_exp_t - P_10Y_t[-1]) / A_10Y_t

        # 2Y Forward Swap at t_next
        P_2Y_t = np.array([reconstruct_bond_t(Tk) for Tk in T_2Y_cashflows])
        A_2Y_t = np.sum(tau_2Y[:, None] * P_2Y_t, axis=0)
        S_2Y_t = (P_T_exp_t - P_2Y_t[-1]) / A_2Y_t

        # Record the current cross-sectional forward spread
        spread_paths[:, k] = S_10Y_t - S_2Y_t
        # -------------------------------------------

        if discount_method == 'trapezoidal':
            r_next = f_0_func(t_next) + np.sum(X + Psi_t, axis=0)
            integral_r += 0.5 * (r_curr + r_next) * dt
            r_curr = r_next

    # Phase 2: Stochastic Discount Factor at T_expiry
    if discount_method == 'exact':
        Y_T   = calc_Y(T_expiry)
        Psi_T = calc_Psi(T_expiry)
        P_0_T = P_0_func(T_expiry)
        A_0T = -(1.0 - np.exp(-kappa * T_expiry)) / kappa
        exponent_discount = np.sum(A_0T * (X + Psi_T) - 0.5 * (A_0T**2) * Y_T, axis=0)
        stochastic_discount = P_0_T * np.exp(exponent_discount)
    elif discount_method == 'trapezoidal':
        stochastic_discount = np.exp(-integral_r)
    else:
        raise ValueError("discount_method must be 'exact' or 'trapezoidal'")

    # Phase 3: Payoff and aggregation
    # The final column of spread_paths (where t_next == T_expiry) 
    # perfectly equals the terminal spot spread.
    terminal_spreads   = spread_paths[:, -1]
    payoffs            = np.maximum(terminal_spreads - strike, 0.0)
    discounted_payoffs = payoffs * stochastic_discount

    mc_price   = float(np.mean(discounted_payoffs))
    mc_std_err = float(np.std(discounted_payoffs) / np.sqrt(num_paths))

    return mc_price, mc_std_err, spread_paths

def run_martingale_test(kappa, sigma, T_test, P_0_func, f_0_func,
                        num_paths=100000, num_steps=100, 
                        discount_method='exact', seed=42):
    """
    Validates E[B(t)^{-1} * P(t,T)] = P(0,T).
    """
    rng = np.random.default_rng(seed)
    kappa = np.asarray(kappa, dtype=float).reshape(3, 1)
    sigma = np.asarray(sigma, dtype=float).reshape(3, 1)

    dt = T_test / num_steps
    time_grid = np.linspace(0, T_test, num_steps + 1)

    def calc_Y(t): return (sigma**2 / (2.0*kappa)) * (1.0 - np.exp(-2.0*kappa*t))
    def calc_Psi(t): return (sigma**2 / (2.0*kappa**2)) * (1.0 - np.exp(-kappa*t))**2

    Y_dt = calc_Y(dt)
    std_X_dt = np.sqrt(Y_dt)
    exp_k_dt = np.exp(-kappa * dt)

    target_P0 = P_0_func(T_test)
    X = np.zeros((3, num_paths))
    
    integral_r = np.zeros(num_paths)
    if discount_method == 'trapezoidal':
        r_curr = f_0_func(0.0)

    mc_expected_values = [target_P0]

    for k in range(num_steps):
        t_next = (k + 1) * dt
        Z = rng.standard_normal((3, num_paths))
        X = exp_k_dt * X + std_X_dt * Z

        Y_next   = calc_Y(t_next)
        Psi_next = calc_Psi(t_next)

        if discount_method == 'exact':
            A_0_tnext = -(1.0 - np.exp(-kappa * t_next)) / kappa
            exponent_disc = np.sum(A_0_tnext * (X + Psi_next) - 0.5 * (A_0_tnext**2) * Y_next, axis=0)
            stochastic_discount = P_0_func(t_next) * np.exp(exponent_disc)
        elif discount_method == 'trapezoidal':
            r_next = f_0_func(t_next) + np.sum(X + Psi_next, axis=0)
            integral_r += 0.5 * (r_curr + r_next) * dt
            r_curr = r_next
            stochastic_discount = np.exp(-integral_r)

        B = -(1.0 - np.exp(-kappa * (T_test - t_next))) / kappa
        exponent_bond = np.sum(B * (X + Psi_next) - 0.5 * (B**2) * Y_next, axis=0)
        P_t_T = (P_0_func(T_test) / P_0_func(t_next)) * np.exp(exponent_bond)

        discounted_bond = stochastic_discount * P_t_T
        mc_expected_values.append(float(np.mean(discounted_bond)))

    error_bps = (np.array(mc_expected_values) - target_P0) * 10000.0
    theoretical_noise = theoretical_noise = np.std(discounted_bond) / np.sqrt(num_paths) * 10000.0

    print(f"\n--- Martingale Test Results ({discount_method.upper()} Method) ---")
    print(f"Target P(0, {T_test:.2f}) : {target_P0:.6f}")
    print(f"Max absolute error : {np.max(np.abs(error_bps)):.4f} bps")
    print(f"Mean error         : {np.mean(error_bps[1:]):.4f} bps")
    print(f"Theoretical noise  : ±{theoretical_noise:.4f} bps (1σ)")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi = 500)
    fig.suptitle(f"Martingale Test: $E[B(t)^{{-1}} P(t,{T_test:.1f})]$ vs $P(0,{T_test:.1f})$", fontsize=12)

    axes[0].plot(time_grid, error_bps, color='#185FA5', lw=1.5, label='MC error')
    axes[0].axhline(0, color='black', ls='--', lw=1.2)
    axes[0].axhline(+theoretical_noise, color='red', ls=':', lw=1.0, label=f'+1σ noise ({theoretical_noise:.1f}bp)')
    axes[0].axhline(-theoretical_noise, color='red', ls=':', lw=1.0)
    axes[0].set_xlabel("Simulation time (years)")
    axes[0].set_ylabel("Error (bps)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(error_bps[1:], bins=20, color='#185FA5', alpha=0.7, edgecolor='white')
    axes[1].axvline(0, color='black', ls='--', lw=1.2)
    axes[1].axvline(np.mean(error_bps[1:]), color='red', ls='-', lw=1.5, label=f'Mean = {np.mean(error_bps[1:]):.3f} bp')
    axes[1].set_xlabel("Error (bps)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'martingale_test_{discount_method}.pdf', dpi=500, bbox_inches='tight')
    plt.show()

    return time_grid, error_bps

from scipy.stats import norm
def price_analytical_cms_spread(cell_data_10Y, cell_data_2Y, kappa, sigma, strike, option_type='call'):
    """
    Prices a CMS Spread Option (10Y - 2Y) analytically using the HJM frozen-drift 
    Bachelier approximation, pulling exact market data directly from the calibration matrix.
    """
    kappa = np.asarray(kappa, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # 1. Extract common option expiry parameters (T_alpha)
    T_alpha = cell_data_10Y['T_alpha']
    D_alpha = cell_data_10Y['D_alpha'] # P(0, T_alpha)
    B_alpha = -(1.0 - np.exp(-kappa * T_alpha)) / kappa
    
    # --- HELPER: Compute Loadings for a specific swap leg ---
    def get_loadings(cell_data):
        T_beta = cell_data['T_beta']
        D_beta = cell_data['D_beta']
        S_0 = cell_data['S']
        A_0 = cell_data['A']
        
        T_coupons = cell_data['T_coupons']
        tau_coupons = cell_data['tau_coupons']
        D_coupons = cell_data['D_coupons']
        
        # B(0, T_beta) and B(0, T_coupons)
        B_beta = -(1.0 - np.exp(-kappa * T_beta)) / kappa
        B_coupons = -(1.0 - np.exp(-kappa * T_coupons[:, None])) / kappa
        
        # Annuity sensitivity: dlnA / dX = (1/A_0) * sum(tau_j * D_j * B_j)
        dlnA_dX = np.sum(tau_coupons[:, None] * D_coupons[:, None] * B_coupons, axis=0) / A_0
        
        # Swap rate loading: Lambda = (D_alpha*B_alpha - D_beta*B_beta)/A_0 - S_0 * dlnA_dX
        Lambda = ((D_alpha * B_alpha - D_beta * B_beta) / A_0) - (S_0 * dlnA_dX)
        
        return S_0, dlnA_dX, Lambda
    # --------------------------------------------------------
    
    # 2. Extract specific leg loadings and sensitivities
    S_10_0, dlnA10_dX, Lambda_10 = get_loadings(cell_data_10Y)
    S_2_0, dlnA2_dX, Lambda_2 = get_loadings(cell_data_2Y)
    
    # 3. Calculate True Instantaneous Volatilities
    vol_10 = Lambda_10 * sigma
    vol_2  = Lambda_2 * sigma
    
    # 4. Calculate Convexity Adjustments (Covariance with Numeraire)
    CA_10 = T_alpha * np.sum(vol_10 * sigma * (B_alpha - dlnA10_dX))
    CA_2  = T_alpha * np.sum(vol_2  * sigma * (B_alpha - dlnA2_dX))
    
    # 5. Adjusted Forward Rates under T_alpha-forward measure
    E_S10 = S_10_0 + CA_10
    E_S2  = S_2_0 + CA_2
    mu_spread = E_S10 - E_S2
    
    # 6. Variance and Volatility of the Spread
    var_spread = T_alpha * np.sum((vol_10 - vol_2)**2)
    sigma_spread = np.sqrt(var_spread)
    
    # 7. Evaluate Bachelier Formula
    if sigma_spread > 1e-12:
        d = (mu_spread - strike) / sigma_spread
    else:
        d = np.inf if (mu_spread - strike) > 0 else -np.inf
        
    if option_type.lower() == 'call':
        intrinsic = (mu_spread - strike) * norm.cdf(d)
        time_value = sigma_spread * norm.pdf(d)
    elif option_type.lower() == 'put':
        intrinsic = (strike - mu_spread) * norm.cdf(-d)
        time_value = sigma_spread * norm.pdf(-d)
    else:
        raise ValueError("option_type must be either 'call' or 'put'")
        
    PV = D_alpha * (intrinsic + time_value)
    
    # Return a dictionary of diagnostics for clean output printing
    results = {
        'PV': PV,
        'S_10_initial': S_10_0,
        'S_2_initial': S_2_0,
        'CA_10': CA_10,
        'CA_2': CA_2,
        'mu_spread': mu_spread,
        'sigma_spread': sigma_spread,
        'D_alpha': D_alpha
    }
    
    return results