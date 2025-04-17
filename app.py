import streamlit as st
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.integrate import quad
from scipy import stats

def cf_negative_binomial(t, r, p):
    """Characteristic function of a Negative Binomial distribution."""
    return (p / (1 - (1 - p) * np.exp(1j * t))) ** r

def cf_product(t, params_list):
    """Product of characteristic functions of multiple NB distributions."""
    cf = np.ones_like(t, dtype=complex)
    for r, p in params_list:
        cf *= cf_negative_binomial(t, r, p)
    return cf

def compute_mean_std(params_list):
    """Compute mean and std of a sum of independent NB distributions."""
    means = [(r * (1 - p)) / p for r, p in params_list]
    variances = [(r * (1 - p)) / (p ** 2) for r, p in params_list]
    mean_y = sum(means)
    std_y = np.sqrt(sum(variances))
    return mean_y, std_y

def inverse_cf_pdf(y_vals, cf_func):
    """Numerically compute PDF using Gil-Pelaez inversion formula. Formula from Theorem 6.7 page 78 of https://wiredspace.wits.ac.za/server/api/core/bitstreams/aa0a75c9-30ef-4508-8134-9e4838ac2bd0/content"""
    pdf = []
    for y in y_vals:
        integrand = lambda t: np.real(np.exp(-1j * t * y) * cf_func(t))
        integral, _ = quad(integrand, -np.pi, np.pi)
        pdf.append(integral / 2*np.pi)
    return np.array(pdf)


def run_inversion_with_six_sigma(params_list, sigma_factor=6):
    """Compute and plot PDF using numerical inversion with bounds from six-sigma rule."""
    # Characteristic function
    cf_func = lambda t: cf_product(t, params_list)

    # Get mean and std for sum of NB
    mean_y, std_y = compute_mean_std(params_list)
    A = mean_y - sigma_factor * std_y
    B = mean_y + sigma_factor * std_y
    y_vals = np.arange(max(0, int(A)), int(B))

    # Compute PDF using numerical integration
    pdf_vals = inverse_cf_pdf(y_vals, cf_func)

    return y_vals, pdf_vals, mean_y, std_y


def xp_for_level(level):
    """
    Compute total XP required for the given level in OSRS.
    Level 1 requires 0 XP.

    Formula: total XP for level L is the sum for i=1 to L-1 of:
    floor((i + 300 * 2^(i/7)) / 4)

    This formula is widely used in OSRS XP calculations.
    """
    if level <= 1:
        return 0
    total_xp = 0
    for i in range(1, level):
        xp_i = math.floor((i + 300 * (2 ** (i / 7))) / 4)
        total_xp += xp_i
    return total_xp


def simulate_logs_distribution(current_level, num_logs, xp_per_log=25):
    """
    Simulate log cutting, reporting how many logs are cut at each woodcutting level.

    Parameters:
    current_level: Player's current woodcutting level (at least 1, and below 99).
    num_logs: Total number of logs the player intends to cut.
    xp_per_log: XP earned per log (default is 25 XP, as with regular logs).

    Returns:
    A dictionary mapping levels to the number of logs cut while at that level.

    The simulation works by starting with the total XP for the player's
    current level (as given by xp_for_level) and then "adding" xp_per_log per log.
    When XP reaches the threshold for the next level (via xp_for_level), the count
    for the current level is recorded and the simulation continues at the next level.

    Note: The simulation assumes that the XP per log is constant (which is true
    if you are using the same tree type).
    """

    # Initialize current XP based on the player's current level.
    current_xp = xp_for_level(current_level)
    level_distribution = {} # Will map level -> number of logs cut at that level.
    logs_remaining = num_logs
    level = current_level

    # Simulate until either we run out of logs or we reach level 99.
    while logs_remaining > 0 and level < 99:
        # XP required to reach the next level
        xp_next_level = xp_for_level(level + 1)
        xp_gap = xp_next_level - current_xp

        # How many logs are needed to level up from current level?
        logs_needed = math.ceil(xp_gap / xp_per_log)

        if logs_needed > logs_remaining:
            # Not enough logs to level up.
            level_distribution[level] = level_distribution.get(level, 0) + logs_remaining
            current_xp += logs_remaining * xp_per_log
            logs_remaining = 0
        else:
            # Record that we cut logs_needed logs at the current level.
            level_distribution[level] = level_distribution.get(level, 0) + logs_needed
            logs_remaining -= logs_needed
            # Set XP exactly at threshold and move to next level.
            current_xp = xp_next_level
            level += 1

    # If logs remain (and level is 99), record them all at level 99.
    if logs_remaining > 0:
        level_distribution[99] = level_distribution.get(99, 0) + logs_remaining

    return level_distribution

def woodcut_success_chance(level, low_chance, high_chance):
    slope = (high_chance - low_chance) / 99.
    return (slope * level + low_chance) / 255


def calculate_approx_expected_time(starting_level: int, num_logs: int, xp_per_log: float, axe_type):
    d = simulate_logs_distribution(starting_level, num_logs, xp_per_log)
    params = []
    for level, logs in d.items():
        chance = woodcut_success_chance(level, axe_high_low[axe_type][0], axe_high_low[axe_type][1])
        params.append((logs, chance))
    mu, sigma = compute_mean_std(params)
    mu = (mu + num_logs) * 2.4 / 3600
    sigma *= 2.4 / 3600
    return mu, sigma, max(list(d.keys()))


def calculate_woodcutting_time_distribution(starting_level: int, num_logs: int, xp_per_log: float, axe_type: str):
    d = simulate_logs_distribution(starting_level, num_logs, xp_per_log)
    params = []
    for level, logs in d.items():
        chance = woodcut_success_chance(level, axe_high_low[axe_type][0], axe_high_low[axe_type][1])
        params.append((logs, chance))
    pdf_support, pdf_vals, mu, sigma = run_inversion_with_six_sigma(params)
    pdf_support = (pdf_support + num_logs) * 2.4 / 3600
    
    # Normalize pdf, mu, and sigma
    pdf_vals /= np.sum(pdf_vals)
    mu = (mu + num_logs) * 2.4 / 3600
    sigma *= 2.4 / 3600

    # Calculate the cumulative distribution function (CDF)
    cdf = np.cumsum(pdf_vals)

    # Find the quartiles
    quartiles = np.interp([0.25, 0.5, 0.75], cdf, pdf_support)
    return pdf_support, pdf_vals, mu, sigma, quartiles, max(list(d.keys()))




axe_assets = {
"Bronze Axe": "https://oldschool.runescape.wiki/images/Bronze_axe.png",
"Iron Axe": "https://oldschool.runescape.wiki/images/Iron_axe.png",
"Steel Axe": "https://oldschool.runescape.wiki/images/Steel_axe.png",
"Black Axe": "https://oldschool.runescape.wiki/images/Black_axe.png",
"Mithril Axe": "https://oldschool.runescape.wiki/images/Mithril_axe.png",
"Adamant Axe": "https://oldschool.runescape.wiki/images/Adamant_axe.png",
"Rune Axe": "https://oldschool.runescape.wiki/images/Rune_axe.png",
"Dragon Axe": "https://oldschool.runescape.wiki/images/Dragon_axe.png",
}

axe_high_low = {
    'Bronze Axe': (16, 50),
    'Iron Axe': (24, 75),
    'Steel Axe': (32, 100),
    'Black Axe': (36, 112),
    'Mithril Axe': (40, 125),
    'Adamant Axe': (48, 150),
    'Rune Axe': (56, 175),
    'Dragon Axe': (60, 187)
}


# Log type lookup: key is log type, value is a tuple (image URL, xp gained per log)
log_assets = {
"Normal Log": ("https://oldschool.runescape.wiki/images/Logs.png", 25),
"Oak Log": ("https://oldschool.runescape.wiki/images/Oak_logs.png", 37.5),
"Willow Log": ("https://oldschool.runescape.wiki/images/Willow_logs.png", 67.5),
"Teak Log": ("https://oldschool.runescape.wiki/images/Teak_logs.png", 85),
"Maple Log": ("https://oldschool.runescape.wiki/images/Maple_logs.png", 100),
"Mahogany Log": ("https://oldschool.runescape.wiki/images/Mahogany_logs.png", 125),
"Yew Log": ("https://oldschool.runescape.wiki/images/Yew_logs.png", 175),
"Magic Log": ("https://oldschool.runescape.wiki/images/Magic_logs.png", 250),
"Redwood Log": ("https://oldschool.runescape.wiki/images/Redwood_logs.png", 380)
}

# -----------------------
# Streamlit UI
# -----------------------

st.title("OSRS Woodcutting Grind Calculator")

st.markdown("""
This app calculates **how long it takes to cut a certain number of logs** in OSRS, and your level at the end of the grind.
Gives some idea of grind variance based on the distribution of possible grind times. 25th, 50th, and 75th percentiles are highlighted with colored lines. 
Select the axe type and log type below (the log type determines the XP gained per log),
then input your current woodcutting level and the total number of logs you plan to cut.
""")

# Sidebar for asset selection:
st.sidebar.header("Settings")
selected_axe = st.sidebar.selectbox("Select Axe Type", list(axe_assets.keys()))
st.sidebar.image(axe_assets[selected_axe], width=120)
selected_log = st.sidebar.selectbox("Select Log Type", list(log_assets.keys()))
st.sidebar.image(log_assets[selected_log][0], width=120)

# Input parameters from main page:
current_level = st.number_input("Enter your current Woodcutting level", min_value=1, max_value=99, value=15, step=1)
num_logs = st.number_input("Enter the total number of logs you plan to cut", min_value=1, value=100, step=1)

# Determine XP per log based on selected log type
xp_per_log = log_assets[selected_log][1]
st.write(f"**XP per {selected_log}:** {xp_per_log}")

if st.button("Approximate Grind Time Stats"):
    mu, sigma, ending_level = calculate_approx_expected_time(current_level, num_logs, xp_per_log, selected_axe)
    st.markdown(f"**Mean expected time to cut {num_logs} {selected_log} with {selected_axe}:** {mu:.2f} hours")
    st.markdown(f"**Level at end of the grind:** {ending_level}")

    # 1st quartile (25th percentile)
    q1 = stats.norm.ppf(0.25, loc=mu, scale=sigma)

    # 2nd quartile (50th percentile, median)
    q2 = stats.norm.ppf(0.50, loc=mu, scale=sigma)

    # 3rd quartile (75th percentile)
    q3 = stats.norm.ppf(0.75, loc=mu, scale=sigma)

    st.write(f"**Time Quartiles (in hours) from normal approximation:**")
    st.write(f"Q1 (25th percentile): {q1:.2f} hours")
    st.write(f"Median (50th percentile): {q2:.2f} hours")
    st.write(f"Q3 (75th percentile): {q3:.2f} hours")

    A = mu - 6 * sigma
    B = mu + 6 * sigma
    y_vals = np.arange(max(0, int(A)), int(B))
    pdf_vals = stats.norm.pdf(y_vals, loc=mu, scale=sigma)

    # Create histogram
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_vals,
        y=pdf_vals,
        mode='lines',
        line=dict(color='blue'),
        name='Approximate PMF'
    ))

    # Update layout for better visualization
    fig.update_layout(
        title="Approximate Probability Mass Function (PMF) of Time Distribution Using Normal",
        xaxis_title="Time (hours)",
        yaxis_title="Probability Density",
        template="plotly_white",
        legend_title="Legend"
    )

    # Add vertical lines for quartiles
    fig.add_trace(go.Scatter(
        x=[q1, q1],
        y=[0, max(pdf_vals)],
        mode='lines',
        line=dict(color='orange', dash='dash'),
        name=f"Q1: {q1:.2f}h"
    ))
    fig.add_trace(go.Scatter(
        x=[q2, q2],
        y=[0, max(pdf_vals)],
        mode='lines',
        line=dict(color='green', dash='dash'),
        name=f"Median: {q2:.2f}h"
    ))
    fig.add_trace(go.Scatter(
        x=[q3, q3],
        y=[0, max(pdf_vals)],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name=f"Q3: {q3:.2f}h"
    ))

    # Display the plot in Streamlit
    st.plotly_chart(fig)

# Run the simulation when the user clicks a button
if st.button("Grind Time Stats"):
    pdf_support, pdf_vals, mu, sigma, quartiles, ending_level = calculate_woodcutting_time_distribution(current_level, num_logs, xp_per_log, selected_axe)
    st.markdown(f"**Mean expected time to cut {num_logs} {selected_log} with {selected_axe}:** {mu:.2f} hours")
    st.markdown(f"**Level at end of the grind:** {ending_level}")


    # Calculate quartiles
    q1, q2, q3 = quartiles
    st.write(f"**True Time Quartiles (in hours):**")
    st.write(f"Q1 (25th percentile): {q1:.2f} hours")
    st.write(f"Median (50th percentile): {q2:.2f} hours")
    st.write(f"Q3 (75th percentile): {q3:.2f} hours")

    # Create histogram
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pdf_support,
        y=pdf_vals,
        mode='lines',
        line=dict(color='blue'),
        name='PMF'
    ))

    # Update layout for better visualization
    fig.update_layout(
        title="Probability Mass Function (PMF) of Time Distribution",
        xaxis_title="Time (hours)",
        yaxis_title="Probability Density",
        template="plotly_white",
        legend_title="Legend"
    )

    # Add vertical lines for quartiles
    fig.add_trace(go.Scatter(
        x=[q1, q1],
        y=[0, max(pdf_vals)],
        mode='lines',
        line=dict(color='orange', dash='dash'),
        name=f"Q1: {q1:.2f}h"
    ))
    fig.add_trace(go.Scatter(
        x=[q2, q2],
        y=[0, max(pdf_vals)],
        mode='lines',
        line=dict(color='green', dash='dash'),
        name=f"Median: {q2:.2f}h"
    ))
    fig.add_trace(go.Scatter(
        x=[q3, q3],
        y=[0, max(pdf_vals)],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name=f"Q3: {q3:.2f}h"
    ))

    # Display the plot in Streamlit
    st.plotly_chart(fig)

st.markdown("""
---
This calculator uses 
            
1. OSRS XP formula:
            
$$XP(L) = \sum_{i=1}^{L–1} \left\lfloor (i + 300 \cdot 2^{i/7}) / 4 \\right\\rfloor$$
            
2. Inversion of characteristic function of a sum of independent negative binomial distributions to calculate the distribution of time to cut $N$ logs. This numerical integration is kind of shitty and slow, and somewhat erratic.
3. Axe high and low chances (out of 255) from wiki + linear interpolation
""")

st.table(pd.DataFrame(axe_high_low).T.rename(columns={0: "Low Chance", 1: "High Chance"}))
st.markdown("""to calculate the expected time to cut $N$ logs. Does not account for respawn times, and assumes that 4 game ticks = 1 woodcutting roll = 2.4 seconds.""")

