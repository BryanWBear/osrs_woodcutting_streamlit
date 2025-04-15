import streamlit as st
import math
import pandas as pd

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


def calculate_expected_time(starting_level: int, num_logs: int, xp_per_log: float, axe_type):
    d = simulate_logs_distribution(starting_level, num_logs, xp_per_log)
    expected_time = 0
    for level, logs in d.items():
        chance = woodcut_success_chance(level, axe_high_low[axe_type][0], axe_high_low[axe_type][1])
        expected_time += (logs * (1 - chance) / chance) * 2.4

    return expected_time / 3600, max(list(d.keys()))

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

st.title("OSRS Woodcutting Grind Simulator")

st.markdown("""
This app simulates **how long it takes to cut a certain number of logs** in OSRS, and your level at the end of the grind.
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

# Run the simulation when the user clicks a button
if st.button("Mean (Average) Grind Time"):
    distribution, ending_level = calculate_expected_time(current_level, num_logs, xp_per_log, selected_axe)
    st.markdown(f"**Total expected time to cut {num_logs} logs with {selected_axe}:** {distribution} hours")
    st.markdown(f"**Level at end of the grind:** {ending_level}")

st.markdown("""
---
This simulator uses 
            
1. OSRS XP formula:
            
$$XP(L) = \sum_{i=1}^{L–1} \left\lfloor (i + 300 \cdot 2^{i/7}) / 4 \\right\\rfloor$$
            
2. Sum of mean of negative binomial distributions
3. Axe high and low chances (out of 255) from wiki + linear interpolation
""")

st.table(pd.DataFrame(axe_high_low).T.rename(columns={0: "Low Chance", 1: "High Chance"}))
st.markdown("""to calculate the expected time to cut $N$ logs. Does not account for respawn times, and assumes that 4 game ticks = 1 woodcutting roll = 2.4 seconds.""")
