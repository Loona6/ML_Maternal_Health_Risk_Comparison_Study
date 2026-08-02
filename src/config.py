import os

RANDOM_STATE = 42
N_SPLITS = 5
DATASET_FILENAME = "Maternal Health Risk Data Set.csv"

# Make dataset path robust
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "raw", DATASET_FILENAME)

# =====================================================================
# REPRODUCIBILITY SETTINGS
# =====================================================================
# Ten seeds ensure that results are not dependent on a single random
# corruption pattern. Reporting mean ± std across seeds gives
# statistically reliable robustness estimates.
DEGRADATION_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]

# =====================================================================
# EXPERIMENTAL SAFETY CRITERION
# =====================================================================
# A model is flagged for analysis if its High Risk Recall drops by
# more than this threshold from the Clean baseline.
# This is a research decision for robustness comparison, not a
# clinically validated guideline.
HIGH_RISK_RECALL_DROP_THRESHOLD = 0.10

# =====================================================================
# CLINICAL SEVERITY TIERS
# =====================================================================
# Tier names are kept consistent with the original naming convention.
#
# Missingness levels represent controlled stress-test scenarios,
# not exact estimates of clinical missing data rates.
#
# BP drift values are motivated by BP device validation studies
# (ISO 81060-2). Lower values represent acceptable device variation,
# while higher values simulate degraded measurement conditions.
#
# Temperature bias represents systematic under-reporting from
# low-cost or poorly calibrated measurement devices.
# Sentinel tier names — used to apply tier-specific logic (e.g. seed skipping, flagging).
PRISTINE_TIER_NAME = "Pristine Baseline"
SEVERE_TIER_NAME   = "Tier 3 (Severe)"

TIERS = {
    "Pristine Baseline": {
        "missing_rate": 0.00,
        "bp_drift":     0.0,
        "temp_bias":    (0.0, 0.0)
    },
    "Tier 1 (Mild)": {
        "missing_rate": 0.05,   # Minor recording issues
        "bp_drift":     5.0,    # ±5 mmHg: slightly inaccurate but functional device
        "temp_bias":    (-0.5, -0.1)  # Mild systematic under-reporting
    },
    "Tier 2 (Moderate)": {
        "missing_rate": 0.15,   # Noticeable clinical data incompleteness
        "bp_drift":     10.0,   # ±10 mmHg: realistic uncertainty from common devices
        "temp_bias":    (-1.0, -0.3)  # Moderate calibration drift
    },
    "Tier 3 (Severe)": {
        "missing_rate": 0.30,   # Severe resource-constrained environment
        "bp_drift":     15.0,   # ±15 mmHg: poorly calibrated or degraded device
        "temp_bias":    (-1.5, -0.5)  # Significant systematic underestimation
    }
}
