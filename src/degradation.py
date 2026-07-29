import numpy as np
import pandas as pd

# =====================================================================
# THE DATA DEGRADATION ENGINE
# =====================================================================
# Each method in this class represents a distinct, real-world failure
# mode of clinical data collection. Parameters are chosen to reflect
# a spectrum of degradation severity rather than exact clinical rates.
# =====================================================================

class DataDegradationEngine:
    def __init__(self, random_state=42):
        self.rs = np.random.RandomState(random_state)

    def apply_mcar(self, df, missing_rate=0.05):
        """
        Simulates Missing Completely At Random (MCAR) across all features.

        Real-world problem:
            Some clinical measurements may be missing because of incomplete
            data entry, temporary device failure, or workflow disruptions.
            The missingness is unrelated to the actual values — a nurse may
            simply forget to record a reading.

        Parameters:
            missing_rate: Proportion of values randomly set to NaN.
                          5%  -> Minor recording issues
                          15% -> Noticeable clinical data incompleteness
                          30% -> Severe resource-constrained environment

        Note:
            These rates represent controlled stress-test levels covering
            mild to severe data availability degradation. They are not
            direct estimates from a specific clinical dataset.
        """
        corrupted_df = df.copy()
        # MCAR missingness simulates random loss of clinical measurements.
        # Missing rates represent increasing levels of data availability
        # problems encountered in low-resource healthcare environments.
        mask = self.rs.rand(*corrupted_df.shape) < missing_rate
        corrupted_df[mask] = np.nan
        return corrupted_df

    def apply_glucose_stockout(self, df, missing_rate=0.30):
        """
        Simulates blood glucose measurement unavailability (MNAR).

        Real-world problem:
            Unlike random missingness, this models equipment-specific failure.
            Blood glucose measurement (BS column) depends on the availability
            of test strips and dedicated devices. In under-resourced clinics,
            these consumables can run out, making glucose data unavailable
            for entire patient batches rather than randomly.

        Parameters:
            missing_rate: Proportion of glucose readings removed.

        Note:
            This is MNAR (Missing Not At Random) because the values are
            missing due to resource constraints, not random chance.
        """
        corrupted_df = df.copy()
        target_col = 'BS'
        if target_col in corrupted_df.columns:
            # MNAR simulation: glucose values are removed to represent
            # equipment-specific failure — glucose testing depends on
            # availability of specific devices or test strips.
            mask = self.rs.rand(len(corrupted_df)) < missing_rate
            corrupted_df.loc[mask, target_col] = np.nan
        return corrupted_df

    def inject_bp_noise(self, df, max_drift=15.0,
                        target_cols=None):
        """
        Injects Gaussian measurement error into blood pressure readings.

        Real-world problem:
            Blood pressure is a critical maternal health indicator, especially
            for detecting hypertension-related risks such as preeclampsia.
            Real BP devices introduce measurement errors due to calibration
            issues, device quality, cuff placement, and environmental factors.

        Noise model:
            error ~ Normal(0, sigma)  where  sigma = max_drift / 1.96
            This ensures ~95% of errors fall within [-max_drift, +max_drift].

        Severity levels (motivated by ISO 81060-2 BP validation standards):
            ±5 mmHg  -> Slightly inaccurate but functional device
            ±10 mmHg -> Realistic measurement uncertainty from common devices
            ±15 mmHg -> Poorly calibrated or degraded measurement equipment

        Note:
            BP noise simulates sensor measurement error. Drift values are
            motivated by BP device validation studies (ISO 81060-2 accuracy
            requirements). Larger deviations represent poor calibration,
            sensor ageing, or environmental effects.
        """
        if target_cols is None:
            target_cols = ['SystolicBP', 'DiastolicBP']

        corrupted_df = df.copy()
        # sigma scales drift so that 95% of errors are within ±max_drift
        sigma = max_drift / 1.96
        for col in target_cols:
            if col in corrupted_df.columns:
                noise = self.rs.normal(0, sigma, size=len(corrupted_df))
                corrupted_df[col] = corrupted_df[col] + noise
        return corrupted_df

    def inject_temperature_bias(self, df, bias_range=(-1.5, -0.5),
                                target_col='BodyTemp'):
        """
        Injects systematic negative bias into body temperature readings.

        Real-world problem:
            Temperature readings can differ from true values due to measurement
            method (oral, axillary, tympanic), device calibration drift, or
            the use of low-cost sensors that consistently underestimate.
            This represents systematic under-reporting rather than random noise.

        Noise model:
            bias ~ Uniform(bias_range[0], bias_range[1])
            A negative bias is applied — each patient's temperature is
            reduced by a value drawn from the bias range.

        Severity levels:
            -0.1 to -0.5°F -> Mild systematic under-reporting
            -0.3 to -1.0°F -> Moderate calibration drift
            -0.5 to -1.5°F -> Significant systematic underestimation

        Note:
            Temperature bias simulates systematic measurement error.
            Negative shifts represent devices that consistently underestimate
            actual body temperature.
        """
        corrupted_df = df.copy()
        if target_col in corrupted_df.columns:
            bias = self.rs.uniform(bias_range[0], bias_range[1],
                                   size=len(corrupted_df))
            corrupted_df[target_col] = corrupted_df[target_col] + bias
        return corrupted_df

    def apply_degradation(self, df, tier_config):
        """
        Applies a complete degradation configuration to a DataFrame.

        Orchestrates all degradation functions based on the severity
        parameters from a single TIERS entry. This keeps the notebook
        clean — it calls apply_degradation() once per tier rather than
        invoking each corruption function manually.

        Args:
            df (pd.DataFrame): Clean test data.
            tier_config (dict): A config dict from config.TIERS, e.g.:
                {
                    "missing_rate": 0.15,
                    "bp_drift": 10.0,
                    "temp_bias": (-1.0, -0.3)
                }

        Returns:
            corrupted_df (pd.DataFrame): Degraded data with NaN values.
        """
        corrupted = df.copy()

        if tier_config["missing_rate"] > 0:
            corrupted = self.apply_mcar(
                corrupted, missing_rate=tier_config["missing_rate"])
            corrupted = self.apply_glucose_stockout(
                corrupted, missing_rate=tier_config["missing_rate"])

        if tier_config["bp_drift"] > 0:
            corrupted = self.inject_bp_noise(
                corrupted, max_drift=tier_config["bp_drift"])

        if tier_config["temp_bias"] != (0.0, 0.0):
            corrupted = self.inject_temperature_bias(
                corrupted, bias_range=tier_config["temp_bias"])

        return corrupted

    def calculate_missing_rate(self, original_df, corrupted_df):
        """
        Validates degradation by comparing requested vs actual missing rates.

        Returns a dict with overall and per-column missing statistics.
        Useful for verifying that corruptions were applied as expected.

        Example output:
            {
                'overall_actual_rate': 0.148,
                'per_column': {'BS': 0.302, 'SystolicBP': 0.05, ...}
            }
        """
        n_total = original_df.size
        n_missing = corrupted_df.isna().sum().sum()
        overall_rate = n_missing / n_total

        per_col = {}
        for col in original_df.columns:
            n_col = len(corrupted_df)
            n_col_missing = corrupted_df[col].isna().sum()
            per_col[col] = round(n_col_missing / n_col, 4)

        return {
            "overall_actual_rate": round(overall_rate, 4),
            "per_column": per_col
        }
