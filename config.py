# Configuration file for electricity theft detection

# File paths
DATA_PATH = 'electricity_new.csv'
MODEL_SAVE_PATH = 'iso_forest.pkl'
SCALER_SAVE_PATH = 'scaler.pkl'
RESULTS_SAVE_PATH = 'anomaly_results.csv'

# Model parameters
ISOLATION_FOREST_PARAMS = {
    'n_estimators': 200,
    'max_samples': 'auto',
    'contamination': 0.08,  # Based on data analysis
    'random_state': 42,
    'verbose': 1
}

# Feature columns to use
FEATURE_COLUMNS = [
    'Usage (kWh)', 'TimeOfDay', 'VoltageFluctuations',
    'NumberOfResidents', 'ApplianceCount', 'IndustrialAreaNearby',
    'PreviousTheftHistory', 'AverageDailyUsage', 'BillPaymentDelay (days)',
    'UnusualUsageSpike'
]

# Business rule thresholds
BUSINESS_RULES = {
    'low_consumption_ratio': 0.3,
    'high_bill_delay': 60,
    'suspicious_voltage_fluctuation': 2,
    'high_risk_industrial_area': 1
}

# Validation parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42