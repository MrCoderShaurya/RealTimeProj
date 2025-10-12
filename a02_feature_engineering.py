import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import config

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.engineered_features = []
        
    def create_basic_features(self):
        """Create basic engineered features"""
        print("🔧 Creating basic engineered features...")
        
        try:
            # Usage patterns
            self.df['usage_per_resident'] = self.df['Usage (kWh)'] / (self.df['NumberOfResidents'] + 1)
            self.df['usage_per_appliance'] = self.df['Usage (kWh)'] / (self.df['ApplianceCount'] + 1)
            
            # Behavioral features
            self.df['usage_vs_average_ratio'] = self.df['Usage (kWh)'] / (self.df['AverageDailyUsage'] + 1)
            self.df['bill_delay_risk'] = np.where(self.df['BillPaymentDelay (days)'] > config.BUSINESS_RULES['high_bill_delay'], 1, 0)
            
            # Risk composite features
            self.df['high_risk_area'] = ((self.df['IndustrialAreaNearby'] == 1) & 
                                       (self.df['PreviousTheftHistory'] == 1)).astype(int)
            
            self.df['suspicious_voltage'] = (self.df['VoltageFluctuations'] >= 
                                           config.BUSINESS_RULES['suspicious_voltage_fluctuation']).astype(int)
            
            # Time-based risk (assuming higher risk during certain hours)
            self.df['high_risk_time'] = ((self.df['TimeOfDay'] >= 2) & (self.df['TimeOfDay'] <= 4)).astype(int)
            
            self.engineered_features.extend([
                'usage_per_resident', 'usage_per_appliance', 'usage_vs_average_ratio',
                'bill_delay_risk', 'high_risk_area', 'suspicious_voltage', 'high_risk_time'
            ])
            
        except Exception as e:
            print(f"❌ Error in basic feature engineering: {e}")
            
        return self.df
    
    def create_interaction_features(self):
        """Create interaction features between important variables"""
        print("🔧 Creating interaction features...")
        
        try:
            # Interaction between usage patterns and resident count
            self.df['low_usage_high_residents'] = (
                (self.df['usage_per_resident'] < self.df['usage_per_resident'].quantile(0.3)) &
                (self.df['NumberOfResidents'] > self.df['NumberOfResidents'].median())
            ).astype(int)
            
            # Interaction between payment delay and usage spikes
            self.df['delay_and_spike'] = (
                (self.df['BillPaymentDelay (days)'] > config.BUSINESS_RULES['high_bill_delay']) &
                (self.df['UnusualUsageSpike'] == 1)
            ).astype(int)
            
            # Composite risk score (simple weighted sum)
            risk_factors = [
                'PreviousTheftHistory',
                'IndustrialAreaNearby', 
                'suspicious_voltage',
                'bill_delay_risk'
            ]
            
            self.df['composite_risk_score'] = self.df[risk_factors].sum(axis=1)
            
            self.engineered_features.extend([
                'low_usage_high_residents', 'delay_and_spike', 'composite_risk_score'
            ])
            
        except Exception as e:
            print(f"❌ Error in interaction feature engineering: {e}")
            
        return self.df
    
    def create_statistical_features(self):
        """Create statistical features based on dataset characteristics"""
        print("🔧 Creating statistical features...")
        
        try:
            # Z-scores for usage
            usage_mean = self.df['Usage (kWh)'].mean()
            usage_std = self.df['Usage (kWh)'].std()
            self.df['usage_zscore'] = (self.df['Usage (kWh)'] - usage_mean) / usage_std
            
            # Percentile ranks
            self.df['usage_percentile'] = self.df['Usage (kWh)'].rank(pct=True)
            self.df['delay_percentile'] = self.df['BillPaymentDelay (days)'].rank(pct=True)
            
            # Binary flags for extremes
            self.df['extremely_low_usage'] = (self.df['usage_zscore'] < -2).astype(int)
            self.df['extremely_high_delay'] = (self.df['delay_percentile'] > 0.9).astype(int)
            
            self.engineered_features.extend([
                'usage_zscore', 'usage_percentile', 'delay_percentile',
                'extremely_low_usage', 'extremely_high_delay'
            ])
            
        except Exception as e:
            print(f"❌ Error in statistical feature engineering: {e}")
            
        return self.df
    
    def prepare_features_for_training(self):
        """Prepare final feature set and scale features"""
        print("🔧 Preparing features for training...")
        
        try:
            # Combine original and engineered features
            all_features = config.FEATURE_COLUMNS + self.engineered_features
            
            # Remove any potential duplicates
            all_features = list(set(all_features))
            
            print(f"Total features: {len(all_features)}")
            print("Features:", all_features)
            
            # Handle any infinite values
            self.df = self.df.replace([np.inf, -np.inf], np.nan)
            
            # Fill remaining NaN values with 0
            self.df[all_features] = self.df[all_features].fillna(0)
            
            # Ensure all features are numeric
            for feature in all_features:
                self.df[feature] = pd.to_numeric(self.df[feature], errors='coerce')
            
            self.df[all_features] = self.df[all_features].fillna(0)
            
            # Scale features
            X = self.df[all_features]
            X_scaled = self.scaler.fit_transform(X)
            
            # Create feature names for scaled features
            feature_names = [f"{col}_scaled" for col in all_features]
            X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names, index=self.df.index)
            
            # Combine with original dataframe
            self.df = pd.concat([self.df, X_scaled_df], axis=1)
            
            self.final_feature_columns = feature_names
            self.X_scaled = X_scaled
            
            print(f"✅ Final scaled features: {len(self.final_feature_columns)}")
            
        except Exception as e:
            print(f"❌ Error in feature preparation: {e}")
            # Fallback to original features only
            all_features = config.FEATURE_COLUMNS
            X = self.df[all_features].fillna(0)
            self.X_scaled = self.scaler.fit_transform(X)
            self.final_feature_columns = [f"{col}_scaled" for col in all_features]
            
        return self.df, self.X_scaled, self.final_feature_columns
    
    def get_feature_summary(self):
        """Get summary of engineered features"""
        summary = {}
        for feature in self.engineered_features:
            try:
                summary[feature] = {
                    'mean': self.df[feature].mean(),
                    'std': self.df[feature].std(),
                    'correlation_with_theft': self.df[feature].corr(self.df['Theft'])
                }
            except:
                summary[feature] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'correlation_with_theft': np.nan
                }
        
        return pd.DataFrame(summary).T

def main(df):
    engineer = FeatureEngineer(df)
    engineer.create_basic_features()
    engineer.create_interaction_features()
    engineer.create_statistical_features()
    df_processed, X_scaled, feature_columns = engineer.prepare_features_for_training()
    feature_summary = engineer.get_feature_summary()
    
    print("\n🔍 Engineered Features Summary:")
    # Handle case where correlation_with_theft might have NaN
    if 'correlation_with_theft' in feature_summary.columns:
        feature_summary_clean = feature_summary.dropna(subset=['correlation_with_theft'])
        if not feature_summary_clean.empty:
            print(feature_summary_clean.sort_values('correlation_with_theft', ascending=False))
        else:
            print("No valid correlation data available")
    else:
        print(feature_summary)
    
    return df_processed, X_scaled, feature_columns, engineer.scaler

if __name__ == "__main__":
    # This would be called from main pipeline
    pass