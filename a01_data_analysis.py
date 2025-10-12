import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import config

class DataAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.analysis_results = {}
        
    def load_data(self):
        """Load and basic data validation"""
        print("📊 Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Data shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        return self.df
    
    def basic_analysis(self):
        """Perform basic data analysis"""
        print("\n🔍 Basic Data Analysis:")
        print(self.df.info())
        print("\nMissing values:")
        print(self.df.isnull().sum())
        
        # Check theft distribution
        theft_count = self.df['Theft'].sum()
        theft_percentage = (theft_count / len(self.df)) * 100
        print(f"\n🎯 Theft Distribution: {theft_count} cases ({theft_percentage:.2f}%)")
        
        self.analysis_results['theft_count'] = theft_count
        self.analysis_results['theft_percentage'] = theft_percentage
        
    def statistical_analysis(self):
        """Statistical analysis of features - FIXED VERSION"""
        print("\n📈 Statistical Analysis:")
        
        # Separate theft and normal cases
        theft_cases = self.df[self.df['Theft'] == 1]
        normal_cases = self.df[self.df['Theft'] == 0]
        
        stats_comparison = {}
        
        for feature in config.FEATURE_COLUMNS:
            theft_mean = theft_cases[feature].mean()
            normal_mean = normal_cases[feature].mean()
            difference = theft_mean - normal_mean
            t_stat, p_value = stats.ttest_ind(
                theft_cases[feature].dropna(), 
                normal_cases[feature].dropna()
            )
            
            stats_comparison[feature] = {
                'theft_mean': theft_mean,
                'normal_mean': normal_mean,
                'difference': difference,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        # Convert to DataFrame for better visualization - FIXED HERE
        stats_df = pd.DataFrame(stats_comparison).T
        
        # Ensure all numeric columns are actually numeric
        numeric_columns = ['theft_mean', 'normal_mean', 'difference', 'p_value']
        for col in numeric_columns:
            stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce')
        
        # Now create abs_difference as numeric
        stats_df['abs_difference'] = abs(stats_df['difference'])
        
        print("\nFeature Comparison (Theft vs Normal):")
        # Sort by absolute difference and display top features
        top_features = stats_df.nlargest(5, 'abs_difference')[['theft_mean', 'normal_mean', 'difference', 'p_value']]
        print(top_features)
        
        self.analysis_results['stats_comparison'] = stats_df
        return stats_df
    
    def create_correlation_analysis(self):
        """Analyze correlations with theft"""
        print("\n📊 Correlation Analysis:")
        
        # Calculate correlations with theft
        correlations = self.df[config.FEATURE_COLUMNS + ['Theft']].corr()['Theft'].sort_values(ascending=False)
        print("Correlation with Theft:")
        for feature, corr in correlations.items():
            if feature != 'Theft':
                print(f"  {feature}: {corr:.3f}")
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.df[config.FEATURE_COLUMNS + ['Theft']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                  fmt='.2f', linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.analysis_results['correlations'] = correlations
        return correlations
    
    def create_distribution_plots(self):
        """Create distribution plots for key features"""
        print("\n📈 Creating distribution plots...")
        
        key_features = ['Usage (kWh)', 'AverageDailyUsage', 'BillPaymentDelay (days)', 'NumberOfResidents']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(key_features):
            # Plot distributions for theft vs normal
            theft_data = self.df[self.df['Theft'] == 1][feature]
            normal_data = self.df[self.df['Theft'] == 0][feature]
            
            axes[i].hist(normal_data, bins=30, alpha=0.7, label='Normal', color='blue', density=True)
            axes[i].hist(theft_data, bins=30, alpha=0.7, label='Theft', color='red', density=True)
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create box plots for the same features
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(key_features):
            data_to_plot = [self.df[self.df['Theft'] == 0][feature], 
                          self.df[self.df['Theft'] == 1][feature]]
            axes[i].boxplot(data_to_plot, labels=['Normal', 'Theft'])
            axes[i].set_title(f'Box Plot of {feature}')
            axes[i].set_ylabel(feature)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feature_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self):
        """Generate comprehensive data analysis report - FIXED VERSION"""
        print("\n" + "="*50)
        print("📋 DATA ANALYSIS REPORT")
        print("="*50)
        
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Theft Cases: {self.analysis_results['theft_count']} ({self.analysis_results['theft_percentage']:.2f}%)")
        
        print("\n🔍 Key Insights:")
        stats_df = self.analysis_results['stats_comparison']
        
        # FIXED: Ensure we're working with numeric data
        top_features = stats_df.nlargest(3, 'abs_difference')
        
        for feature, row in top_features.iterrows():
            trend = "HIGHER" if row['difference'] > 0 else "LOWER"
            print(f"  - {feature}: Theft cases have {trend} values ({row['difference']:.2f})")
        
        print("\n🎯 Recommended Contamination Parameter:")
        recommended_contamination = min(0.15, max(0.05, self.analysis_results['theft_percentage'] / 100 + 0.02))
        print(f"  Suggested: {recommended_contamination:.3f}")
        
        # Update config with recommended contamination
        config.ISOLATION_FOREST_PARAMS['contamination'] = recommended_contamination
        print(f"  Updated model contamination parameter to: {recommended_contamination}")
        
        return self.analysis_results

def main():
    analyzer = DataAnalyzer(config.DATA_PATH)
    df = analyzer.load_data()
    analyzer.basic_analysis()
    analyzer.statistical_analysis()
    analyzer.create_correlation_analysis()
    analyzer.create_distribution_plots()
    results = analyzer.generate_report()
    
    return df, results

if __name__ == "__main__":
    df, results = main()