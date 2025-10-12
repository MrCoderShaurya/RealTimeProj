import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import config

class SHAPAnalyzer:
    def __init__(self, model, X_scaled, feature_names, df):
        self.model = model
        self.X_scaled = X_scaled
        self.feature_names = feature_names
        self.df = df
        self.explainer = None
        self.shap_values = None
        
    def create_shap_explainer(self):
        """Create SHAP explainer for the Isolation Forest model"""
        print("🔍 Creating SHAP explainer...")
        
        try:
            # Create SHAP explainer for Isolation Forest
            self.explainer = shap.TreeExplainer(self.model)
            
            # Calculate SHAP values
            self.shap_values = self.explainer.shap_values(self.X_scaled)
            
            print("✅ SHAP explainer created successfully!")
        except Exception as e:
            print(f"❌ Error creating SHAP explainer: {e}")
            self.explainer = None
            self.shap_values = None
            
        return self.explainer, self.shap_values
    
    def summary_plot(self):
        """Create SHAP summary plot"""
        if self.shap_values is None:
            print("❌ No SHAP values available")
            return
            
        print("📊 Creating SHAP summary plot...")
        
        try:
            plt.figure(figsize=(12, 8))
            shap.summary_plot(self.shap_values, self.X_scaled, 
                            feature_names=self.feature_names, 
                            show=False)
            plt.title('SHAP Summary Plot - Feature Importance for Anomaly Detection')
            plt.tight_layout()
            plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Also create bar plot for mean absolute SHAP values
            plt.figure(figsize=(12, 8))
            shap.summary_plot(self.shap_values, self.X_scaled, 
                            feature_names=self.feature_names, 
                            plot_type="bar", show=False)
            plt.title('SHAP Feature Importance (Mean Absolute Impact)')
            plt.tight_layout()
            plt.savefig('shap_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"❌ Error creating SHAP summary plot: {e}")
    
    def analyze_individual_predictions(self, num_cases=10):
        """Analyze individual predictions for top anomalies"""
        if self.shap_values is None:
            print("❌ No SHAP values available")
            return
            
        print(f"🔍 Analyzing top {num_cases} anomalies...")
        
        # Get top anomalies (highest anomaly probability)
        top_anomalies_indices = self.df.nlargest(num_cases, 'anomaly_probability').index
        
        for i, idx in enumerate(top_anomalies_indices):
            actual_theft = self.df.loc[idx, 'Theft']
            anomaly_prob = self.df.loc[idx, 'anomaly_probability']
            
            print(f"\n{'='*50}")
            print(f"🚨 Case {i+1}: Household Index {idx}")
            print(f"Anomaly Probability: {anomaly_prob:.4f}")
            print(f"Actual Theft: {actual_theft}")
            print(f"{'='*50}")
            
            try:
                # Create force plot for individual prediction
                plt.figure(figsize=(12, 4))
                shap.force_plot(self.explainer.expected_value, 
                              self.shap_values[idx, :], 
                              self.X_scaled[idx, :],
                              feature_names=self.feature_names,
                              matplotlib=True, show=False)
                plt.title(f'SHAP Explanation for Case {i+1} (Theft: {actual_theft})')
                plt.tight_layout()
                plt.savefig(f'shap_force_plot_case_{i+1}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Print feature contributions
                feature_contributions = pd.DataFrame({
                    'feature': self.feature_names,
                    'value': self.X_scaled[idx, :],
                    'shap_value': self.shap_values[idx, :]
                })
                
                # Sort by absolute SHAP value
                feature_contributions['abs_shap'] = abs(feature_contributions['shap_value'])
                top_contributors = feature_contributions.nlargest(5, 'abs_shap')
                
                print("Top 5 contributing features:")
                for _, row in top_contributors.iterrows():
                    direction = "INCREASES" if row['shap_value'] > 0 else "DECREASES"
                    print(f"  {row['feature']}: {row['shap_value']:.4f} ({direction} anomaly score)")
                    
            except Exception as e:
                print(f"❌ Error analyzing case {i+1}: {e}")
    
    def dependency_analysis(self, top_features=5):
        """Create dependency plots for top features"""
        if self.shap_values is None:
            print("❌ No SHAP values available")
            return
            
        print(f"📈 Creating dependency plots for top {top_features} features...")
        
        try:
            # Get top features by mean absolute SHAP value
            mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)
            top_feature_indices = np.argsort(mean_abs_shap)[-top_features:][::-1]
            
            for i, feature_idx in enumerate(top_feature_indices):
                feature_name = self.feature_names[feature_idx]
                
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(feature_idx, self.shap_values, self.X_scaled,
                                  feature_names=self.feature_names,
                                  show=False)
                plt.title(f'SHAP Dependency Plot: {feature_name}')
                plt.tight_layout()
                plt.savefig(f'shap_dependency_{feature_name}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Created dependency plot for {feature_name}")
        except Exception as e:
            print(f"❌ Error in dependency analysis: {e}")
    
    def compare_theft_vs_normal_shap(self):
        """Compare SHAP values between theft and normal cases"""
        if self.shap_values is None:
            print("❌ No SHAP values available")
            return None
            
        print("🔍 Comparing SHAP patterns: Theft vs Normal cases...")
        
        theft_indices = self.df[self.df['Theft'] == 1].index
        normal_indices = self.df[self.df['Theft'] == 0].index
        
        if len(theft_indices) > 0 and len(normal_indices) > 0:
            try:
                # Calculate mean absolute SHAP for each group
                theft_shap_mean = np.mean(np.abs(self.shap_values[theft_indices, :]), axis=0)
                normal_shap_mean = np.mean(np.abs(self.shap_values[normal_indices, :]), axis=0)
                
                # Create comparison DataFrame
                comparison_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'theft_importance': theft_shap_mean,
                    'normal_importance': normal_shap_mean
                })
                
                comparison_df['difference'] = comparison_df['theft_importance'] - comparison_df['normal_importance']
                comparison_df['ratio'] = comparison_df['theft_importance'] / comparison_df['normal_importance']
                
                print("\n📊 Feature Importance Comparison (Theft vs Normal):")
                print(comparison_df.nlargest(10, 'difference')[['feature', 'theft_importance', 'normal_importance', 'difference']])
                
                # Plot comparison
                plt.figure(figsize=(12, 8))
                top_features = comparison_df.nlargest(10, 'difference')
                
                x = np.arange(len(top_features))
                width = 0.35
                
                plt.bar(x - width/2, top_features['theft_importance'], width, label='Theft Cases', alpha=0.8)
                plt.bar(x + width/2, top_features['normal_importance'], width, label='Normal Cases', alpha=0.8)
                
                plt.xlabel('Features')
                plt.ylabel('Mean |SHAP Value|')
                plt.title('Feature Importance: Theft Cases vs Normal Cases')
                plt.xticks(x, top_features['feature'], rotation=45, ha='right')
                plt.legend()
                plt.tight_layout()
                plt.savefig('shap_theft_vs_normal.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                return comparison_df
            except Exception as e:
                print(f"❌ Error in theft vs normal comparison: {e}")
                return None
        
        return None
    
    def generate_shap_report(self):
        """Generate comprehensive SHAP analysis report"""
        print("\n" + "="*50)
        print("🔬 COMPREHENSIVE SHAP ANALYSIS REPORT")
        print("="*50)
        
        # Create explainer and calculate SHAP values
        try:
            self.create_shap_explainer()
            
            if self.shap_values is not None:
                # Create various SHAP visualizations
                self.summary_plot()
                self.analyze_individual_predictions(num_cases=8)
                self.dependency_analysis(top_features=6)
                comparison_df = self.compare_theft_vs_normal_shap()
                
                print("\n✅ SHAP analysis completed!")
                print("Generated files:")
                print("  - shap_summary.png (Overall feature importance)")
                print("  - shap_feature_importance.png (Mean absolute impact)")
                print("  - shap_force_plot_case_*.png (Individual explanations)")
                print("  - shap_dependency_*.png (Feature dependency plots)")
                print("  - shap_theft_vs_normal.png (Theft vs normal comparison)")
                
                return {
                    'explainer': self.explainer,
                    'shap_values': self.shap_values,
                    'comparison_df': comparison_df
                }
            else:
                print("❌ SHAP analysis failed - no SHAP values generated")
                return None
                
        except Exception as e:
            print(f"❌ Error in SHAP analysis: {e}")
            return None

def main(model, X_scaled, feature_columns, df_processed):
    shap_analyzer = SHAPAnalyzer(model, X_scaled, feature_columns, df_processed)
    shap_results = shap_analyzer.generate_shap_report()
    
    return shap_results

if __name__ == "__main__":
    # This would be called from main pipeline
    pass