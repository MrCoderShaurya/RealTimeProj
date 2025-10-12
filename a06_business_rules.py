import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config

class BusinessRulesEngine:
    def __init__(self, df):
        self.df = df
        self.business_rules_results = {}
        
    def apply_business_rules(self):
        """Apply business rules for theft detection"""
        print("🏢 Applying business rules...")
        
        try:
            # Rule 1: Suspiciously low consumption compared to average
            self.df['rule_low_consumption'] = (
                self.df['usage_vs_average_ratio'] < config.BUSINESS_RULES['low_consumption_ratio']
            ).astype(int)
            
            # Rule 2: High bill payment delay
            self.df['rule_high_delay'] = (
                self.df['BillPaymentDelay (days)'] > config.BUSINESS_RULES['high_bill_delay']
            ).astype(int)
            
            # Rule 3: High voltage fluctuations
            self.df['rule_voltage_issues'] = (
                self.df['VoltageFluctuations'] >= config.BUSINESS_RULES['suspicious_voltage_fluctuation']
            ).astype(int)
            
            # Rule 4: High-risk industrial area with previous theft history
            self.df['rule_high_risk_area'] = (
                (self.df['IndustrialAreaNearby'] == 1) & 
                (self.df['PreviousTheftHistory'] == 1)
            ).astype(int)
            
            # Rule 5: Unusual usage patterns
            self.df['rule_unusual_patterns'] = (
                (self.df['UnusualUsageSpike'] == 1) |
                (self.df['extremely_low_usage'] == 1) |
                (self.df['low_usage_high_residents'] == 1)
            ).astype(int)
            
            # Calculate business rule score (sum of triggered rules)
            rule_columns = ['rule_low_consumption', 'rule_high_delay', 'rule_voltage_issues', 
                           'rule_high_risk_area', 'rule_unusual_patterns']
            
            self.df['business_rule_score'] = self.df[rule_columns].sum(axis=1)
            self.df['business_rule_alert'] = (self.df['business_rule_score'] >= 2).astype(int)
            
            print("✅ Business rules applied successfully!")
            
        except Exception as e:
            print(f"❌ Error applying business rules: {e}")
            # Create default columns if business rules fail
            self.df['business_rule_score'] = 0
            self.df['business_rule_alert'] = 0
            rule_columns = []
            
        return self.df, rule_columns
    
    def analyze_rule_effectiveness(self, rule_columns):
        """Analyze effectiveness of each business rule"""
        print("\n📊 Business Rule Effectiveness Analysis:")
        
        rule_effectiveness = {}
        
        for rule in rule_columns:
            try:
                # Rule triggered cases
                rule_triggered = self.df[self.df[rule] == 1]
                total_triggered = len(rule_triggered)
                
                if total_triggered > 0:
                    # Theft cases among triggered
                    theft_triggered = rule_triggered['Theft'].sum()
                    theft_rate = theft_triggered / total_triggered
                    
                    # Coverage of total theft cases
                    total_theft = self.df['Theft'].sum()
                    coverage = theft_triggered / total_theft if total_theft > 0 else 0
                    
                    rule_effectiveness[rule] = {
                        'total_triggered': total_triggered,
                        'theft_triggered': theft_triggered,
                        'theft_rate': theft_rate,
                        'coverage': coverage
                    }
                    
                    print(f"\n{rule}:")
                    print(f"  Triggered: {total_triggered} cases")
                    print(f"  Theft cases: {theft_triggered}")
                    print(f"  Theft rate: {theft_rate:.4f}")
                    print(f"  Coverage: {coverage:.4f}")
            except:
                print(f"❌ Error analyzing rule {rule}")
        
        self.business_rules_results['rule_effectiveness'] = rule_effectiveness
        return rule_effectiveness
    
    def create_hybrid_system(self):
        """Combine ML model with business rules"""
        print("\n🔄 Creating hybrid detection system...")
        
        try:
            # Combine ML probability with business rules
            self.df['hybrid_score'] = (
                self.df['anomaly_probability'] * 0.7 +  # ML model weight
                (self.df['business_rule_score'] / 5) * 0.3  # Business rules weight
            )
            
            # Apply hybrid threshold
            self.df['hybrid_alert'] = (self.df['hybrid_score'] > 0.5).astype(int)
            
            # Evaluate hybrid system
            hybrid_accuracy = (self.df['hybrid_alert'] == self.df['Theft']).mean()
            hybrid_precision = ((self.df['hybrid_alert'] == 1) & (self.df['Theft'] == 1)).sum() / max(1, self.df['hybrid_alert'].sum())
            hybrid_recall = ((self.df['hybrid_alert'] == 1) & (self.df['Theft'] == 1)).sum() / max(1, self.df['Theft'].sum())
            
            print(f"Hybrid System Performance:")
            print(f"  Accuracy:  {hybrid_accuracy:.4f}")
            print(f"  Precision: {hybrid_precision:.4f}")
            print(f"  Recall:    {hybrid_recall:.4f}")
            
            self.business_rules_results['hybrid_performance'] = {
                'accuracy': hybrid_accuracy,
                'precision': hybrid_precision,
                'recall': hybrid_recall
            }
        except Exception as e:
            print(f"❌ Error creating hybrid system: {e}")
            self.df['hybrid_score'] = self.df['anomaly_probability']
            self.df['hybrid_alert'] = self.df['anomaly_prediction']
            
        return self.df
    
    def create_business_insights(self):
        """Generate business insights from the rules"""
        print("\n💡 Generating business insights...")
        
        insights = []
        
        try:
            # Insight 1: Most common rule combinations
            rule_columns = [col for col in self.df.columns if col.startswith('rule_')]
            if rule_columns:
                rule_combinations = self.df[rule_columns].apply(tuple, axis=1).value_counts().head(5)
                
                print("\n📊 Most Common Rule Combinations:")
                for combination, count in rule_combinations.items():
                    theft_rate = self.df[self.df[rule_columns].apply(tuple, axis=1) == combination]['Theft'].mean()
                    print(f"  {combination}: {count} cases, theft rate: {theft_rate:.4f}")
                    insights.append(f"Rule combination {combination} appears in {count} cases with {theft_rate:.1%} theft rate")
            
            # Insight 2: High-risk segments
            high_risk_segment = self.df[
                (self.df['business_rule_score'] >= 3) & 
                (self.df['anomaly_probability'] > 0.7)
            ]
            
            if len(high_risk_segment) > 0:
                segment_theft_rate = high_risk_segment['Theft'].mean()
                print(f"\n🎯 High-Risk Segment:")
                print(f"  Size: {len(high_risk_segment)} households")
                print(f"  Theft rate: {segment_theft_rate:.4f}")
                insights.append(f"High-risk segment ({len(high_risk_segment)} households) has {segment_theft_rate:.1%} theft rate")
            
            # Insight 3: Rule performance by area type
            industrial_theft_rate = self.df[self.df['IndustrialAreaNearby'] == 1]['Theft'].mean()
            non_industrial_theft_rate = self.df[self.df['IndustrialAreaNearby'] == 0]['Theft'].mean()
            
            print(f"\n🏭 Theft Rates by Area Type:")
            print(f"  Industrial areas: {industrial_theft_rate:.4f}")
            print(f"  Non-industrial areas: {non_industrial_theft_rate:.4f}")
            insights.append(f"Theft rate in industrial areas ({industrial_theft_rate:.1%}) vs non-industrial ({non_industrial_theft_rate:.1%})")
        except Exception as e:
            print(f"❌ Error generating business insights: {e}")
        
        self.business_rules_results['insights'] = insights
        return insights
    
    def generate_business_report(self):
        """Generate comprehensive business rules report"""
        print("\n" + "="*50)
        print("🏢 BUSINESS RULES ANALYSIS REPORT")
        print("="*50)
        
        # Apply business rules
        df_with_rules, rule_columns = self.apply_business_rules()
        
        # Analyze rule effectiveness
        rule_effectiveness = self.analyze_rule_effectiveness(rule_columns)
        
        # Create hybrid system
        df_hybrid = self.create_hybrid_system()
        
        # Generate insights
        insights = self.create_business_insights()
        
        # Create visualization of rule effectiveness
        if rule_effectiveness:
            self.plot_rule_effectiveness(rule_effectiveness)
        
        print("\n✅ Business rules analysis completed!")
        
        return df_hybrid, self.business_rules_results
    
    def plot_rule_effectiveness(self, rule_effectiveness):
        """Plot business rule effectiveness"""
        if not rule_effectiveness:
            return
        
        rules = list(rule_effectiveness.keys())
        theft_rates = [rule_effectiveness[rule]['theft_rate'] for rule in rules]
        coverage_rates = [rule_effectiveness[rule]['coverage'] for rule in rules]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Theft rate plot
        ax1.bar(rules, theft_rates, color='skyblue', alpha=0.7)
        ax1.set_title('Business Rule Theft Detection Rate')
        ax1.set_ylabel('Theft Rate')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Coverage plot
        ax2.bar(rules, coverage_rates, color='lightcoral', alpha=0.7)
        ax2.set_title('Business Rule Coverage of Total Theft')
        ax2.set_ylabel('Coverage Rate')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('business_rule_effectiveness.png', dpi=300, bbox_inches='tight')
        plt.close()

def main(df_processed):
    rules_engine = BusinessRulesEngine(df_processed)
    df_final, business_results = rules_engine.generate_business_report()
    
    # Save final results
    try:
        output_columns = [
            'Usage (kWh)', 'TimeOfDay', 'NumberOfResidents', 'ApplianceCount',
            'IndustrialAreaNearby', 'PreviousTheftHistory', 'AverageDailyUsage',
            'BillPaymentDelay (days)', 'UnusualUsageSpike', 'Theft',
            'anomaly_probability', 'business_rule_score', 'hybrid_score', 'hybrid_alert'
        ]
        
        # Only include columns that exist
        available_columns = [col for col in output_columns if col in df_final.columns]
        final_output = df_final[available_columns]
        final_output.to_csv(config.RESULTS_SAVE_PATH, index=False)
        
        print(f"\n💾 Final results saved to: {config.RESULTS_SAVE_PATH}")
    except Exception as e:
        print(f"❌ Error saving final results: {e}")
    
    return df_final, business_results

if __name__ == "__main__":
    # This would be called from main pipeline
    pass