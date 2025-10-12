"""
MAIN EXECUTION FILE - ELECTRICITY THEFT DETECTION SYSTEM
Complete pipeline from data analysis to final results
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime

# Import all modules
from a01_data_analysis import main as data_analysis_main
from a02_feature_engineering import main as feature_engineering_main
from a03_isolation_forest_model import main as model_training_main
from a04_model_evaluation import main as model_evaluation_main
from a05_shap_analysis import main as shap_analysis_main
from a06_business_rules import main as business_rules_main

def main():
    print("="*70)
    print("🔌 ELECTRICITY THEFT DETECTION SYSTEM")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        # Step 1: Data Analysis
        print("\n" + "="*50)
        print("📊 STEP 1: DATA ANALYSIS")
        print("="*50)
        df, data_analysis_results = data_analysis_main()
        
        # Step 2: Feature Engineering
        print("\n" + "="*50)
        print("🔧 STEP 2: FEATURE ENGINEERING")
        print("="*50)
        df_processed, X_scaled, feature_columns, scaler = feature_engineering_main(df)
        
        # Step 3: Model Training
        print("\n" + "="*50)
        print("🤖 STEP 3: MODEL TRAINING")
        print("="*50)
        df_with_predictions, theft_model, model_performance = model_training_main(
            df_processed, X_scaled, feature_columns, scaler
        )
        
        # Step 4: Model Evaluation
        print("\n" + "="*50)
        print("📈 STEP 4: MODEL EVALUATION")
        print("="*50)
        evaluation_results = model_evaluation_main(
            df_with_predictions, 
            df_with_predictions['anomaly_prediction'], 
            df_with_predictions['anomaly_probability']
        )
        
        # Step 5: SHAP Analysis
        print("\n" + "="*50)
        print("🔬 STEP 5: SHAP ANALYSIS")
        print("="*50)
        shap_results = shap_analysis_main(
            theft_model.model, X_scaled, feature_columns, df_with_predictions
        )
        
        # Step 6: Business Rules Integration
        print("\n" + "="*50)
        print("🏢 STEP 6: BUSINESS RULES INTEGRATION")
        print("="*50)
        df_final, business_results = business_rules_main(df_with_predictions)
        
        # Calculate total execution time
        end_time = time.time()
        total_time = end_time - start_time
        
        # Final Summary
        print("\n" + "="*70)
        print("🎉 SYSTEM EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Execution Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
        # Key Results Summary
        print("\n📋 KEY RESULTS SUMMARY:")
        print(f"• Dataset Size: {len(df)} records")
        print(f"• Actual Theft Cases: {df['Theft'].sum()} ({df['Theft'].mean()*100:.2f}%)")
        print(f"• Features Engineered: {len(feature_columns)}")
        
        if 'basic_metrics' in evaluation_results:
            metrics = evaluation_results['basic_metrics']
            print(f"• Model Accuracy: {metrics['accuracy']:.4f}")
            print(f"• Model Precision: {metrics['precision']:.4f}")
            print(f"• Model Recall: {metrics['recall']:.4f}")
            print(f"• Model F1-Score: {metrics['f1_score']:.4f}")
        
        if 'hybrid_performance' in business_results:
            hybrid = business_results['hybrid_performance']
            print(f"• Hybrid System Accuracy: {hybrid['accuracy']:.4f}")
            print(f"• Hybrid System Recall: {hybrid['recall']:.4f}")
        
        print("\n💾 Output Files Generated:")
        print("  - Data analysis plots (correlation_matrix.png, feature_distributions.png, etc.)")
        print("  - Model evaluation plots (confusion_matrix.png, roc_curve.png, etc.)")
        print("  - SHAP analysis plots (shap_summary.png, shap_force_plot_*.png, etc.)")
        print("  - Business rules analysis (business_rule_effectiveness.png)")
        print("  - Model files (isolation_forest_model.pkl, feature_scaler.pkl)")
        print("  - Final results (theft_detection_results.csv)")
        
        print("\n✅ Pipeline completed successfully!")
        
        return {
            'df_final': df_final,
            'data_analysis': data_analysis_results,
            'model_performance': model_performance,
            'evaluation': evaluation_results,
            'shap_analysis': shap_results,
            'business_rules': business_results
        }
        
    except Exception as e:
        print(f"\n❌ ERROR in pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()