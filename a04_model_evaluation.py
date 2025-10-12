import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import config

class ModelEvaluator:
    def __init__(self, df, theft_labels, predictions, probabilities):
        self.df = df
        self.theft_labels = theft_labels
        self.predictions = predictions
        self.probabilities = probabilities
        self.evaluation_results = {}
    
    def comprehensive_metrics(self):
        """Calculate comprehensive evaluation metrics"""
        from sklearn.metrics import (
            precision_score, recall_score, f1_score, 
            accuracy_score, confusion_matrix, classification_report
        )
        
        print("📈 Comprehensive Model Evaluation")
        print("="*50)
        
        # Basic metrics
        accuracy = accuracy_score(self.theft_labels, self.predictions)
        precision = precision_score(self.theft_labels, self.predictions, zero_division=0)
        recall = recall_score(self.theft_labels, self.predictions, zero_division=0)
        f1 = f1_score(self.theft_labels, self.predictions, zero_division=0)
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Store results
        self.evaluation_results['basic_metrics'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        return self.evaluation_results['basic_metrics']
    
    def plot_confusion_matrix(self):
        """Plot detailed confusion matrix"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(self.theft_labels, self.predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Predicted Normal', 'Predicted Theft'],
                   yticklabels=['Actual Normal', 'Actual Theft'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return cm
    
    def plot_roc_curve(self):
        """Plot ROC curve"""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, thresholds = roc_curve(self.theft_labels, self.probabilities)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.evaluation_results['auc_score'] = roc_auc
        return roc_auc
    
    def plot_precision_recall_curve(self):
        """Plot Precision-Recall curve"""
        from sklearn.metrics import precision_recall_curve
        
        precision, recall, thresholds = precision_recall_curve(self.theft_labels, self.probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return precision, recall, thresholds
    
    def analyze_false_positives_negatives(self):
        """Analyze false positives and false negatives"""
        # Identify false positives (predicted theft but actual normal)
        false_positives = self.df[(self.predictions == 1) & (self.theft_labels == 0)]
        
        # Identify false negatives (predicted normal but actual theft)
        false_negatives = self.df[(self.predictions == 0) & (self.theft_labels == 1)]
        
        # Identify true positives (correctly detected theft)
        true_positives = self.df[(self.predictions == 1) & (self.theft_labels == 1)]
        
        print(f"\n🔍 Error Analysis:")
        print(f"False Positives: {len(false_positives)}")
        print(f"False Negatives: {len(false_negatives)}")
        print(f"True Positives: {len(true_positives)}")
        
        # Analyze characteristics of false positives
        if len(false_positives) > 0:
            print(f"\n📊 False Positives Analysis:")
            fp_summary = false_positives[config.FEATURE_COLUMNS].describe()
            print("Feature statistics for false positives:")
            print(fp_summary.loc[['mean', 'std']])
        
        # Analyze characteristics of false negatives
        if len(false_negatives) > 0:
            print(f"\n📊 False Negatives Analysis:")
            fn_summary = false_negatives[config.FEATURE_COLUMNS].describe()
            print("Feature statistics for false negatives:")
            print(fn_summary.loc[['mean', 'std']])
        
        self.evaluation_results['error_analysis'] = {
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_positives': true_positives
        }
        
        return false_positives, false_negatives
    
    def threshold_analysis(self):
        """Analyze performance at different probability thresholds"""
        thresholds = np.arange(0.1, 1.0, 0.1)
        threshold_results = []
        
        for threshold in thresholds:
            # Apply threshold
            threshold_predictions = (self.probabilities >= threshold).astype(int)
            
            # Calculate metrics
            accuracy = (threshold_predictions == self.theft_labels).mean()
            
            # Handle division by zero
            precision_numerator = ((threshold_predictions == 1) & (self.theft_labels == 1)).sum()
            precision_denominator = max(1, (threshold_predictions == 1).sum())
            precision = precision_numerator / precision_denominator
            
            recall_numerator = ((threshold_predictions == 1) & (self.theft_labels == 1)).sum()
            recall_denominator = max(1, self.theft_labels.sum())
            recall = recall_numerator / recall_denominator
            
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            threshold_results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
        
        threshold_df = pd.DataFrame(threshold_results)
        
        # Plot threshold analysis
        plt.figure(figsize=(10, 6))
        plt.plot(threshold_df['threshold'], threshold_df['accuracy'], label='Accuracy', marker='o')
        plt.plot(threshold_df['threshold'], threshold_df['precision'], label='Precision', marker='s')
        plt.plot(threshold_df['threshold'], threshold_df['recall'], label='Recall', marker='^')
        plt.plot(threshold_df['threshold'], threshold_df['f1_score'], label='F1-Score', marker='d')
        plt.xlabel('Probability Threshold')
        plt.ylabel('Score')
        plt.title('Model Performance vs Probability Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Find optimal threshold (maximizing F1-score)
        optimal_idx = threshold_df['f1_score'].idxmax()
        optimal_threshold = threshold_df.loc[optimal_idx, 'threshold']
        
        print(f"\n🎯 Optimal Threshold Analysis:")
        print(f"Recommended threshold: {optimal_threshold:.2f}")
        print(threshold_df.loc[optimal_idx])
        
        self.evaluation_results['threshold_analysis'] = threshold_df
        self.evaluation_results['optimal_threshold'] = optimal_threshold
        
        return threshold_df, optimal_threshold
    
    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        print("\n" + "="*60)
        print("📋 COMPREHENSIVE MODEL EVALUATION REPORT")
        print("="*60)
        
        # Basic metrics
        basic_metrics = self.comprehensive_metrics()
        
        # ROC and PR curves
        auc_score = self.plot_roc_curve()
        self.plot_precision_recall_curve()
        
        # Confusion matrix
        self.plot_confusion_matrix()
        
        # Error analysis
        false_positives, false_negatives = self.analyze_false_positives_negatives()
        
        # Threshold analysis
        threshold_df, optimal_threshold = self.threshold_analysis()
        
        print(f"\n📈 Summary:")
        print(f"AUC Score: {auc_score:.4f}")
        print(f"F1-Score: {basic_metrics['f1_score']:.4f}")
        print(f"Detection Rate: {basic_metrics['recall']:.4f}")
        print(f"False Positive Rate: {len(false_positives)/len(self.df):.4f}")
        print(f"Recommended Threshold: {optimal_threshold:.2f}")
        
        return self.evaluation_results

def main(df_processed, predictions, probabilities):
    evaluator = ModelEvaluator(df_processed, df_processed['Theft'], predictions, probabilities)
    evaluation_results = evaluator.generate_evaluation_report()
    
    return evaluation_results

if __name__ == "__main__":
    # This would be called from main pipeline
    pass