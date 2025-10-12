import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import joblib
import config

class TheftDetectionModel:
    def __init__(self, params=config.ISOLATION_FOREST_PARAMS):
        self.params = params
        self.model = IsolationForest(**params)
        self.feature_columns = None
        self.scaler = None
        
    def train_model(self, X_scaled, feature_columns, scaler):
        """Train the Isolation Forest model"""
        print("🤖 Training Isolation Forest model...")
        
        self.feature_columns = feature_columns
        self.scaler = scaler
        
        # Fit the model
        self.model.fit(X_scaled)
        
        print("✅ Model training completed!")
        return self.model
    
    def predict_anomalies(self, X_scaled):
        """Predict anomalies and get scores"""
        print("🔮 Making predictions...")
        
        # Get anomaly predictions (-1 for anomalies, 1 for normal)
        anomaly_predictions = self.model.predict(X_scaled)
        
        # Get anomaly scores (the more negative, the more anomalous)
        anomaly_scores = self.model.decision_function(X_scaled)
        
        # Convert to binary (0 = normal, 1 = anomaly)
        binary_predictions = np.where(anomaly_predictions == -1, 1, 0)
        
        return binary_predictions, anomaly_scores
    
    def calculate_anomaly_probability(self, anomaly_scores):
        """Convert anomaly scores to probabilities (0-1 scale)"""
        # Normalize scores to 0-1 range (1 = most anomalous)
        min_score = anomaly_scores.min()
        max_score = anomaly_scores.max()
        
        if max_score == min_score:
            probabilities = np.zeros_like(anomaly_scores)
        else:
            # Invert and normalize: lower scores = higher probability of anomaly
            probabilities = 1 - (anomaly_scores - min_score) / (max_score - min_score)
        
        return probabilities
    
    def evaluate_model_performance(self, df, predictions, probabilities, theft_labels):
        """Evaluate model performance against actual theft labels"""
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        
        print("\n📊 Model Performance Evaluation:")
        print("="*50)
        
        # Basic metrics
        accuracy = (predictions == theft_labels).mean()
        print(f"Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(theft_labels, predictions, 
                                 target_names=['Normal', 'Theft']))
        
        # Confusion matrix
        cm = confusion_matrix(theft_labels, predictions)
        print("\nConfusion Matrix:")
        print(cm)
        
        # AUC-ROC if we have probabilities
        auc_score = None
        if len(np.unique(theft_labels)) > 1:
            try:
                auc_score = roc_auc_score(theft_labels, probabilities)
                print(f"\nAUC-ROC Score: {auc_score:.4f}")
            except:
                print("\nAUC-ROC Score: Could not calculate")
        
        # Calculate detection rates
        total_theft_cases = theft_labels.sum()
        detected_theft_cases = ((predictions == 1) & (theft_labels == 1)).sum()
        detection_rate = detected_theft_cases / total_theft_cases if total_theft_cases > 0 else 0
        
        false_positives = ((predictions == 1) & (theft_labels == 0)).sum()
        total_normal_cases = (theft_labels == 0).sum()
        false_positive_rate = false_positives / total_normal_cases if total_normal_cases > 0 else 0
        
        print(f"\n🎯 Theft Detection Rate: {detection_rate:.4f} ({detected_theft_cases}/{total_theft_cases})")
        print(f"🚫 False Positive Rate: {false_positive_rate:.4f} ({false_positives}/{total_normal_cases})")
        
        return {
            'accuracy': accuracy,
            'detection_rate': detection_rate,
            'false_positive_rate': false_positive_rate,
            'auc_score': auc_score
        }
    
    def save_model(self, model_path=config.MODEL_SAVE_PATH):
        """Save the trained model and scaler"""
        print(f"💾 Saving model to {model_path}...")
        
        # Save model
        joblib.dump(self.model, model_path)
        
        # Save scaler
        joblib.dump(self.scaler, config.SCALER_SAVE_PATH)
        
        print("✅ Model and scaler saved successfully!")
    
    def load_model(self, model_path=config.MODEL_SAVE_PATH):
        """Load a trained model"""
        print(f"📂 Loading model from {model_path}...")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(config.SCALER_SAVE_PATH)
        
        print("✅ Model and scaler loaded successfully!")
        return self.model

def main(df_processed, X_scaled, feature_columns, scaler):
    # Initialize and train model
    theft_model = TheftDetectionModel()
    trained_model = theft_model.train_model(X_scaled, feature_columns, scaler)
    
    # Make predictions
    predictions, anomaly_scores = theft_model.predict_anomalies(X_scaled)
    probabilities = theft_model.calculate_anomaly_probability(anomaly_scores)
    
    # Add results to dataframe
    df_processed['anomaly_prediction'] = predictions
    df_processed['anomaly_score'] = anomaly_scores
    df_processed['anomaly_probability'] = probabilities
    
    # Evaluate performance
    performance = theft_model.evaluate_model_performance(
        df_processed, predictions, probabilities, df_processed['Theft']
    )
    
    # Save model
    theft_model.save_model()
    
    return df_processed, theft_model, performance

if __name__ == "__main__":
    # This would be called from main pipeline
    pass