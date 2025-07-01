import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class WasteClassificationSystem:
    def __init__(self):
        self.binary_model = None  # Will expire unsold prediction
        self.action_model = None  # Action recommendation
        self.calibrated_model = None  # For confidence scores
        self.scaler = StandardScaler()
        self.feature_names = None
        self.model_performance = {}
        
    def train_binary_classifier(self, X, y_binary, test_size=0.2):
        """
        Train binary classifier to predict if item will expire unsold
        """
        print("Training binary waste prediction model...")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=test_size, random_state=42, stratify=y_binary
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        # Try multiple algorithms
        models = {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        best_score = 0
        best_model = None
        best_name = None
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train_balanced, y_train_balanced, 
                cv=5, scoring='roc_auc'
            )
            
            print(f"{name} CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            if cv_scores.mean() > best_score:
                best_score = cv_scores.mean()
                best_model = model
                best_name = name
        
        # Train best model
        print(f"\nSelected best model: {best_name}")
        self.binary_model = best_model
        self.binary_model.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate
        y_pred = self.binary_model.predict(X_test_scaled)
        y_pred_proba = self.binary_model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\nBinary Classification Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Create calibrated model for confidence scores
        self.calibrated_model = CalibratedClassifierCV(
            self.binary_model, method='sigmoid', cv=3
        )
        self.calibrated_model.fit(X_train_balanced, y_train_balanced)
        
        # Store performance metrics
        self.model_performance = {
            'model_name': best_name,
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'cv_score': best_score,
            'training_samples': len(X_train_balanced),
            'test_samples': len(X_test)
        }
        
        return {
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'model_name': best_name,
            'cv_score': best_score
        }
    
    def train_action_classifier(self, X, y_multiclass, test_size=0.2):
        """
        Train multi-class classifier for action recommendations
        """
        print("\nTraining action recommendation model...")
        
        # Filter out 'keep' class for more focused training
        mask = y_multiclass != 'keep'
        X_filtered = X[mask]
        y_filtered = y_multiclass[mask]
        
        if len(X_filtered) == 0:
            print("No action data available for training")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_filtered, test_size=test_size, random_state=42, 
            stratify=y_filtered
        )
        
        # Scale features
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost for multi-class
        self.action_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        self.action_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.action_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Action Classification Accuracy: {accuracy:.4f}")
        print(f"\nAction Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'model_name': 'XGBoost',
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def predict_waste_probability(self, X):
        """
        Predict waste probability for given features
        """
        if self.binary_model is None:
            raise ValueError("Binary model not trained. Call train_binary_classifier first.")
        
        X_scaled = self.scaler.transform(X)
        
        # Get probability from calibrated model if available
        if self.calibrated_model is not None:
            probabilities = self.calibrated_model.predict_proba(X_scaled)[:, 1]
        else:
            probabilities = self.binary_model.predict_proba(X_scaled)[:, 1]
        
        return probabilities
    
    def predict_action(self, X):
        """
        Predict recommended action for given features
        """
        if self.action_model is None:
            # Return default action if model not trained
            return ['keep'] * len(X)
        
        X_scaled = self.scaler.transform(X)
        actions = self.action_model.predict(X_scaled)
        
        return actions
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model
        """
        if self.binary_model is None:
            return None
        
        if hasattr(self.binary_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.binary_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return None
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model on test data
        """
        if self.binary_model is None:
            raise ValueError("Model not trained")
        
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions
        y_pred = self.binary_model.predict(X_test_scaled)
        y_pred_proba = self.binary_model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        return {
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        import joblib
        
        model_data = {
            'binary_model': self.binary_model,
            'action_model': self.action_model,
            'calibrated_model': self.calibrated_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_performance': self.model_performance
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model
        """
        import joblib
        
        model_data = joblib.load(filepath)
        
        self.binary_model = model_data['binary_model']
        self.action_model = model_data['action_model']
        self.calibrated_model = model_data['calibrated_model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_performance = model_data['model_performance']
        
        print(f"Model loaded from {filepath}")