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
        
    def train_binary_classifier(self, X, y_binary, test_size=0.2):
        """
        Train binary classifier to predict if item will expire unsold
        """
        print("Training binary waste prediction model...")
        
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
        
        print(f"\nBinary Classification Results:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Create calibrated model for confidence scores
        self.calibrated_model = CalibratedClassifierCV(
            self.binary_model, method='sigmoid', cv=3
        )
        self.calibrated_model.fit(X_train_balanced, y_train_balanced)
        
        self.feature_names = X.columns.tolist()
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'model_name': best_name
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