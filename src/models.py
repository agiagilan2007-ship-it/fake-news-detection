"""
Machine learning models for Fake News Detection
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import joblib
from pathlib import Path


class FakeNewsDetector:
    """Machine learning model for fake news detection"""
    
    def __init__(self, model_type='logistic'):
        """
        Initialize the detector
        
        Args:
            model_type (str): Type of model - 'logistic', 'random_forest', or 'naive_bayes'
        """
        self.model_type = model_type
        
        if model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Features to predict
            
        Returns:
            predictions: Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Args:
            X: Features to predict
            
        Returns:
            probabilities: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def save_model(self, filepath='models/detector.joblib'):
        """
        Save the trained model
        
        Args:
            filepath (str): Path to save the model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/detector.joblib'):
        """
        Load a trained model
        
        Args:
            filepath (str): Path to the model file
        """
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")
