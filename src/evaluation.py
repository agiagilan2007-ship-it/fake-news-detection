"""
Evaluation module for model assessment
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import json
from pathlib import Path


class ModelEvaluator:
    """Evaluates model performance"""
    
    @staticmethod
    def evaluate(y_true, y_pred, y_proba=None):
        """
        Evaluate model performance
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            dict: Evaluation metrics
        """
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        if y_proba is not None:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba[:, 1]))
            except:
                metrics['roc_auc'] = None
        
        return metrics
    
    @staticmethod
    def get_classification_report(y_true, y_pred):
        """
        Get detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            str: Classification report
        """
        return classification_report(
            y_true, y_pred,
            target_names=['Fake News', 'Real News']
        )
    
    @staticmethod
    def save_results(metrics, filepath='results/evaluation_results.json'):
        """
        Save evaluation results to file
        
        Args:
            metrics (dict): Evaluation metrics
            filepath (str): Path to save results
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Results saved to {filepath}")
    
    @staticmethod
    def print_results(metrics, report=None):
        """
        Print evaluation results
        
        Args:
            metrics (dict): Evaluation metrics
            report (str): Classification report
        """
        print("\n" + "=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)
        
        print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        
        if metrics.get('roc_auc'):
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        print("\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"  True Negatives:  {cm[0][0]}")
        print(f"  False Positives: {cm[0][1]}")
        print(f"  False Negatives: {cm[1][0]}")
        print(f"  True Positives:  {cm[1][1]}")
        
        if report:
            print("\n" + report)
        
        print("=" * 60)
