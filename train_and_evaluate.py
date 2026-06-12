"""
Main training and evaluation script for Fake News Detection
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_sample_data, save_sample_data, split_data
from src.feature_extraction import FeatureExtractor
from src.models import FakeNewsDetector
from src.evaluation import ModelEvaluator


def main():
    """Main training and evaluation pipeline"""
    
    print("\n" + "=" * 70)
    print("FAKE NEWS DETECTION - TRAINING AND EVALUATION")
    print("=" * 70)
    
    # Step 1: Load sample data
    print("\n[1/5] Loading sample data...")
    df = save_sample_data('data/sample_news.csv')
    print(f"✓ Loaded {len(df)} samples")
    print(f"  - Real news: {(df['label'] == 1).sum()}")
    print(f"  - Fake news: {(df['label'] == 0).sum()}")
    
    # Step 2: Split data
    print("\n[2/5] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"✓ Training set: {len(X_train)} samples")
    print(f"✓ Testing set: {len(X_test)} samples")
    
    # Step 3: Extract features
    print("\n[3/5] Extracting TF-IDF features...")
    feature_extractor = FeatureExtractor(max_features=5000)
    X_train_features = feature_extractor.fit_transform(X_train)
    X_test_features = feature_extractor.transform(X_test)
    print(f"✓ Feature matrix shape: {X_train_features.shape}")
    
    # Step 4: Train models
    print("\n[4/5] Training machine learning models...")
    models_to_train = ['logistic', 'random_forest', 'naive_bayes']
    results = {}
    
    for model_name in models_to_train:
        print(f"\n  Training {model_name.upper()} model...")
        detector = FakeNewsDetector(model_type=model_name)
        detector.train(X_train_features, y_train)
        
        # Make predictions
        y_pred = detector.predict(X_test_features)
        y_proba = detector.predict_proba(X_test_features)
        
        # Evaluate
        metrics = ModelEvaluator.evaluate(y_test, y_pred, y_proba)
        results[model_name] = metrics
        
        print(f"  ✓ {model_name.upper()} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        # Save best model
        detector.save_model(f'models/{model_name}_detector.joblib')
    
    # Step 5: Display results
    print("\n[5/5] Displaying evaluation results...")
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        if metrics.get('roc_auc'):
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['f1'])
    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best_model[0].upper()}")
    print(f"F1-Score: {best_model[1]['f1']:.4f}")
    print("=" * 70)
    
    # Save results
    ModelEvaluator.save_results(results, 'results/all_models_results.json')
    
    # Test with sample predictions
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)
    
    sample_texts = [
        "Scientists discover new breakthrough in cancer research",
        "Celebrity reveals shocking secret about illuminati conspiracy"
    ]
    
    # Use best model for predictions
    best_detector = FakeNewsDetector(model_type=best_model[0])
    best_detector.load_model(f'models/{best_model[0]}_detector.joblib')
    
    sample_features = feature_extractor.transform(sample_texts)
    predictions = best_detector.predict(sample_features)
    probabilities = best_detector.predict_proba(sample_features)
    
    for i, text in enumerate(sample_texts):
        pred_label = "REAL NEWS" if predictions[i] == 1 else "FAKE NEWS"
        confidence = probabilities[i][predictions[i]]
        print(f"\nText: {text}")
        print(f"Prediction: {pred_label} (Confidence: {confidence:.2%})")
    
    print("\n" + "=" * 70)
    print("✓ Training and evaluation complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
