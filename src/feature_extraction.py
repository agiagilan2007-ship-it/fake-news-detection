"""
Feature extraction module for Fake News Detection
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocessing import TextPreprocessor


class FeatureExtractor:
    """Extracts features from text for model training"""
    
    def __init__(self, max_features=5000):
        """
        Initialize feature extractor
        
        Args:
            max_features (int): Maximum number of features to extract
        """
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.is_fitted = False
    
    def fit_transform(self, texts):
        """
        Fit vectorizer and transform texts
        
        Args:
            texts (list or pd.Series): List of text samples
            
        Returns:
            scipy.sparse matrix: TF-IDF feature matrix
        """
        # Preprocess texts
        processed_texts = [' '.join(self.preprocessor.preprocess(text)) for text in texts]
        
        # Fit and transform
        features = self.vectorizer.fit_transform(processed_texts)
        self.is_fitted = True
        
        return features
    
    def transform(self, texts):
        """
        Transform texts using fitted vectorizer
        
        Args:
            texts (list or pd.Series): List of text samples
            
        Returns:
            scipy.sparse matrix: TF-IDF feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transforming")
        
        # Preprocess texts
        processed_texts = [' '.join(self.preprocessor.preprocess(text)) for text in texts]
        
        # Transform
        features = self.vectorizer.transform(processed_texts)
        
        return features
