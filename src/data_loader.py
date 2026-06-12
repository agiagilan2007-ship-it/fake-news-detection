"""
Data loading module for Fake News Detection
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


def load_sample_data():
    """
    Load or create sample news data for demonstration
    
    Returns:
        pd.DataFrame: Dataset with text and labels
    """
    # Sample fake news and real news data
    sample_data = {
        'text': [
            # Real news examples
            'Scientists discover new species of deep sea fish in Pacific Ocean',
            'Government announces new infrastructure investment plan',
            'Major tech company releases latest smartphone model',
            'University researchers publish breakthrough cancer study',
            'Olympic games conclude with closing ceremony',
            'Central bank raises interest rates amid inflation concerns',
            'International climate conference reaches agreement on emissions',
            'Hospital opens new emergency wing to serve community',
            'Stock market reaches record highs this quarter',
            'New vaccine approved by health authorities',
            
            # Fake news examples
            'Celebrities banned from social media by secret government agency',
            'Ancient alien artifacts discovered under pyramids',
            'Miracle cure for all diseases found in remote village',
            'Billionaire secretly controls world governments from space station',
            'Unknown element discovered that defies physics laws',
            'Conspiracy theory: major corporation fakes product recalls',
            'Shocking claim: historical events never actually happened',
            'False report: politician caught in impossible scandal',
            'Unproven claim: common food causes instant transformation',
            'Misinformation: vaccines contain tracking microchips',
        ],
        'label': [
            # Real news = 1
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            # Fake news = 0
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]
    }
    
    df = pd.DataFrame(sample_data)
    return df


def save_sample_data(filepath='data/sample_news.csv'):
    """
    Save sample data to CSV file
    
    Args:
        filepath (str): Path to save the data
    """
    df = load_sample_data()
    
    # Create data directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(filepath, index=False)
    print(f"Sample data saved to {filepath}")
    
    return df


def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    
    Args:
        df (pd.DataFrame): Input dataframe
        test_size (float): Proportion of test data
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df['text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test
