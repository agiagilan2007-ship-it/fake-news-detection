"""
Main entry point for Fake News Detection project
"""

from src.preprocessing import TextPreprocessor


def main():
    """Main function"""
    print("=" * 50)
    print("Fake News Detection System")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Example usage
    sample_text = "This is a sample news article for testing the preprocessing pipeline."
    
    print("\nOriginal text:")
    print(sample_text)
    
    print("\nPreprocessed tokens:")
    processed = preprocessor.preprocess(sample_text)
    print(processed)
    
    print("\n" + "=" * 50)
    print("Project initialized successfully!")
    print("Next steps:")
    print("1. Add your dataset to the data/ folder")
    print("2. Update the models and feature extraction modules")
    print("3. Train your models")
    print("4. Evaluate and test results")
    print("=" * 50)


if __name__ == "__main__":
    main()
