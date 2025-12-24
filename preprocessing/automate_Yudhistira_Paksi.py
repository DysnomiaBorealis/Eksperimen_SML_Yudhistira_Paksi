import pandas as pd
import numpy as np
import re
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


def load_dataset(file_path='indo_spam_raw.csv'):
    """
    Step 1: Load the raw dataset
    
    Args:
        file_path (str): Path to the raw dataset
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print(f"[1/7] Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"      Dataset loaded! Shape: {df.shape}")
    print(f"      Columns: {list(df.columns)}")
    return df


def handle_missing_values(df):
    """
    Step 2: Handle missing values
    
    Args:
        df (pd.DataFrame): Raw dataset
    
    Returns:
        pd.DataFrame: Dataset after removing missing values
    """
    print("\n[2/7] Handling missing values...")
    initial_rows = len(df)
    df = df.dropna()
    removed = initial_rows - len(df)
    print(f"      Removed {removed} rows with missing values")
    print(f"      Remaining: {len(df)} rows")
    return df


def remove_duplicates(df):
    """
    Step 3: Remove duplicate data
    
    Args:
        df (pd.DataFrame): Dataset
    
    Returns:
        pd.DataFrame: Dataset after removing duplicates
    """
    print("\n[3/7] Removing duplicate data...")
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    print(f"      Removed {removed} duplicate rows")
    print(f"      Remaining: {len(df)} rows")
    return df


def clean_text(text):
    """
    Text cleaning function for preprocessing
    Performs: lowercase, remove URLs/emails/phone numbers, remove special chars
    
    Args:
        text (str): Raw text to be cleaned
    
    Returns:
        str: Cleaned text
    """
    # Convert ke lowercase
    text = str(text).lower()
    
    # Hapus URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Hapus email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Hapus phone numbers
    text = re.sub(r'\d{5,}', '', text)
    
    # Hapus special characters dan digits
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Hapus whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def perform_text_cleaning_and_encoding(df):
    """
    Step 4: Perform text cleaning and categorical encoding
    
    Args:
        df (pd.DataFrame): Dataset
    
    Returns:
        pd.DataFrame: Dataset with cleaned text and encoded labels
    """
    print("\n[4/7] Performing text cleaning and encoding...")
    
    # Apply text cleaning
    df['cleaned_text'] = df['Pesan'].apply(clean_text)
    print(f"      Text cleaning completed")
    
    # Encode target variable
    df['label'] = df['Kategori'].map({'Spam': 1, 'ham': 0})
    print(f"      Label encoding completed (Spam=1, ham=0)")
    
    return df


def split_data(df, test_size=0.2, random_state=42):
    """
    Step 5: Data binning - Split data into training and testing sets
    
    Args:
        df (pd.DataFrame): Preprocessed dataset
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("\n[5/7] Splitting data into train and test sets...")
    
    X = df['cleaned_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"      Training set: {len(X_train)} samples")
    print(f"      Testing set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test


def vectorize_text(X_train, X_test, max_features=3000, min_df=2, max_df=0.8):
    """
    Step 6: TF-IDF Vectorization
    
    Args:
        X_train: Training text data
        X_test: Testing text data
        max_features (int): Maximum number of features
        min_df (int): Minimum document frequency
        max_df (float): Maximum document frequency
    
    Returns:
        tuple: (X_train_tfidf, X_test_tfidf, vectorizer)
    """
    print("\n[6/7] Performing TF-IDF vectorization...")
    
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df, max_df=max_df)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"      TF-IDF matrix shape (train): {X_train_tfidf.shape}")
    print(f"      TF-IDF matrix shape (test): {X_test_tfidf.shape}")
    print(f"      Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    return X_train_tfidf, X_test_tfidf, vectorizer


def save_preprocessed_data(df, vectorizer=None, output_path='preprocessing/indo_spam_preprocessing.csv'):
    """
    Step 7: Save preprocessed data to CSV and vectorizer to joblib
    
    Args:
        df (pd.DataFrame): Preprocessed dataset
        vectorizer: Fitted TF-IDF vectorizer (optional)
        output_path (str): Path to save the preprocessed data
    """
    print(f"\n[7/7] Saving preprocessed data and vectorizer...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Select columns to save
    columns_to_save = ['Kategori', 'Pesan', 'cleaned_text', 'label']
    df_to_save = df[columns_to_save].copy()
    
    # Save to CSV
    df_to_save.to_csv(output_path, index=False)
    print(f"      Preprocessed data saved to: {output_path}")
    
    # Save vectorizer if provided
    if vectorizer is not None:
        vectorizer_path = os.path.join(os.path.dirname(output_path), 'vectorizer.joblib')
        joblib.dump(vectorizer, vectorizer_path)
        print(f"      Vectorizer saved to: {vectorizer_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    print(f"Total records: {len(df_to_save)}")
    print(f"Spam messages: {df_to_save['label'].sum()}")
    print(f"Ham messages: {len(df_to_save) - df_to_save['label'].sum()}")
    print(f"Spam ratio: {df_to_save['label'].mean():.2%}")
    print("="*60)


def preprocess_and_prepare_data(raw_data_path='indo_spam_raw.csv', 
                                  output_path='preprocessing/indo_spam_preprocessing.csv'):
    """
    Main preprocessing pipeline that returns training-ready data
    
    This function converts the experiment notebook steps into an automated pipeline
    with the same preprocessing stages but in a functional structure.
    
    Args:
        raw_data_path (str): Path to raw dataset
        output_path (str): Path to save preprocessed data
    
    Returns:
        dict: Dictionary containing:
            - X_train_tfidf: Training features (vectorized)
            - X_test_tfidf: Testing features (vectorized)
            - y_train: Training labels
            - y_test: Testing labels
            - vectorizer: Fitted TF-IDF vectorizer
            - df_processed: Preprocessed dataframe
    """
    print("="*60)
    print("INDONESIAN SPAM DETECTION - AUTOMATED PREPROCESSING")
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Step 1: Load dataset
    df = load_dataset(raw_data_path)
    
    # Step 2: Handle missing values
    df = handle_missing_values(df)
    
    # Step 3: Remove duplicates
    df = remove_duplicates(df)
    
    # Step 4: Text cleaning and encoding
    df = perform_text_cleaning_and_encoding(df)
    
    # Step 5: Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Step 6: Vectorization
    X_train_tfidf, X_test_tfidf, vectorizer = vectorize_text(X_train, X_test)
    
    # Step 7: Save preprocessed data and vectorizer
    save_preprocessed_data(df, vectorizer, output_path)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nReturning training-ready data:")
    print("  - X_train_tfidf: Vectorized training features")
    print("  - X_test_tfidf: Vectorized testing features")
    print("  - y_train: Training labels")
    print("  - y_test: Testing labels")
    print("  - vectorizer: Fitted TF-IDF vectorizer")
    print("  - df_processed: Preprocessed dataframe")
    print("="*60)
    
    return {
        'X_train_tfidf': X_train_tfidf,
        'X_test_tfidf': X_test_tfidf,
        'y_train': y_train,
        'y_test': y_test,
        'vectorizer': vectorizer,
        'df_processed': df
    }


def main():
    """
    Main function to run the preprocessing pipeline
    """
    try:
        # Run preprocessing and get training-ready data
        result = preprocess_and_prepare_data()
        
        # The returned data is ready for model training
        # Example usage:
        # from sklearn.naive_bayes import MultinomialNB
        # model = MultinomialNB()
        # model.fit(result['X_train_tfidf'], result['y_train'])
        
        print("\nData is now ready for model training!")
        
    except FileNotFoundError as e:
        print(f"\nError: File not found - {e}")
        print("Please ensure 'indo_spam.csv' exists in the current directory.")
        
    except Exception as e:
        print(f"\nError occurred during preprocessing: {e}")
        raise


if __name__ == "__main__":
    main()

