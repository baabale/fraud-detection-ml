#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script creates a balanced sample dataset from the main financial fraud detection dataset.
It analyzes the main dataset and extracts a representative sample with different fraud classes.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define file paths
main_dataset_path = Path("/Users/baabale/Documents/MasterStudies/thesis/data/raw/financial_fraud_detection_dataset.csv")
sample_dataset_path = Path("/Users/baabale/Documents/MasterStudies/thesis/data/raw/sample_dataset.csv")

def analyze_dataset(df):
    """Analyze the dataset and print key statistics"""
    print(f"Total records: {len(df)}")
    
    # Fraud distribution
    fraud_count = df['is_fraud'].value_counts()
    print("\nFraud distribution:")
    print(fraud_count)
    print(f"Fraud percentage: {fraud_count.get(True, 0) / len(df) * 100:.2f}%")
    
    # Fraud type distribution
    if 'fraud_type' in df.columns:
        fraud_type_count = df[df['is_fraud'] == True]['fraud_type'].value_counts()
        print("\nFraud type distribution:")
        print(fraud_type_count)
    
    # Transaction type distribution
    print("\nTransaction type distribution:")
    print(df['transaction_type'].value_counts())
    
    # Merchant category distribution
    print("\nMerchant category distribution:")
    print(df['merchant_category'].value_counts())
    
    # Payment channel distribution
    print("\nPayment channel distribution:")
    print(df['payment_channel'].value_counts())
    
    # Device used distribution
    print("\nDevice used distribution:")
    print(df['device_used'].value_counts())

def create_balanced_sample(df, sample_size=10000):
    """Create a balanced sample with representative fraud and non-fraud transactions"""
    # Separate fraud and non-fraud transactions
    fraud_df = df[df['is_fraud'] == True]
    non_fraud_df = df[df['is_fraud'] == False]
    
    # Calculate how many fraud transactions to include (oversampling fraud cases)
    # Aim for around 30% fraud cases in the sample for better model training
    fraud_sample_size = int(sample_size * 0.3)
    non_fraud_sample_size = sample_size - fraud_sample_size
    
    print(f"\nCreating sample with {fraud_sample_size} fraud and {non_fraud_sample_size} non-fraud transactions")
    
    # If we don't have enough fraud cases, take all available
    if len(fraud_df) < fraud_sample_size:
        fraud_sample = fraud_df
        print(f"Warning: Only {len(fraud_df)} fraud transactions available")
    else:
        # Sample fraud transactions, ensuring we get a mix of different fraud types if possible
        if 'fraud_type' in df.columns and not fraud_df['fraud_type'].isna().all():
            # Stratified sampling by fraud type
            fraud_types = fraud_df['fraud_type'].dropna().unique()
            if len(fraud_types) > 0:
                fraud_sample = pd.DataFrame()
                
                # Calculate samples per fraud type
                samples_per_type = fraud_sample_size // len(fraud_types)
                remaining = fraud_sample_size % len(fraud_types)
                
                for fraud_type in fraud_types:
                    type_df = fraud_df[fraud_df['fraud_type'] == fraud_type]
                    # Adjust sample size if we don't have enough of this type
                    type_sample_size = min(samples_per_type, len(type_df))
                    type_sample = type_df.sample(type_sample_size)
                    fraud_sample = pd.concat([fraud_sample, type_sample])
                
                # Add remaining samples from any fraud type
                if remaining > 0 and len(fraud_sample) < fraud_sample_size:
                    remaining_sample = fraud_df[~fraud_df.index.isin(fraud_sample.index)].sample(
                        min(remaining, fraud_sample_size - len(fraud_sample))
                    )
                    fraud_sample = pd.concat([fraud_sample, remaining_sample])
            else:
                # Simple random sampling if no fraud type information
                fraud_sample = fraud_df.sample(fraud_sample_size)
        else:
            # Simple random sampling if no fraud type information
            fraud_sample = fraud_df.sample(fraud_sample_size)
    
    # Sample non-fraud transactions, ensuring diversity in transaction types
    # Stratified sampling by transaction type
    non_fraud_sample = pd.DataFrame()
    transaction_types = non_fraud_df['transaction_type'].unique()
    
    # Calculate samples per transaction type
    samples_per_type = non_fraud_sample_size // len(transaction_types)
    remaining = non_fraud_sample_size % len(transaction_types)
    
    for tx_type in transaction_types:
        type_df = non_fraud_df[non_fraud_df['transaction_type'] == tx_type]
        # Adjust sample size if we don't have enough of this type
        type_sample_size = min(samples_per_type, len(type_df))
        type_sample = type_df.sample(type_sample_size)
        non_fraud_sample = pd.concat([non_fraud_sample, type_sample])
    
    # Add remaining samples from any transaction type
    if remaining > 0 and len(non_fraud_sample) < non_fraud_sample_size:
        remaining_sample = non_fraud_df[~non_fraud_df.index.isin(non_fraud_sample.index)].sample(
            min(remaining, non_fraud_sample_size - len(non_fraud_sample))
        )
        non_fraud_sample = pd.concat([non_fraud_sample, remaining_sample])
    
    # Combine fraud and non-fraud samples
    balanced_sample = pd.concat([fraud_sample, non_fraud_sample])
    
    # Shuffle the sample
    balanced_sample = balanced_sample.sample(frac=1).reset_index(drop=True)
    
    return balanced_sample

def main():
    print("Loading main dataset...")
    # Read the dataset in chunks to handle large file
    chunk_size = 500000
    chunks = []
    
    for chunk in pd.read_csv(main_dataset_path, chunksize=chunk_size):
        chunks.append(chunk)
        print(f"Loaded chunk with {len(chunk)} records")
    
    # Combine all chunks
    df = pd.concat(chunks)
    print(f"Total records loaded: {len(df)}")
    
    # Analyze the dataset
    print("\n=== Dataset Analysis ===")
    analyze_dataset(df)
    
    # Create a balanced sample
    print("\n=== Creating Balanced Sample ===")
    sample_df = create_balanced_sample(df, sample_size=10000)
    
    # Analyze the sample
    print("\n=== Sample Dataset Analysis ===")
    analyze_dataset(sample_df)
    
    # Save the sample
    sample_df.to_csv(sample_dataset_path, index=False)
    print(f"\nSample dataset saved to {sample_dataset_path}")

if __name__ == "__main__":
    main()
