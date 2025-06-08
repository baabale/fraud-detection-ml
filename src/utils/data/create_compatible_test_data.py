#!/usr/bin/env python
"""
Utility script to generate test data compatible with the model monitoring script.
This creates synthetic data with the same structure as the reference data.
"""
import os
import sys
import argparse
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('create_compatible_test_data')

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def load_reference_data(reference_data_path):
    """
    Load reference data to use as a template for the test data structure.
    
    Args:
        reference_data_path (str): Path to reference data
        
    Returns:
        DataFrame: Reference data
    """
    logger.info(f"Loading reference data from {reference_data_path}")
    
    if reference_data_path.endswith('.parquet'):
        df = pd.read_parquet(reference_data_path)
    elif reference_data_path.endswith('.csv'):
        df = pd.read_csv(reference_data_path)
    else:
        raise ValueError(f"Unsupported file format: {reference_data_path}")
    
    logger.info(f"Loaded reference data with {len(df)} samples and {len(df.columns)} features")
    return df

def generate_synthetic_transaction(reference_df, transaction_id, is_fraud=False):
    """
    Generate a synthetic transaction with the same structure as the reference data.
    
    Args:
        reference_df (DataFrame): Reference data to use as a template
        transaction_id (int): Transaction ID
        is_fraud (bool): Whether to generate a fraudulent transaction
        
    Returns:
        dict: Synthetic transaction
    """
    # Get column names from reference data
    columns = reference_df.columns.tolist()
    
    # Generate timestamp
    timestamp = datetime.now() - timedelta(days=random.randint(0, 30), 
                                          hours=random.randint(0, 23), 
                                          minutes=random.randint(0, 59))
    
    # Extract unique values from reference data for categorical columns
    transaction_types = reference_df['transaction_type'].unique().tolist()
    merchant_categories = reference_df['merchant_category'].unique().tolist()
    locations = reference_df['location'].unique().tolist() if 'location' in columns else ['Unknown']
    device_used = reference_df['device_used'].unique().tolist() if 'device_used' in columns else ['Mobile', 'Desktop', 'ATM']
    payment_channels = reference_df['payment_channel'].unique().tolist() if 'payment_channel' in columns else ['Online', 'In-store', 'Mobile']
    
    # Generate base transaction
    transaction = {
        'transaction_id': f"TX{transaction_id}",
        'timestamp': timestamp.isoformat(),
        'sender_account': f"ACC{random.randint(1000000, 9999999)}",
        'receiver_account': f"ACC{random.randint(1000000, 9999999)}",
    }
    
    # Generate amount based on fraud status
    if is_fraud:
        # Fraudulent transactions tend to have unusual amounts
        amount_type = random.choice(['very_small', 'very_large', 'unusual'])
        if amount_type == 'very_small':
            amount = round(random.uniform(0.01, 1.0), 2)
        elif amount_type == 'very_large':
            amount = round(random.uniform(5000, 50000), 2)
        else:  # unusual
            amount = round(random.uniform(1, 9999), 2)
            # Make amount have unusual cents
            amount = int(amount) + random.choice([0.01, 0.03, 0.07, 0.09, 0.11, 0.13, 0.17, 0.19])
    else:
        # Normal transactions have more typical amounts
        amount_type = random.choice(['small', 'medium', 'large'])
        if amount_type == 'small':
            amount = round(random.uniform(1, 100), 2)
        elif amount_type == 'medium':
            amount = round(random.uniform(100, 1000), 2)
        else:  # large
            amount = round(random.uniform(1000, 5000), 2)
    
    transaction['amount'] = amount
    
    # Add categorical fields
    transaction['transaction_type'] = random.choice(transaction_types)
    transaction['merchant_category'] = random.choice(merchant_categories)
    transaction['location'] = random.choice(locations)
    transaction['device_used'] = random.choice(device_used)
    transaction['is_fraud'] = 1 if is_fraud else 0
    transaction['fraud_type'] = random.choice(['account_takeover', 'identity_theft', 'card_not_present']) if is_fraud else None
    
    # Add derived fields
    transaction['time_since_last_transaction'] = random.randint(60, 86400)  # seconds
    transaction['spending_deviation_score'] = random.uniform(0, 1) if not is_fraud else random.uniform(0.7, 1)
    transaction['velocity_score'] = random.uniform(0, 1) if not is_fraud else random.uniform(0.7, 1)
    transaction['geo_anomaly_score'] = random.uniform(0, 1) if not is_fraud else random.uniform(0.7, 1)
    transaction['payment_channel'] = random.choice(payment_channels)
    transaction['ip_address'] = f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
    transaction['device_hash'] = ''.join(random.choices('0123456789abcdef', k=32))
    
    # Add time-based features
    transaction['amount_log'] = np.log1p(amount)
    transaction['hour_of_day'] = timestamp.hour
    transaction['day_of_week'] = timestamp.weekday()
    transaction['month'] = timestamp.month
    transaction['year'] = timestamp.year
    transaction['fraud_label'] = 1 if is_fraud else 0
    
    # Add normalized scores
    transaction['velocity_score_norm'] = transaction['velocity_score'] / 1.0
    transaction['geo_anomaly_bin'] = 1 if transaction['geo_anomaly_score'] > 0.7 else 0
    transaction['device_channel'] = f"{transaction['device_used']}_{transaction['payment_channel']}"
    transaction['spending_risk'] = random.uniform(0, 1) if not is_fraud else random.uniform(0.7, 1)
    
    return transaction

def generate_compatible_test_data(reference_data_path, output_path, num_transactions=500, fraud_ratio=0.15):
    """
    Generate test data compatible with the reference data structure.
    
    Args:
        reference_data_path (str): Path to reference data
        output_path (str): Path to save the generated test data
        num_transactions (int): Number of transactions to generate
        fraud_ratio (float): Ratio of fraudulent transactions
        
    Returns:
        DataFrame: Generated test data
    """
    # Load reference data
    reference_df = load_reference_data(reference_data_path)
    
    # Generate synthetic transactions
    transactions = []
    for i in range(num_transactions):
        # Determine if this transaction should be fraudulent
        is_fraud = random.random() < fraud_ratio
        
        # Generate transaction
        transaction = generate_synthetic_transaction(reference_df, i + 1, is_fraud)
        transactions.append(transaction)
    
    # Convert to DataFrame
    test_df = pd.DataFrame(transactions)
    
    # Ensure all columns from reference data are present
    for col in reference_df.columns:
        if col not in test_df.columns:
            if reference_df[col].dtype == 'object':
                test_df[col] = None
            else:
                test_df[col] = 0
    
    # Reorder columns to match reference data
    test_df = test_df[reference_df.columns]
    
    # Save to output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_path.endswith('.parquet'):
        test_df.to_parquet(output_path, index=False)
    elif output_path.endswith('.csv'):
        test_df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_path}")
    
    logger.info(f"Generated {num_transactions} transactions and saved to {output_path}")
    logger.info(f"Fraud ratio: {test_df['is_fraud'].mean():.2f}")
    
    return test_df

def main():
    """Main function to parse arguments and generate test data."""
    parser = argparse.ArgumentParser(description='Generate test data compatible with reference data')
    parser.add_argument('--reference-data', type=str, default='data/processed/transactions.parquet',
                        help='Path to reference data')
    parser.add_argument('--output-path', type=str, default='data/processed/test_data.parquet',
                        help='Path to save the generated test data')
    parser.add_argument('--num-transactions', type=int, default=500,
                        help='Number of transactions to generate')
    parser.add_argument('--fraud-ratio', type=float, default=0.15,
                        help='Ratio of fraudulent transactions')
    
    args = parser.parse_args()
    
    # Generate test data
    generate_compatible_test_data(
        args.reference_data,
        args.output_path,
        args.num_transactions,
        args.fraud_ratio
    )

if __name__ == "__main__":
    main()
