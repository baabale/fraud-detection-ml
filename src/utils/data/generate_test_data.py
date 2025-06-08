"""
Utility script to generate synthetic transaction data for testing.
This can be used to create test data for the streaming pipeline.
"""
import os
import argparse
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import socket
import time

def generate_transaction(transaction_id, timestamp=None, fraud_probability=0.01, previous_transactions=None):
    """
    Generate a single synthetic transaction.
    
    Args:
        transaction_id (int): Transaction ID
        timestamp (datetime, optional): Transaction timestamp
        fraud_probability (float): Probability of generating a fraudulent transaction
        previous_transactions (list, optional): List of previous transactions for the same account
        
    Returns:
        dict: Transaction data
    """
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now()
    
    # Decide if this transaction is fraudulent
    is_fraud = random.random() < fraud_probability
    
    # Generate account numbers
    sender_account = f"ACC{random.randint(1000000, 9999999)}"
    receiver_account = f"ACC{random.randint(1000000, 9999999)}"
    
    # Generate transaction amount
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
    
    # Generate merchant category
    merchant_categories = [
        'Grocery', 'Restaurant', 'Retail', 'Travel', 'Entertainment',
        'Utilities', 'Healthcare', 'Education', 'Financial', 'Other'
    ]
    merchant_category = random.choice(merchant_categories)
    
    # Generate location
    countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Japan', 'Australia', 'Brazil', 'India', 'China']
    cities = ['New York', 'Los Angeles', 'London', 'Berlin', 'Paris', 'Tokyo', 'Sydney', 'Sao Paulo', 'Mumbai', 'Beijing']
    
    if is_fraud:
        # Fraudulent transactions might have unusual location patterns
        country = random.choice(countries)
        city = random.choice(cities)
    else:
        # Normal transactions tend to have consistent locations
        country_idx = random.randint(0, len(countries) - 1)
        country = countries[country_idx]
        city = cities[country_idx]  # Match city to country for consistency
    
    # Generate device type
    device_types = ['Mobile', 'Desktop', 'Tablet', 'ATM', 'POS']
    device_type = random.choice(device_types)
    
    # Generate payment method
    payment_methods = ['Credit Card', 'Debit Card', 'Bank Transfer', 'Mobile Payment', 'Cash']
    payment_method = random.choice(payment_methods)
    
    # Generate account balances
    oldbalanceOrg = round(random.uniform(1000, 50000), 2)
    newbalanceOrig = max(0, oldbalanceOrg - amount)
    oldbalanceDest = round(random.uniform(500, 20000), 2)
    newbalanceDest = oldbalanceDest + amount
    
    # Generate time-based features
    hour = timestamp.hour
    day = timestamp.day
    month = timestamp.month
    
    # Calculate cyclical time features
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    day_sin = np.sin(2 * np.pi * day / 31)
    day_cos = np.cos(2 * np.pi * day / 31)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # Calculate derived features
    amount_log = np.log1p(amount)  # Log-transformed amount
    
    # Generate anomaly scores
    # Higher scores for fraudulent transactions
    if is_fraud:
        spending_deviation_score = random.uniform(0.7, 1.0)
        velocity_score = random.uniform(0.7, 1.0)
        geo_anomaly_score = random.uniform(0.7, 1.0)
    else:
        spending_deviation_score = random.uniform(0.0, 0.3)
        velocity_score = random.uniform(0.0, 0.3)
        geo_anomaly_score = random.uniform(0.0, 0.3)
    
    # Normalize velocity score
    velocity_score_norm = velocity_score / 1.0
    
    # Generate transaction history features
    if previous_transactions and len(previous_transactions) > 0:
        # Calculate time since previous transaction
        prev_timestamp = datetime.fromisoformat(previous_transactions[-1]['timestamp'])
        time_since_prev_tx = (timestamp - prev_timestamp).total_seconds() / 3600  # hours
        
        # Calculate amount delta and ratio compared to previous transaction
        prev_amount = previous_transactions[-1]['amount']
        amount_delta = amount - prev_amount
        amount_ratio = amount / max(1, prev_amount)  # Avoid division by zero
        
        # Calculate aggregates over last 5 transactions (or fewer if not available)
        last_n = min(5, len(previous_transactions))
        recent_txs = previous_transactions[-last_n:]
        recent_amounts = [tx['amount'] for tx in recent_txs]
        
        avg_amount_last5 = sum(recent_amounts) / len(recent_amounts)
        max_amount_last5 = max(recent_amounts)
        tx_count_last5 = len(recent_txs)
        
        # Calculate transaction frequency and time span
        if len(previous_transactions) > 1:
            first_timestamp = datetime.fromisoformat(previous_transactions[0]['timestamp'])
            time_span_days = (timestamp - first_timestamp).total_seconds() / (24 * 3600)  # days
            transaction_frequency = len(previous_transactions) / max(1, time_span_days)  # tx per day
        else:
            time_span_days = 0
            transaction_frequency = 0
        
        # Calculate amount deviation from average
        amount_deviation_from_avg = amount - avg_amount_last5
        
        # Calculate transaction velocity (transactions per hour)
        if time_span_days > 0:
            tx_velocity = len(previous_transactions) / (time_span_days * 24)  # tx per hour
        else:
            tx_velocity = 0
    else:
        # Default values when no transaction history is available
        time_since_prev_tx = 24  # Assume 24 hours since last transaction
        amount_delta = 0
        amount_ratio = 1
        avg_amount_last5 = amount
        max_amount_last5 = amount
        tx_count_last5 = 0
        time_span_days = 0
        transaction_frequency = 0
        amount_deviation_from_avg = 0
        tx_velocity = 0
    
    # Create transaction with all features
    transaction = {
        'transaction_id': f"TX{transaction_id}",
        'timestamp': timestamp.isoformat(),
        'amount': amount,
        'sender_account': sender_account,
        'receiver_account': receiver_account,
        'merchant_category': merchant_category,
        'country': country,
        'city': city,
        'device_type': device_type,
        'payment_method': payment_method,
        'is_fraud': 1 if is_fraud else 0,
        # Add balance features
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        # Add time features
        'hour': hour,
        # Add cyclical time features
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'day_sin': day_sin,
        'day_cos': day_cos,
        'month_sin': month_sin,
        'month_cos': month_cos,
        # Add derived features
        'amount_log': amount_log,
        # Add anomaly scores
        'spending_deviation_score': spending_deviation_score,
        'velocity_score': velocity_score,
        'geo_anomaly_score': geo_anomaly_score,
        'velocity_score_norm': velocity_score_norm,
        # Add transaction history features
        'time_since_prev_tx': time_since_prev_tx,
        'amount_delta': amount_delta,
        'amount_ratio': amount_ratio,
        'avg_amount_last5': avg_amount_last5,
        'max_amount_last5': max_amount_last5,
        'tx_count_last5': tx_count_last5,
        'time_span_days': time_span_days,
        'transaction_frequency': transaction_frequency,
        'amount_deviation_from_avg': amount_deviation_from_avg,
        'tx_velocity': tx_velocity
    }
    
    return transaction

def generate_transactions(num_transactions, output_path=None, format='json', 
                         start_time=None, time_interval=60, fraud_probability=0.01):
    """
    Generate multiple synthetic transactions.
    
    Args:
        num_transactions (int): Number of transactions to generate
        output_path (str, optional): Path to save the transactions
        format (str): Output format ('json', 'csv', or 'parquet')
        start_time (datetime, optional): Starting timestamp for transactions
        time_interval (int): Time interval between transactions in seconds
        fraud_probability (float): Probability of generating a fraudulent transaction
        
    Returns:
        list or DataFrame: Generated transactions
    """
    if start_time is None:
        start_time = datetime.now()
    
    # Generate transactions
    transactions = []
    
    # Create account-based transaction history
    account_history = {}
    
    for i in range(num_transactions):
        # Calculate timestamp for this transaction
        timestamp = start_time + timedelta(seconds=i * time_interval)
        
        # Randomly decide if this transaction should be associated with an existing account
        use_existing_account = random.random() < 0.3 and len(account_history) > 0
        
        if use_existing_account:
            # Select a random existing account
            sender_account = random.choice(list(account_history.keys()))
            previous_transactions = account_history[sender_account]
        else:
            sender_account = None
            previous_transactions = None
        
        # Generate transaction
        transaction = generate_transaction(i + 1, timestamp, fraud_probability, previous_transactions)
        transactions.append(transaction)
        
        # Update account history
        if sender_account is None:
            sender_account = transaction['sender_account']
        
        if sender_account not in account_history:
            account_history[sender_account] = []
        
        account_history[sender_account].append(transaction)
    
    # Save transactions if output path is provided
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(transactions, f, indent=2)
        elif format == 'csv':
            pd.DataFrame(transactions).to_csv(output_path, index=False)
        elif format == 'parquet':
            pd.DataFrame(transactions).to_parquet(output_path, index=False)
        
        print(f"Generated {num_transactions} transactions and saved to {output_path}")
    
    # Return as DataFrame or list depending on format
    if format in ['csv', 'parquet']:
        return pd.DataFrame(transactions)
    else:
        return transactions

def stream_transactions(num_transactions, host='localhost', port=9999, 
                       interval=1, fraud_probability=0.01):
    """
    Stream synthetic transactions to a socket.
    
    Args:
        num_transactions (int): Number of transactions to generate
        host (str): Socket host
        port (int): Socket port
        interval (float): Time interval between transactions in seconds
        fraud_probability (float): Probability of generating a fraudulent transaction
    """
    # Create a socket connection
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        print(f"Connected to {host}:{port}")
        
        # Generate and stream transactions
        for i in range(num_transactions):
            # Generate transaction
            transaction = generate_transaction(i + 1, datetime.now(), fraud_probability)
            
            # Convert to JSON and send
            json_data = json.dumps(transaction) + '\n'
            s.send(json_data.encode())
            
            print(f"Sent transaction {i+1}/{num_transactions}: {transaction['transaction_id']}")
            
            # Wait for the next interval
            time.sleep(interval)
        
        # Close the connection
        s.close()
        print("Streaming completed")
    
    except ConnectionRefusedError:
        print(f"Connection refused to {host}:{port}. Make sure the server is running.")
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    """
    Main function to generate synthetic transaction data.
    """
    parser = argparse.ArgumentParser(description='Generate synthetic transaction data')
    parser.add_argument('--num-transactions', type=int, default=100,
                        help='Number of transactions to generate')
    parser.add_argument('--output-path', type=str, default=None,
                        help='Path to save the transactions')
    parser.add_argument('--format', type=str, choices=['json', 'csv', 'parquet', 'stream'],
                        default='json', help='Output format')
    parser.add_argument('--fraud-probability', type=float, default=0.01,
                        help='Probability of generating a fraudulent transaction')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Socket host (for streaming)')
    parser.add_argument('--port', type=int, default=9999,
                        help='Socket port (for streaming)')
    parser.add_argument('--interval', type=float, default=1,
                        help='Time interval between transactions in seconds (for streaming)')
    args = parser.parse_args()
    
    if args.format == 'stream':
        # Stream transactions to a socket
        stream_transactions(
            args.num_transactions,
            args.host,
            args.port,
            args.interval,
            args.fraud_probability
        )
    else:
        # Generate transactions and save to file
        if args.output_path is None:
            if args.format == 'json':
                args.output_path = '../../data/test/transactions.json'
            elif args.format == 'csv':
                args.output_path = '../../data/test/transactions.csv'
            elif args.format == 'parquet':
                args.output_path = '../../data/test/transactions.parquet'
        
        generate_transactions(
            args.num_transactions,
            args.output_path,
            args.format,
            fraud_probability=args.fraud_probability
        )

if __name__ == "__main__":
    main()
