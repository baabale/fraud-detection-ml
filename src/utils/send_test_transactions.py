#!/usr/bin/env python
"""
Utility script to send test transactions to a socket server for testing
the streaming fraud detection system.
"""
import socket
import json
import time
import random
import argparse
from datetime import datetime, timedelta

def generate_transaction(transaction_id=None, is_fraudulent=False):
    """
    Generate a random transaction, optionally with fraudulent characteristics.
    
    Args:
        transaction_id (str, optional): Transaction ID. If None, one will be generated.
        is_fraudulent (bool): Whether to generate a fraudulent transaction.
        
    Returns:
        dict: Transaction data
    """
    # Generate transaction ID if not provided
    if transaction_id is None:
        transaction_id = f"TX{random.randint(10000, 99999)}"
    
    # Generate timestamp (current time or slightly in the past)
    timestamp = datetime.now() - timedelta(seconds=random.randint(0, 300))
    timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S")
    
    # Generate sender and receiver accounts
    sender_account = f"ACC{random.randint(1000, 9999)}"
    receiver_account = f"ACC{random.randint(1000, 9999)}"
    
    # Merchant categories
    merchant_categories = [
        "retail", "food", "travel", "entertainment", "healthcare", 
        "education", "utilities", "technology", "automotive", "financial"
    ]
    
    # Generate transaction amount
    if is_fraudulent:
        # Fraudulent transactions tend to be either very small or very large
        if random.random() < 0.5:
            amount = round(random.uniform(0.01, 1.0), 2)  # Very small amount
        else:
            amount = round(random.uniform(5000.0, 50000.0), 2)  # Very large amount
        
        # Fraudulent transactions often happen at unusual times
        hour = random.randint(0, 5)  # Late night/early morning
        timestamp = datetime.now().replace(hour=hour, minute=random.randint(0, 59))
        timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        
        # Fraudulent transactions often use unusual merchant categories
        merchant_category = random.choice(["crypto", "gambling", "foreign_exchange", "unclassified"])
    else:
        # Normal transaction amount
        amount = round(random.uniform(10.0, 1000.0), 2)
        
        # Normal merchant category
        merchant_category = random.choice(merchant_categories)
    
    # Create transaction data
    transaction = {
        "transaction_id": transaction_id,
        "timestamp": timestamp_str,
        "amount": amount,
        "sender_account": sender_account,
        "receiver_account": receiver_account,
        "merchant_category": merchant_category
    }
    
    return transaction

def send_transactions(host, port, num_transactions, delay, fraud_ratio=0.1):
    """
    Send test transactions to a socket server.
    
    Args:
        host (str): Socket server host
        port (int): Socket server port
        num_transactions (int): Number of transactions to send
        delay (float): Delay between transactions in seconds
        fraud_ratio (float): Ratio of fraudulent transactions to generate
    """
    # Create a socket connection
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # Connect to the socket server
        print(f"Connecting to socket server at {host}:{port}...")
        sock.connect((host, port))
        print("Connected successfully!")
        
        # Send transactions
        print(f"Sending {num_transactions} transactions (fraud ratio: {fraud_ratio})...")
        
        for i in range(num_transactions):
            # Determine if this transaction should be fraudulent
            is_fraudulent = random.random() < fraud_ratio
            
            # Generate transaction
            transaction = generate_transaction(is_fraudulent=is_fraudulent)
            
            # Convert to JSON and send
            transaction_json = json.dumps(transaction)
            sock.sendall((transaction_json + "\n").encode())
            
            # Print transaction details
            fraud_indicator = "🚨 FRAUD" if is_fraudulent else "✅ NORMAL"
            print(f"[{i+1}/{num_transactions}] {fraud_indicator} - ID: {transaction['transaction_id']}, "
                  f"Amount: ${transaction['amount']:.2f}, Category: {transaction['merchant_category']}")
            
            # Wait before sending the next transaction
            time.sleep(delay)
        
        print("All transactions sent successfully!")
    
    except ConnectionRefusedError:
        print(f"Error: Connection refused. Make sure the socket server is running at {host}:{port}")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Close the socket connection
        sock.close()

def main():
    """Main function to parse arguments and send transactions."""
    parser = argparse.ArgumentParser(description="Send test transactions to a socket server")
    parser.add_argument("--host", type=str, default="localhost", 
                        help="Socket server host (default: localhost)")
    parser.add_argument("--port", type=int, default=9999, 
                        help="Socket server port (default: 9999)")
    parser.add_argument("--num-transactions", type=int, default=10, 
                        help="Number of transactions to send (default: 10)")
    parser.add_argument("--delay", type=float, default=1.0, 
                        help="Delay between transactions in seconds (default: 1.0)")
    parser.add_argument("--fraud-ratio", type=float, default=0.1, 
                        help="Ratio of fraudulent transactions (default: 0.1)")
    
    args = parser.parse_args()
    
    # Send transactions
    send_transactions(
        args.host, 
        args.port, 
        args.num_transactions, 
        args.delay, 
        args.fraud_ratio
    )

if __name__ == "__main__":
    main()
