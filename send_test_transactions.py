#!/usr/bin/env python
"""
Script to generate synthetic transactions and send them to a socket server
for testing the streaming fraud detection system.
"""
import socket
import time
import json
import random
import datetime
import uuid
import argparse
from threading import Thread

def generate_transaction():
    """Generate a synthetic transaction."""
    transaction = {
        "transaction_id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "amount": round(random.uniform(10, 10000), 2),
        "sender_account": f"ACC-{random.randint(10000, 99999)}",
        "receiver_account": f"ACC-{random.randint(10000, 99999)}",
        "merchant_category": random.choice(["retail", "food", "travel", "entertainment", "utility"])
    }
    
    # Occasionally generate suspicious transactions
    if random.random() < 0.05:  # 5% chance of fraud
        # High amount transaction
        transaction["amount"] = round(random.uniform(5000, 50000), 2)
        
        # Add some fraud indicators
        if random.random() < 0.5:
            transaction["receiver_account"] = f"SUSP-{random.randint(10000, 99999)}"
    
    return transaction

def send_transactions(host, port, interval):
    """Send transactions to a socket server."""
    try:
        # Create a socket connection
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((host, port))
        s.listen(1)
        print(f"Socket server started on {host}:{port}")
        print("Waiting for Spark Streaming to connect...")
        
        conn, addr = s.accept()
        print(f"Connected by {addr}")
        
        try:
            count = 0
            while True:
                # Generate a transaction
                transaction = generate_transaction()
                
                # Convert to JSON and send
                json_data = json.dumps(transaction)
                conn.send((json_data + "\n").encode())
                
                count += 1
                if count % 10 == 0:
                    print(f"Sent {count} transactions")
                
                # Wait for the specified interval
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("Stopping transaction generator")
        finally:
            conn.close()
            s.close()
            
    except Exception as e:
        print(f"Error: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Generate and send synthetic transactions')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Host to bind the socket server to')
    parser.add_argument('--port', type=int, default=9999,
                        help='Port to bind the socket server to')
    parser.add_argument('--interval', type=float, default=0.5,
                        help='Interval between transactions in seconds')
    args = parser.parse_args()
    
    send_transactions(args.host, args.port, args.interval)

if __name__ == "__main__":
    main()
