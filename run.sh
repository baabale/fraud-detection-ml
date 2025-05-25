#!/bin/bash
# Script to set up and run the Fraud Detection System

# Color codes for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}   ANOMALY-BASED FRAUD DETECTION SYSTEM - SETUP SCRIPT${NC}"
echo -e "${BLUE}============================================================${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "\n${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "\n${YELLOW}Failed to create virtual environment with python3. Trying python...${NC}"
        python -m venv venv
    fi
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Check if data directory has files
if [ ! -f "data/raw/financial_fraud_detection_dataset.csv" ]; then
    echo -e "\n${YELLOW}Dataset not found in data/raw directory.${NC}"
    echo -e "${YELLOW}Please download the Kaggle Financial Transactions Dataset and place it in the data/raw directory.${NC}"
    echo -e "${YELLOW}You can continue without the dataset, but some functionality will be limited.${NC}"
fi

# Run the main script
echo -e "\n${GREEN}Setup complete! Starting the Fraud Detection System...${NC}"
echo -e "${GREEN}============================================================${NC}"
python main.py

# Deactivate virtual environment when done
deactivate
