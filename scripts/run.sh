#!/bin/bash
# Script to set up and run the Fraud Detection System

# Color codes for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}   ANOMALY-BASED FRAUD DETECTION SYSTEM - SETUP SCRIPT${NC}"
echo -e "${BLUE}============================================================${NC}"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo -e "\n${YELLOW}Detected Python version: ${PYTHON_VERSION}${NC}"

# Check if Python 3.10 is available (best for TensorFlow compatibility)
if command -v python3.10 &> /dev/null; then
    echo -e "${GREEN}Python 3.10 found. Using it for better compatibility with TensorFlow.${NC}"
    PYTHON_CMD="python3.10"
else
    echo -e "${YELLOW}Python 3.10 not found. Using default Python version.${NC}"
    PYTHON_CMD="python3"
fi

# Check if virtual environment exists
VENV_NAME="venv_fraud"
if [ ! -d "$VENV_NAME" ]; then
    echo -e "\n${YELLOW}Creating virtual environment...${NC}"
    $PYTHON_CMD -m venv $VENV_NAME
    if [ $? -ne 0 ]; then
        echo -e "\n${YELLOW}Failed to create virtual environment with $PYTHON_CMD. Trying alternatives...${NC}"
        
        if command -v python3.9 &> /dev/null; then
            echo -e "${YELLOW}Trying with Python 3.9...${NC}"
            python3.9 -m venv $VENV_NAME
        elif command -v python3.8 &> /dev/null; then
            echo -e "${YELLOW}Trying with Python 3.8...${NC}"
            python3.8 -m venv $VENV_NAME
        else
            echo -e "${RED}Could not create a virtual environment. Please install Python 3.8, 3.9, or 3.10.${NC}"
            exit 1
        fi
    fi
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source $VENV_NAME/bin/activate

# Install core dependencies first
echo -e "\n${YELLOW}Installing core dependencies...${NC}"
pip install numpy pandas scikit-learn matplotlib seaborn pyyaml flask joblib

# Try to install TensorFlow
echo -e "\n${YELLOW}Attempting to install TensorFlow...${NC}"
pip install tensorflow || pip install tensorflow-macos || echo -e "${YELLOW}TensorFlow installation failed. Some functionality will be limited.${NC}"

# Try to install PySpark
echo -e "\n${YELLOW}Attempting to install PySpark...${NC}"
pip install pyspark || echo -e "${YELLOW}PySpark installation failed. Some functionality will be limited.${NC}"

# Try to install MLflow
echo -e "\n${YELLOW}Attempting to install MLflow...${NC}"
pip install mlflow || echo -e "${YELLOW}MLflow installation failed. Some functionality will be limited.${NC}"

# Check if data directory has files
if [ ! -f "data/raw/financial_fraud_detection_dataset.csv" ]; then
    echo -e "\n${YELLOW}Dataset not found in data/raw directory.${NC}"
    echo -e "${YELLOW}Please download the Kaggle Financial Transactions Dataset and place it in the data/raw directory.${NC}"
    echo -e "${YELLOW}You can continue without the dataset, but some functionality will be limited.${NC}"
fi

# Create logs directory
echo -e "\n${YELLOW}Creating logs directory...${NC}"
mkdir -p logs

# Run the main script
echo -e "\n${GREEN}Setup complete! Starting the Fraud Detection System...${NC}"
echo -e "${GREEN}============================================================${NC}"
python main.py

# Deactivate virtual environment when done
deactivate
