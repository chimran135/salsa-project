#!/bin/bash

# Source the Conda initialization script
source ~/anaconda3/etc/profile.d/conda.sh

# Activate the Conda environment
conda activate salsa

# Check if the "salsa" environment is activated
if [ "$CONDA_DEFAULT_ENV" == "salsa" ]; then
    echo "Conda environment 'salsa' activated successfully."
else
    echo "Failed to activate Conda environment 'salsa'."
    exit 1  # Exit the script with an error code
fi

# Navigate to Porject Directory
cd /var/www/html/salsa

# Run your FastAPI application or other commands here
uvicorn main:app --host 127.0.0.1 --port 8000 --workers 4

