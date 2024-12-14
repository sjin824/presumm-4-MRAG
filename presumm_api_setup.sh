#!/bin/bash

# 1. Create a virtual environment
python3.9 -m venv venv
source venv/bin/activate

# 2. Install PRESUMM required & api required packages. 
pip install -r requirements.txt

# 3. Run the API
python presumm_api.py