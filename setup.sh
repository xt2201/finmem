#!/usr/bin/env bash
set -e

echo "======================================"
echo "FinMem Environment & Data Setup Script"
echo "======================================"

# 1. Create virtual environment with uv
echo ">> Creating virtual environment with uv..."
uv venv .venv --python 3.10

# 2. Activate virtual environment
echo ">> Activating virtual environment..."
source .venv/bin/activate

# 3. Install dependencies
echo ">> Installing dependencies..."
uv pip install -r pyproject.toml

# 4. Prepare data
echo ">> Preparing sample data..."
unzip -o data-pipeline/Fake-Sample-Data.zip -d data/06_input/

echo ">> Converting dataset format..."
python -c "
import pickle
import os

input_path = 'data/06_input/Fake-Sample-Data/example_output/env_data.pkl'
output_path = 'data/06_input/subset_symbols.pkl'

if os.path.exists(input_path):
    with open(input_path, 'rb') as f:
        d = pickle.load(f)
    new_d = {
        k: {
            'price': v[0].get('price', {}), 
            'news': v[1].get('news', {}), 
            'filing_q': v[2].get('filing_q', {}), 
            'filing_k': v[3].get('filing_k', {})
        } 
        for k, v in d.items()
    }
    with open(output_path, 'wb') as f:
        pickle.dump(new_d, f)
    print(f'Successfully converted dataset and saved to {output_path}')
else:
    print(f'Error: Could not find {input_path}')
"

echo ""
echo "======================================"
echo "Setup complete!"
echo "Please ensure you have set CEREBRAS_API_KEY and HF_TOKEN in your .env file."
echo "Activate the virtual environment anytime using: source .venv/bin/activate"
echo "======================================"
