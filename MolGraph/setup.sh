#!/bin/bash
# ============================================================
# MolGraph Setup Script
# Run this once to install all dependencies
# ============================================================

echo "=============================================="
echo "   MolGraph - Environment Setup Script"
echo "=============================================="

# Step 1: Create virtual environment
echo ""
echo "🔧 Step 1: Creating virtual environment..."
python3 -m venv molgraph_env
source molgraph_env/bin/activate

# Step 2: Upgrade pip
echo ""
echo "🔧 Step 2: Upgrading pip..."
pip install --upgrade pip

# Step 3: Install PyTorch (CPU version for compatibility)
echo ""
echo "🔧 Step 3: Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Step 4: Install PyTorch Geometric
echo ""
echo "🔧 Step 4: Installing PyTorch Geometric..."
pip install torch-geometric

# Step 5: Install RDKit
echo ""
echo "🔧 Step 5: Installing RDKit..."
pip install rdkit

# Step 6: Install remaining dependencies
echo ""
echo "🔧 Step 6: Installing remaining dependencies..."
pip install numpy pandas scikit-learn matplotlib seaborn networkx plotly tqdm colorama
pip install fastapi uvicorn[standard] python-multipart pydantic
pip install wandb pytest jupyter ipykernel

echo ""
echo "✅ All dependencies installed successfully!"
echo ""
echo "To activate the environment: source molgraph_env/bin/activate"
echo "To run training: python main.py --mode train"
echo "To run API: python main.py --mode api"
echo "=============================================="
