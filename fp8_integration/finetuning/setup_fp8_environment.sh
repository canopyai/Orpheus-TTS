#!/bin/bash
#
# FP8 Environment Setup Script for Orpheus-TTS
# Configures dependencies for FP8 training with NVIDIA Transformer Engine
#
set -e

echo "----------------------------------------"
echo "Orpheus-TTS FP8 Environment Setup"
echo "----------------------------------------"

# Check CUDA installation
echo "[1/9] Verifying CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: CUDA not found. CUDA 12.1 or later is required."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d'.' -f1)
CUDA_MINOR=$(echo $CUDA_VERSION | cut -d'.' -f2)

echo "CUDA version: $CUDA_VERSION"

if [ "$CUDA_MAJOR" -lt 12 ] || ([ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -lt 1 ]); then
    echo "ERROR: CUDA 12.1 or later required. Found: $CUDA_VERSION"
    exit 1
fi

# Verify GPU compatibility
echo "[2/9] Checking GPU architecture..."
python3 -c "
import torch
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability()
    device_name = torch.cuda.get_device_name()
    print(f'Device: {device_name}')
    print(f'Compute Capability: {capability[0]}.{capability[1]}')
    if capability[0] < 8 or (capability[0] == 8 and capability[1] < 9):
        print('WARNING: Hardware FP8 acceleration requires CC 8.9+ (Ada/Hopper/Blackwell)')
else:
    print('ERROR: No CUDA GPU detected')
    exit(1)
"

# Python environment setup
echo "[3/9] Configuring Python environment..."
read -p "Create virtual environment? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 -m venv fp8_env
    source fp8_env/bin/activate
    pip install --upgrade pip setuptools wheel
fi

# Install PyTorch
echo "[4/9] Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu${CUDA_MAJOR}${CUDA_MINOR}

# Install core dependencies
echo "[5/9] Installing dependencies..."
pip install transformers datasets accelerate peft wandb pyyaml numpy scipy

# Install NVIDIA Transformer Engine
echo "[6/9] Installing NVIDIA Transformer Engine..."
read -p "Build from source? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ ! -d "TransformerEngine" ]; then
        git clone https://github.com/NVIDIA/TransformerEngine.git
    fi
    cd TransformerEngine
    git pull
    
    export NVTE_FRAMEWORK="pytorch"
    export NVTE_WITH_USERBUFFERS=1
    export CUDNN_PATH=/usr/local/cuda
    export MAX_JOBS=$(nproc)
    
    pip install -v .
    cd ..
else:
    pip install git+https://github.com/NVIDIA/TransformerEngine.git
fi

# Install FlashAttention-3
echo "[7/9] Installing FlashAttention-3..."
read -p "Build from source for FP8 support? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ ! -d "flash-attention" ]; then
        git clone https://github.com/Dao-AILab/flash-attention.git
    fi
    cd flash-attention
    git pull
    git checkout main
    
    cd hopper
    export MAX_JOBS=$(nproc)
    python setup.py install
    cd ../..
else:
    echo "WARNING: Installing FlashAttention-2 (no FP8 support)"
    pip install flash-attn
fi

# Verify installation
echo "[8/9] Verifying installation..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')

try:
    import transformer_engine
    print('Transformer Engine: INSTALLED')
except ImportError:
    print('Transformer Engine: NOT INSTALLED')

try:
    from flash_attn import flash_attn_func
    print('FlashAttention: INSTALLED')
except ImportError:
    print('FlashAttention: NOT INSTALLED')

try:
    import transformers
    print(f'Transformers: {transformers.__version__}')
except ImportError:
    print('Transformers: NOT INSTALLED')
"

# Create verification script
echo "[9/9] Creating verification script..."
cat > test_fp8_setup.py << 'EOF'
"""FP8 Setup Verification"""
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

print("Testing FP8 configuration...")

try:
    model = te.Linear(1024, 1024)
    x = torch.randn(32, 1024).cuda()
    
    fp8_recipe = recipe.DelayedScaling(
        fp8_format=recipe.Format.HYBRID,
        amax_history_len=16,
        amax_compute_algo="max"
    )
    
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        y = model(x)
    
    print("FP8 autocast: PASSED")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    
except Exception as e:
    print(f"FP8 autocast: FAILED - {e}")

capability = torch.cuda.get_device_capability()
if capability[0] >= 8 and capability[1] >= 9:
    print(f"GPU FP8 support: YES (CC {capability[0]}.{capability[1]})")
else:
    print(f"GPU FP8 support: LIMITED (CC {capability[0]}.{capability[1]})")
EOF

python3 test_fp8_setup.py

echo ""
echo "----------------------------------------"
echo "Setup Complete"
echo "----------------------------------------"
echo "Environment activation: source fp8_env/bin/activate"
echo "Configuration file: fp8_config.yaml"
echo "Run training: python fp8_finetune.py"
echo ""
echo "For containerized deployment:"
echo "  docker pull nvcr.io/nvidia/pytorch:24.01-py3"
echo "----------------------------------------"
