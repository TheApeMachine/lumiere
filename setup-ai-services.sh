#!/bin/bash

# Setup script for Lumiere AI Services
# Supports both CUDA and Metal (Apple Silicon)

set -e

echo "=========================================="
echo "  Lumiere AI Services Setup"
echo "=========================================="
echo ""

# Detect hardware
detect_hardware() {
    echo "Detecting hardware..."
    
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "✓ NVIDIA GPU detected (CUDA)"
        export HARDWARE="cuda"
        return
    fi
    
    # Check for Apple Silicon
    if [[ $(uname -m) == "arm64" ]] && [[ $(uname) == "Darwin" ]]; then
        echo "✓ Apple Silicon detected (Metal)"
        export HARDWARE="mps"
        return
    fi
    
    # Fallback to CPU
    echo "⚠ No GPU detected, will use CPU"
    export HARDWARE="cpu"
}

# Check Python version
check_python() {
    echo ""
    echo "Checking Python..."
    
    if ! command -v python3 &> /dev/null; then
        echo "✗ Python 3 not found. Please install Python 3.8 or higher."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "✓ Python $PYTHON_VERSION found"
}

# Setup virtual environment
setup_venv() {
    echo ""
    echo "Setting up virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo "✓ Virtual environment created"
    else
        echo "✓ Virtual environment already exists"
    fi
    
    source venv/bin/activate
    pip install --upgrade pip
}

# Install service dependencies
install_service() {
    SERVICE_NAME=$1
    SERVICE_DIR="services/$SERVICE_NAME"
    
    echo ""
    echo "Installing $SERVICE_NAME..."
    
    if [ -f "$SERVICE_DIR/requirements.txt" ]; then
        pip install -r "$SERVICE_DIR/requirements.txt"
        echo "✓ $SERVICE_NAME dependencies installed"
    else
        echo "⚠ No requirements.txt found for $SERVICE_NAME"
    fi
}

# Main installation
main() {
    detect_hardware
    check_python
    setup_venv
    
    echo ""
    echo "Installing AI service dependencies..."
    echo "This may take several minutes..."
    echo ""
    
    # Install each service
    install_service "image-generator"
    install_service "video-generator"
    install_service "audio-analyzer"
    
    echo ""
    echo "=========================================="
    echo "  Setup Complete!"
    echo "=========================================="
    echo ""
    echo "Hardware: $HARDWARE"
    echo ""
    echo "To start services:"
    echo "  1. Activate virtual environment: source venv/bin/activate"
    echo "  2. Start individual service:"
    echo "     python services/image-generator/server.py"
    echo "     python services/video-generator/server.py"
    echo "     python services/audio-analyzer/server.py"
    echo ""
    echo "  Or use Docker Compose:"
    echo "     docker-compose up -d"
    echo ""
    echo "Then start Lumiere with AI services enabled:"
    echo "  USE_AI_SERVICES=true ./lumiere"
    echo ""
}

# Run main installation
main
