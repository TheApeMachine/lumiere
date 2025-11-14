#!/bin/bash
# Setup script for LTX-Video integration

echo "Setting up LTX-Video for enhanced video generation..."

# Activate virtual environment if it exists
if [ -d "../../venv" ]; then
    echo "Activating virtual environment..."
    source ../../venv/bin/activate
fi

# Install latest diffusers with LTX-Video support
echo "Installing latest diffusers with LTX-Video support..."
pip install -U git+https://github.com/huggingface/diffusers

# Install other requirements
echo "Installing video generator requirements..."
pip install -r requirements.txt

echo "Setup complete!"
echo ""
echo "LTX-Video models available:"
echo "- Lightricks/LTX-Video-0.9.6-distilled (recommended - fastest)"
echo "- Lightricks/LTX-Video-0.9.8-dev (highest quality)"
echo ""
echo "The service will automatically download the model on first use."
echo "Restart the video generator service to use LTX-Video."