#!/bin/bash

# Setup script for the Creative Director Service
# Agent-based filmmaking team for dynamic music video generation

echo "🎬 Setting up Lumiere Creative Director Service"
echo "=============================================="

# Check if Python 3.8+ is available
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 0 ]]; then
    echo "❌ Error: Python 3.8 or higher is required"
    echo "   Current version: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY environment variable not set"
    echo "   You'll need to set this to use the agent-based system"
    echo "   Export it in your shell: export OPENAI_API_KEY='your-key-here'"
fi

# Install system dependencies for audio processing
echo "🎵 Checking system dependencies for audio processing..."

# Check for ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  Warning: ffmpeg not found"
    echo "   Install with: brew install ffmpeg (macOS) or apt-get install ffmpeg (Ubuntu)"
fi

# Check for libsndfile (required by soundfile)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if ! brew list libsndfile &> /dev/null; then
        echo "📦 Installing libsndfile for audio processing..."
        brew install libsndfile
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if ! dpkg -l | grep -q libsndfile1-dev; then
        echo "📦 Installing libsndfile for audio processing..."
        sudo apt-get update
        sudo apt-get install -y libsndfile1-dev
    fi
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p temp
mkdir -p examples

# Create environment file template
if [ ! -f ".env" ]; then
    echo "📝 Creating .env template..."
    cat > .env << EOF
# Creative Director Service Configuration
OPENAI_API_KEY=your_openai_api_key_here
DEBUG=false
PORT=5003

# Agent Configuration
MAX_REVISIONS=3
AGENT_TEMPERATURE=0.7

# Audio Analysis
AUDIO_SAMPLE_RATE=22050
MAX_AUDIO_DURATION=300

# Logging
LOG_LEVEL=INFO
EOF
    echo "   ✅ Created .env file - please update with your API keys"
fi

# Test the installation
echo "🧪 Testing installation..."
python3 -c "
import librosa
import numpy as np
import flask
import pydantic
print('✅ Core dependencies imported successfully')
"

if [ $? -eq 0 ]; then
    echo "✅ Installation completed successfully!"
    echo ""
    echo "🚀 To start the Creative Director service:"
    echo "   1. Set your OPENAI_API_KEY in .env file"
    echo "   2. Run: python3 server.py"
    echo "   3. Service will be available at http://localhost:5003"
    echo ""
    echo "🎭 To see a demo of the filmmaking team:"
    echo "   python3 example_usage.py"
    echo ""
    echo "📚 API Endpoints:"
    echo "   GET  /health                 - Health check"
    echo "   POST /analyze-music          - Analyze audio file"
    echo "   POST /create-concept         - Create full concept"
    echo "   POST /generate-scene-prompts - Generate scene prompts"
    echo "   GET  /agent-status           - Check agent status"
else
    echo "❌ Installation test failed"
    echo "   Please check the error messages above"
    exit 1
fi