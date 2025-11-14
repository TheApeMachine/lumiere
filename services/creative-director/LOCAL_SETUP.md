# Local LLM Setup Guide

This guide explains how to set up a fully local creative director service using local LLMs instead of OpenAI's API.

## Prerequisites

- M4 Max with 128GB unified memory (or similar high-memory system)
- Python 3.10-3.13
- Homebrew (for macOS)

## Installation

### 1. Install llama-cpp-python with Metal support (Apple Silicon)

```bash
# For Apple Silicon (M1/M2/M3/M4)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# For Linux/CPU-only
pip install llama-cpp-python
```

### 2. Download a GGUF Model

Recommended models for 128GB unified memory:

#### Option A: Llama 3.1 70B (Q4_K_M quantization)
```bash
# Download using huggingface-cli
pip install huggingface-hub
huggingface-cli download TheBloke/Llama-3.1-70B-Instruct-GGUF --local-dir ./models --local-dir-use-symlinks False

# Or download manually from:
# https://huggingface.co/TheBloke/Llama-3.1-70B-Instruct-GGUF
# Look for: llama-3.1-70b-instruct.Q4_K_M.gguf (~40GB)
```

#### Option B: Mistral 7B Instruct (Q4_K_M quantization) - Faster, smaller
```bash
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF --local-dir ./models --local-dir-use-symlinks False

# Or download manually:
# https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
# Look for: mistral-7b-instruct-v0.2.Q4_K_M.gguf (~4.5GB)
```

#### Option C: Mixtral 8x7B (Q4_K_M quantization) - Best balance
```bash
huggingface-cli download TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF --local-dir ./models --local-dir-use-symlinks False

# Or download manually:
# https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF
# Look for: mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf (~26GB)
```

## Configuration

The system now **automatically downloads models** if they're not found locally, just like transformers!

### Option 1: Use Model Aliases (Easiest)

```bash
# Use a simple alias - model will be downloaded automatically
export LOCAL_LLM_MODEL="mistral-7b"  # or "mixtral-8x7b", "llama-3.1-70b", "llama-3.1-8b"

# Start the service - model downloads automatically on first run
python server.py
```

### Option 2: Use HuggingFace Repo ID

```bash
# Use a HuggingFace repo ID - GGUF file will be found and downloaded automatically
export LOCAL_LLM_MODEL_PATH="TheBloke/Mistral-7B-Instruct-v0.2-GGUF"

# Start the service
python server.py
```

### Option 3: Use Local File Path

```bash
# If you already have a model file locally
export LOCAL_LLM_MODEL_PATH="/path/to/your/model.gguf"

# Start the service
python server.py
```

### Option 4: Use Transformers Models

```bash
# Use transformers library (auto-downloads like normal)
export LOCAL_LLM_MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
export LOCAL_LLM_USE_GGUF="false"

# Start the service
python server.py
```

### Default Behavior

If no environment variables are set, the system defaults to `mixtral-8x7b` (much more capable than 7B models, fits comfortably in 128GB unified memory) and downloads it automatically on first run. For even better quality, you can use `llama-3.1-70b`.

## Model Recommendations

### For Best Quality (128GB unified memory)
- **Llama 3.1 70B Q4_K_M**: ~40GB, excellent quality, slower inference
- **Mixtral 8x7B Q4_K_M**: ~26GB, very good quality, faster than 70B

### For Faster Inference
- **Mistral 7B Q4_K_M**: ~4.5GB, good quality, fastest inference
- **Llama 3.1 8B Q4_K_M**: ~5GB, good quality, fast inference

### Quantization Levels
- **Q4_K_M**: Recommended balance of quality and speed
- **Q5_K_M**: Better quality, larger file size
- **Q8_0**: Near full precision, largest file size

## Performance Tips

1. **Use GGUF models**: They're optimized for local inference and work great with llama-cpp-python
2. **Metal acceleration**: Automatically enabled on Apple Silicon with `-DLLAMA_METAL=on`
3. **Context window**: Default is 4096 tokens, adjust in `local_agents.py` if needed
4. **Batch processing**: The service handles multiple requests sequentially

## Troubleshooting

### Model not loading
- Check that the model path is correct
- Ensure you have enough memory (check with `top` or Activity Monitor)
- Try a smaller quantized model (Q4_K_M instead of Q8_0)

### Slow inference
- Use a smaller model (7B instead of 70B)
- Use a more aggressive quantization (Q4 instead of Q5)
- Reduce `max_tokens` in `local_agents.py`

### Out of memory
- Use a smaller model
- Use more aggressive quantization
- Reduce context window size

## Alternative: Using Transformers

If you prefer using the transformers library instead of GGUF:

```bash
pip install transformers torch accelerate

export LOCAL_LLM_MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
export LOCAL_LLM_USE_GGUF="false"
```

Note: Transformers models use more memory but may be easier to work with for some use cases.

