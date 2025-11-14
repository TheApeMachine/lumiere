# Local LLM Migration Summary

This document summarizes the migration from OpenAI API to fully local LLM inference for the Lumiere creative director service.

## What Changed

### 1. Creative Director Service
- **Replaced**: OpenAI Agents SDK (`openai-agents`)
- **With**: Local LLM system using `llama-cpp-python` (GGUF models) or `transformers` (HuggingFace models)
- **New File**: `services/creative-director/local_agents.py` - Compatible agent wrapper
- **Updated**: `services/creative-director/filmmaking_team.py` - Now supports local LLMs
- **Updated**: `services/creative-director/server.py` - Initializes local LLM on startup

### 2. Image Generator Service
- **Upgraded**: From Stable Diffusion 1.5 to Stable Diffusion XL
- **Benefit**: Significantly better image quality
- **Configurable**: Via `SD_MODEL_ID` environment variable
- **Options**: 
  - `stabilityai/stable-diffusion-xl-base-1.0` (default, best quality)
  - `stabilityai/sdxl-turbo` (faster generation)

### 3. Requirements
- **Added**: `llama-cpp-python>=0.2.0` for local LLM inference
- **Commented**: `openai-agents` (can be re-enabled if needed)

## Quick Start

### 1. Install Dependencies

```bash
cd services/creative-director

# Install llama-cpp-python with Metal support (Apple Silicon)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# Install other requirements
pip install -r requirements.txt
```

### 2. Configure Environment (Models Download Automatically!)

**No manual download needed!** The system automatically downloads models on first run, just like transformers.

```bash
# Option 1: Use a model alias (easiest - recommended)
export LOCAL_LLM_MODEL="mistral-7b"  # Downloads automatically

# Option 2: Use HuggingFace repo ID (also downloads automatically)
export LOCAL_LLM_MODEL_PATH="TheBloke/Mistral-7B-Instruct-v0.2-GGUF"

# Option 3: Use transformers model name (downloads automatically)
export LOCAL_LLM_MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
export LOCAL_LLM_USE_GGUF="false"

# Optional: Configure image model
export SD_MODEL_ID="stabilityai/stable-diffusion-xl-base-1.0"

# Start the service - model downloads automatically on first run
python server.py
```

**Available Model Aliases:**
- `mistral-7b` - Fastest (~4.5GB, downloads automatically)
- `mixtral-8x7b` - Best balance (~26GB, downloads automatically)
- `llama-3.1-70b` - Best quality (~40GB, downloads automatically)
- `llama-3.1-8b` - Good quality, fast (~5GB, downloads automatically)

**Default:** If no environment variable is set, defaults to `mixtral-8x7b` (much more capable than 7B, fits in 128GB unified memory) and downloads automatically.

## Environment Variables

### Creative Director Service

- `LOCAL_LLM_MODEL_PATH`: Path to GGUF model file (required for GGUF mode)
- `LOCAL_LLM_MODEL_NAME`: HuggingFace model name (alternative to model_path)
- `LOCAL_LLM_USE_GGUF`: Set to "true" to use GGUF (default) or "false" for transformers

### Image Generator Service

- `SD_MODEL_ID`: Stable Diffusion model ID (default: `stabilityai/stable-diffusion-xl-base-1.0`)

## Model Recommendations for 128GB Unified Memory

| Model | Size | Quality | Speed | Use Case |
|-------|------|---------|-------|----------|
| Mistral 7B Q4_K_M | ~4.5GB | Good | Fastest | Quick iterations, testing |
| Mixtral 8x7B Q4_K_M | ~26GB | Very Good | Fast | Production, best balance |
| Llama 3.1 70B Q4_K_M | ~40GB | Excellent | Slower | Highest quality needs |

## Benefits of Local Setup

1. **No API Costs**: Run completely free, no OpenAI API charges
2. **Privacy**: All data stays local, no external API calls
3. **Control**: Full control over model selection and configuration
4. **Performance**: With 128GB unified memory, can run large models efficiently
5. **Offline**: Works without internet connection

## Fallback Behavior

The system gracefully falls back to OpenAI Agents SDK if:
- Local LLM is not configured
- Local LLM fails to initialize
- `openai-agents` is installed and `OPENAI_API_KEY` is set

This ensures backward compatibility.

## Troubleshooting

See `services/creative-director/LOCAL_SETUP.md` for detailed troubleshooting guide.

## Next Steps

1. Download a model appropriate for your use case
2. Set `LOCAL_LLM_MODEL_PATH` environment variable
3. Start the service and test with a sample audio file
4. Monitor memory usage and adjust model size if needed

## Performance Notes

- **First run**: Models are loaded into memory (may take 30-60 seconds)
- **Inference speed**: Depends on model size and quantization
  - Mistral 7B: ~10-20 tokens/second
  - Mixtral 8x7B: ~5-10 tokens/second
  - Llama 3.1 70B: ~2-5 tokens/second
- **Memory usage**: GGUF models are memory-efficient, but larger models still need significant RAM

