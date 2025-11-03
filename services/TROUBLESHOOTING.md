# AI Services Troubleshooting Guide

Common issues and solutions for running Lumiere AI services.

## Installation Issues

### Python Dependencies Fail to Install

**Error**: `pip install` fails with compilation errors

**Solution**:
```bash
# Update pip first
pip install --upgrade pip setuptools wheel

# Install with verbose output to see what's failing
pip install -r requirements.txt -v

# For specific packages that fail, try:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # CPU
```

### CUDA Version Mismatch

**Error**: `RuntimeError: CUDA version mismatch`

**Solution**:
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch version
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Runtime Issues

### Out of Memory (OOM)

**Error**: `RuntimeError: CUDA out of memory` or system freezes

**Solutions**:

1. **Reduce image size** (services/image-generator/server.py):
```python
# Change default from 512 to 256
width = data.get('width', 256)
height = data.get('height', 256)
```

2. **Reduce inference steps**:
```python
num_inference_steps = data.get('num_inference_steps', 15)  # Was 20
```

3. **Enable sequential CPU offload** (already enabled by default):
```python
pipeline.enable_model_cpu_offload()
```

4. **Use lower precision**:
```python
# Edit server.py to use float32 instead of float16
torch_dtype=torch.float32
```

5. **Clear cache between generations**:
```python
import gc
torch.cuda.empty_cache()
gc.collect()
```

### GPU Not Detected

**Error**: "Device: cpu" when you have a GPU

**NVIDIA CUDA:**
```bash
# Check GPU
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Apple Silicon Metal:**
```bash
# Check MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"

# If False, update PyTorch
pip install --upgrade torch torchvision
```

### Service Won't Start

**Error**: Port already in use or service crashes on startup

**Solutions**:

1. **Check if port is in use**:
```bash
# Linux/Mac
lsof -i :5001  # Check image generator port
lsof -i :5002  # Check video generator port
lsof -i :5003  # Check audio analyzer port

# Kill process using port
kill -9 <PID>
```

2. **Use different ports**:
```bash
IMAGE_SERVICE_PORT=6001 python services/image-generator/server.py
VIDEO_SERVICE_PORT=6002 python services/video-generator/server.py
AUDIO_SERVICE_PORT=6003 python services/audio-analyzer/server.py
```

3. **Check logs for specific error**:
```bash
python services/image-generator/server.py 2>&1 | tee service.log
```

### Model Download Fails

**Error**: Connection timeout or download fails

**Solutions**:

1. **Use mirror**:
```bash
export HF_ENDPOINT=https://hf-mirror.com  # China mirror
```

2. **Download manually**:
```bash
# Image generator model
huggingface-cli download runwayml/stable-diffusion-v1-5

# Video generator model  
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt
```

3. **Use local model cache**:
```python
# Edit server.py to use local path
model_id = "/path/to/local/model"
```

### Slow Performance on GPU

**Issue**: Generation is slower than expected

**Solutions**:

1. **Check GPU utilization**:
```bash
nvidia-smi -l 1  # Monitor GPU usage
```

2. **Ensure GPU is actually being used**:
```python
# Add to server.py for debugging
print(f"Device: {device}")
print(f"Model device: {pipeline.device}")
```

3. **Disable safety checker** (already done for speed):
```python
safety_checker=None
```

4. **Use xformers** (optional, for even faster generation):
```bash
pip install xformers
```
```python
# Add to server.py after loading pipeline
pipeline.enable_xformers_memory_efficient_attention()
```

## API Issues

### Connection Refused

**Error**: `Connection refused` when calling service

**Solutions**:

1. **Check service is running**:
```bash
curl http://localhost:5001/health
```

2. **Check firewall**:
```bash
# Linux
sudo ufw allow 5001/tcp
sudo ufw allow 5002/tcp
sudo ufw allow 5003/tcp
```

3. **Use correct URL in Go server**:
```bash
export IMAGE_SERVICE_URL=http://localhost:5001
export VIDEO_SERVICE_URL=http://localhost:5002
export AUDIO_SERVICE_URL=http://localhost:5003
```

### Timeout Errors

**Error**: Request times out during generation

**Solutions**:

1. **Increase timeout** (pipeline/ai_services.go):
```go
client: &http.Client{Timeout: 300 * time.Second},  // Increase from 120s
```

2. **For image generation**, reduce inference steps
3. **For video generation**, reduce number of frames

### Generation Produces Poor Results

**Issue**: Images or videos are low quality or don't match prompt

**Solutions**:

1. **Increase inference steps**:
```json
{
  "num_inference_steps": 30,  // More steps = better quality
  "guidance_scale": 8.5       // Higher = more prompt adherence
}
```

2. **Improve prompts**:
```
# Bad
"forest"

# Good  
"mystical forest at sunset, ethereal light beams, cinematic lighting, detailed, 4k"
```

3. **Add negative prompts**:
```json
{
  "negative_prompt": "blurry, low quality, distorted, deformed, ugly"
}
```

## Docker Issues

### Docker Build Fails

**Error**: Build fails during pip install

**Solutions**:

1. **Increase Docker memory**:
```bash
# Docker Desktop: Settings > Resources > Memory (set to 8GB+)
```

2. **Build without cache**:
```bash
docker-compose build --no-cache
```

3. **Use pre-built image** (if available)

### Container Crashes

**Error**: Container exits immediately

**Solutions**:

1. **Check logs**:
```bash
docker-compose logs image-generator
docker-compose logs video-generator
docker-compose logs audio-analyzer
```

2. **Check resource limits**:
```bash
docker stats
```

3. **Run interactively for debugging**:
```bash
docker run -it lumiere-image-generator bash
python server.py
```

### GPU Not Available in Docker

**Error**: CUDA not found in container

**Solutions**:

1. **Install nvidia-docker2**:
```bash
# Ubuntu/Debian
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

2. **Verify GPU accessible**:
```bash
docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

3. **Check docker-compose.yml has GPU config**:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## Platform-Specific Issues

### Apple Silicon (M1/M2/M3)

**Issue**: MPS not working

**Solutions**:

1. **Update macOS** to 12.3+ (required for MPS)

2. **Update PyTorch** to 2.0+:
```bash
pip install --upgrade torch torchvision
```

3. **Fallback to CPU if needed**:
```python
# Edit server.py
device = "cpu"  # Force CPU if MPS has issues
```

### Windows

**Issue**: Various compatibility issues

**Solutions**:

1. **Use WSL2** (recommended):
```bash
wsl --install
# Then follow Linux instructions
```

2. **Or use Docker Desktop** with WSL2 backend

3. **Direct Windows installation** (more complex):
```bash
# Install Visual Studio Build Tools
# Install CUDA Toolkit
# Use conda instead of pip for easier dependencies
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Performance Optimization

### For 4GB VRAM GPUs

```python
# services/image-generator/server.py
pipeline.enable_attention_slicing()
pipeline.enable_model_cpu_offload()

# Use smaller resolution
width, height = 384, 384  # Instead of 512, 512

# Fewer steps
num_inference_steps = 15
```

### For 8GB+ VRAM GPUs

```python
# Can use higher resolution
width, height = 768, 768

# More steps for better quality
num_inference_steps = 30

# Optional: Enable xformers
pipeline.enable_xformers_memory_efficient_attention()
```

### For CPU Only

```python
# Use float32 for better CPU performance
torch_dtype=torch.float32

# Keep resolution low
width, height = 256, 256

# Minimal steps
num_inference_steps = 10
```

## Getting Help

1. **Check logs** - Most issues show clear error messages
2. **Search issues** - Check if others had same problem
3. **Test individually** - Isolate which service has issues
4. **Report bugs** - Include logs, hardware specs, and steps to reproduce

## Useful Commands

```bash
# Check Python environment
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.backends.mps.is_available())"

# Check GPU
nvidia-smi
nvidia-smi -L

# Monitor resources
htop  # CPU/RAM
nvidia-smi -l 1  # GPU
watch -n 1 docker stats  # Docker

# Test service independently
python services/image-generator/server.py
curl http://localhost:5001/health

# Clear cache
rm -rf ~/.cache/huggingface
pip cache purge
```
