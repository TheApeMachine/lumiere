# Lumiere AI Services

This directory contains Python-based AI services that run locally on consumer hardware, supporting both CUDA (NVIDIA) and Metal (Apple Silicon).

## Services

### 1. Image Generator (`image-generator/`)
- **Model**: Stable Diffusion v1.5
- **Purpose**: Generates keyframe images from text prompts
- **Hardware**: CUDA, Metal (MPS), or CPU
- **Port**: 5001

### 2. Video Generator (`video-generator/`)
- **Model**: Stable Video Diffusion / Frame Interpolation
- **Purpose**: Creates animated video segments between keyframes
- **Hardware**: CUDA, Metal (MPS), or CPU (fallback to interpolation)
- **Port**: 5002

### 3. Audio Analyzer (`audio-analyzer/`)
- **Library**: librosa
- **Purpose**: Analyzes audio for tempo, beats, intensity, and key moments
- **Hardware**: CPU (no GPU required)
- **Port**: 5003

## Hardware Support

### NVIDIA GPUs (CUDA)
All services automatically detect and use CUDA when available:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Apple Silicon (Metal/MPS)
Services automatically detect and use Metal Performance Shaders (MPS):
```bash
# Check MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"
```

### CPU Fallback
All services gracefully fall back to CPU if no GPU is available.

## Quick Start

### Option 1: Docker Compose (Recommended)

From the root directory:
```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Option 2: Manual Setup

#### 1. Image Generator
```bash
cd services/image-generator
pip install -r requirements.txt
python server.py
```

#### 2. Video Generator
```bash
cd services/video-generator
pip install -r requirements.txt
python server.py
```

#### 3. Audio Analyzer
```bash
cd services/audio-analyzer
pip install -r requirements.txt
python server.py
```

## Environment Variables

### Image Generator
- `IMAGE_SERVICE_PORT`: Port to run on (default: 5001)

### Video Generator
- `VIDEO_SERVICE_PORT`: Port to run on (default: 5002)

### Audio Analyzer
- `AUDIO_SERVICE_PORT`: Port to run on (default: 5003)

## API Endpoints

### Image Generator

**Health Check**
```bash
curl http://localhost:5001/health
```

**Generate Image**
```bash
curl -X POST http://localhost:5001/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Epic landscape with mountains",
    "output_path": "/path/to/output.png",
    "num_inference_steps": 20,
    "width": 512,
    "height": 512
  }'
```

### Video Generator

**Health Check**
```bash
curl http://localhost:5002/health
```

**Generate Video**
```bash
curl -X POST http://localhost:5002/generate \
  -H "Content-Type: application/json" \
  -d '{
    "start_frame": "/path/to/start.png",
    "end_frame": "/path/to/end.png",
    "output_path": "/path/to/output.mp4",
    "num_frames": 14,
    "fps": 7
  }'
```

### Audio Analyzer

**Health Check**
```bash
curl http://localhost:5003/health
```

**Analyze Audio**
```bash
curl -X POST http://localhost:5003/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "audio_path": "/path/to/audio.mp3"
  }'
```

## Model Information

### Stable Diffusion v1.5
- **Size**: ~4GB
- **VRAM**: 4-6GB recommended (works with 2GB with optimizations)
- **Speed**: ~2-3 seconds per image on modern GPUs
- **License**: CreativeML Open RAIL-M

### Stable Video Diffusion
- **Size**: ~5GB
- **VRAM**: 8GB+ recommended
- **Speed**: ~30-60 seconds per video on modern GPUs
- **Fallback**: Simple frame interpolation on CPU/limited VRAM

### librosa
- **Size**: Lightweight
- **Hardware**: CPU only
- **Speed**: Real-time audio analysis

## Performance Tips

### For Limited VRAM (2-4GB)
1. Image generator uses `enable_attention_slicing()` for memory efficiency
2. Video generator falls back to interpolation on low VRAM
3. Consider using `float16` precision (enabled by default on GPU)

### For Apple Silicon
1. MPS is automatically detected and used
2. Performance is comparable to entry-level NVIDIA GPUs
3. Some operations may fall back to CPU (expected behavior)

### For CPU Only
1. Image generation will be slow (~30-60 seconds per image)
2. Video generation uses fast frame interpolation
3. Audio analysis is fast on CPU

## Troubleshooting

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### CUDA Not Detected
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.version.cuda)"
```

### MPS Not Detected
```bash
# Check PyTorch MPS support
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Out of Memory
- Reduce image size (use 256x256 instead of 512x512)
- Reduce inference steps (use 15 instead of 20)
- Enable CPU offloading (already enabled for image generator)

## Integration with Main Server

The Go server automatically detects and uses these services when:
1. Services are running and accessible
2. `USE_AI_SERVICES=true` environment variable is set

If services are unavailable, the system falls back to placeholder generation.

## Development

### Adding New Models

1. Update `requirements.txt` with new dependencies
2. Modify server.py to initialize new model
3. Add new endpoint for the feature
4. Update documentation

### Testing Services

```bash
# Test each service independently
cd services/image-generator
python -c "from server import get_device; print(get_device())"
```

## License

Services use models under their respective licenses:
- Stable Diffusion: CreativeML Open RAIL-M
- librosa: ISC License
- Flask: BSD License
