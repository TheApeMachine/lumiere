# Lumiere AI Services Architecture

## System Overview with Local AI Models

```
┌─────────────────────────────────────────────────────────────────┐
│                          Client Layer                            │
│            (Web UI, Mobile Apps, CLI, cURL)                      │
└──────────────────────┬──────────────────────────────────────────┘
                       │ HTTP/REST
┌──────────────────────▼──────────────────────────────────────────┐
│                   Main API Server (Go)                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  - Project Management                                       │ │
│  │  - File Upload (MP3, images)                               │ │
│  │  - Pipeline Orchestration                                  │ │
│  │  - Service Health Monitoring                               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                    Port: 8080                                    │
└──────────┬──────────────┬──────────────┬─────────────────────────┘
           │              │              │
           │ HTTP REST    │ HTTP REST    │ HTTP REST
           │              │              │
    ┌──────▼─────┐  ┌────▼─────┐  ┌────▼─────┐
    │  Audio     │  │  Image   │  │  Video   │
    │  Analyzer  │  │Generator │  │Generator │
    │  Service   │  │ Service  │  │ Service  │
    │            │  │          │  │          │
    │ (Python)   │  │(Python)  │  │(Python)  │
    │ Port: 5003 │  │Port:5001 │  │Port:5002 │
    └──────┬─────┘  └────┬─────┘  └────┬─────┘
           │             │              │
    ┌──────▼─────┐  ┌────▼─────┐  ┌────▼─────┐
    │  librosa   │  │  Stable  │  │  Stable  │
    │            │  │Diffusion │  │  Video   │
    │ Audio      │  │   v1.5   │  │Diffusion │
    │ Analysis   │  │          │  │    +     │
    │            │  │  ~4GB    │  │Interpolate│
    │  CPU       │  │          │  │   ~5GB   │
    └────────────┘  └────┬─────┘  └────┬─────┘
                         │              │
                    ┌────▼──────────────▼─────┐
                    │   Hardware Abstraction  │
                    │                         │
                    │  ┌─────────────────┐   │
                    │  │  CUDA (NVIDIA)  │   │
                    │  │  - GTX 1060+    │   │
                    │  │  - 4GB+ VRAM    │   │
                    │  └─────────────────┘   │
                    │                         │
                    │  ┌─────────────────┐   │
                    │  │ Metal (Apple)   │   │
                    │  │  - M1/M2/M3     │   │
                    │  │  - 8GB+ RAM     │   │
                    │  └─────────────────┘   │
                    │                         │
                    │  ┌─────────────────┐   │
                    │  │  CPU Fallback   │   │
                    │  │  - Any CPU      │   │
                    │  │  - 8GB+ RAM     │   │
                    │  └─────────────────┘   │
                    └─────────────────────────┘
```

## Data Flow

### 1. Project Creation
```
User → API Server → File System
  |
  └─> Saves: audio.mp3, character_*.png
```

### 2. Audio Analysis (Service: Audio Analyzer)
```
API Server → Audio Analyzer Service
              |
              ├─> Load MP3 with librosa
              ├─> Analyze tempo/beats (CPU)
              ├─> Calculate intensity curve
              ├─> Detect peak moments
              └─> Return analysis JSON

Response:
  - Duration, tempo, sample rate
  - Intensity curve (1-second intervals)
  - 7 key moments with timestamps
  - Beat times
```

### 3. Image Generation (Service: Image Generator)
```
For each key moment:
  API Server → Image Generator Service
                |
                ├─> Load Stable Diffusion v1.5
                ├─> Device: CUDA/Metal/CPU
                ├─> Generate 512x512 image
                |   - 20 inference steps
                |   - Guidance scale: 7.5
                |   - Attention slicing (memory opt)
                └─> Save PNG to disk

Response:
  - Image path
  - Generation metadata
  
Time: ~3-5s per image on GPU, ~30-60s on CPU
```

### 4. Video Generation (Service: Video Generator)
```
For each pair of consecutive frames:
  API Server → Video Generator Service
                |
                ├─> Load Stable Video Diffusion
                ├─> OR use frame interpolation (CPU)
                ├─> Generate 14 frames @ 7 FPS
                └─> Save MP4 to disk

Response:
  - Video path
  - Frame count, FPS
  
Time: ~30-60s per segment on GPU, ~10s on CPU (interp)
```

### 5. Video Composition
```
API Server (Go):
  ├─> Concatenate all MP4 segments
  ├─> Add original audio track
  └─> Export final_video.mp4
```

## Service Communication

### Health Checks
```go
// Go server periodically checks service health
GET http://localhost:5001/health  // Image Generator
GET http://localhost:5002/health  // Video Generator
GET http://localhost:5003/health  // Audio Analyzer

Response:
{
  "status": "healthy",
  "device": "cuda",  // or "mps" or "cpu"
  "pipeline_loaded": true
}
```

### Fallback Strategy
```go
if serviceAvailable {
    useAIService()
} else {
    useFallback()  // Simulation or placeholder
}
```

## Hardware Detection Flow

```python
def get_device():
    if torch.cuda.is_available():
        return "cuda"  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        return "mps"   # Apple Silicon
    else:
        return "cpu"   # Fallback
```

## Memory Optimization Techniques

### Image Generator
```python
# 1. FP16 precision on GPU
torch_dtype=torch.float16  # Half precision

# 2. Attention slicing
pipeline.enable_attention_slicing()

# 3. CPU offload for large models
pipeline.enable_model_cpu_offload()

# 4. Lower resolution if needed
width, height = 256, 256  # Instead of 512
```

### Video Generator
```python
# 1. Chunk decoding
decode_chunk_size=8

# 2. Fallback to interpolation
if device == "cpu" or low_vram:
    simple_interpolate_frames()  # Fast, lower quality
else:
    stable_video_diffusion()      # Slow, high quality
```

## API Contracts

### Audio Analyzer
```json
POST /analyze
{
  "audio_path": "/path/to/audio.mp3"
}

Response:
{
  "success": true,
  "analysis": {
    "duration": 180.0,
    "tempo": 120.0,
    "intensity_curve": [
      {"timestamp": 0.0, "value": 0.3},
      {"timestamp": 1.0, "value": 0.35},
      ...
    ],
    "key_moments": [
      {
        "timestamp": 0.0,
        "description": "opening scene",
        "intensity": 0.3
      },
      ...
    ]
  }
}
```

### Image Generator
```json
POST /generate
{
  "prompt": "Epic landscape with mountains",
  "output_path": "/path/to/output.png",
  "num_inference_steps": 20,
  "width": 512,
  "height": 512
}

Response:
{
  "success": true,
  "output_path": "/path/to/output.png"
}
```

### Video Generator
```json
POST /generate
{
  "start_frame": "/path/to/start.png",
  "end_frame": "/path/to/end.png",
  "output_path": "/path/to/output.mp4",
  "num_frames": 14,
  "fps": 7
}

Response:
{
  "success": true,
  "output_path": "/path/to/output.mp4",
  "num_frames": 14,
  "fps": 7
}
```

## Deployment Configurations

### Docker Compose (Production)
```yaml
services:
  image-generator:
    # NVIDIA GPU
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    
  # OR for Apple Silicon (no GPU config needed)
```

### Manual (Development)
```bash
# Terminal 1: AI Services
python services/image-generator/server.py &
python services/video-generator/server.py &
python services/audio-analyzer/server.py &

# Terminal 2: Main Server
USE_AI_SERVICES=true ./lumiere
```

## Performance Characteristics

### With NVIDIA GPU (4GB VRAM)
```
Audio Analysis:    2-5 seconds
Image Generation:  3-5 seconds × 7 = 21-35 seconds
Video Generation:  30-60 seconds × 6 = 3-6 minutes
Composition:       5-10 seconds
─────────────────────────────────────────────
Total:             ~5-7 minutes for 3-min song
```

### With Apple Silicon (16GB Unified Memory)
```
Audio Analysis:    2-5 seconds
Image Generation:  4-7 seconds × 7 = 28-49 seconds
Video Generation:  40-70 seconds × 6 = 4-7 minutes
Composition:       5-10 seconds
─────────────────────────────────────────────
Total:             ~6-9 minutes for 3-min song
```

### CPU Only (8GB RAM)
```
Audio Analysis:    2-5 seconds
Image Generation:  30-60 seconds × 7 = 3.5-7 minutes
Video Generation:  10 seconds × 6 = 1 minute (interpolation)
Composition:       5-10 seconds
─────────────────────────────────────────────
Total:             ~5-9 minutes for 3-min song
```

## Error Handling & Resilience

### Service Unavailable
```go
// Go server automatically falls back
if !serviceHealthy {
    log.Warning("Service unavailable, using fallback")
    return fallbackImplementation()
}
```

### Out of Memory
```python
# Python services catch OOM
try:
    generate_image()
except RuntimeError as e:
    if "out of memory" in str(e):
        # Clear cache and retry with lower settings
        torch.cuda.empty_cache()
        return generate_with_lower_settings()
```

### Timeout
```go
// Configurable timeouts
client := &http.Client{
    Timeout: 300 * time.Second,  // 5 minutes for video
}
```

## Monitoring & Observability

### Logs
```bash
# Service logs
docker-compose logs -f image-generator

# Main server logs
./lumiere 2>&1 | tee lumiere.log
```

### Metrics
```bash
# GPU utilization
nvidia-smi -l 1

# Service health
curl http://localhost:5001/health
curl http://localhost:5002/health
curl http://localhost:5003/health
```

### Resource Usage
```bash
# Docker stats
docker stats

# System resources
htop
```

## Security Considerations

1. **No External Network Calls**: All AI models run locally
2. **File Path Validation**: Services validate all file paths
3. **Resource Limits**: Docker containers have memory/CPU limits
4. **Input Sanitization**: Prompts and paths are sanitized
5. **Service Isolation**: Each service runs in separate container

## Scaling Strategies

### Horizontal Scaling
- Run multiple instances of each service
- Use load balancer (nginx, traefik)
- Share file system (NFS, S3)

### Vertical Scaling
- Upgrade GPU (more VRAM)
- More CPU cores for parallel processing
- More RAM for larger models

### Queue-based Processing
- Add Redis/RabbitMQ for job queue
- Multiple workers process jobs
- Better resource utilization

## Future Enhancements

1. **Model Caching**: Keep models in memory between requests
2. **Batch Processing**: Generate multiple images in one batch
3. **LoRA Support**: Fine-tuned models for specific styles
4. **Model Switching**: Multiple models for different styles
5. **Real-time Monitoring**: Grafana dashboard for metrics
6. **A/B Testing**: Compare different models/settings
