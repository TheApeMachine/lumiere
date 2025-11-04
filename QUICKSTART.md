# Quick Start Guide

Get Lumiere running with local AI models in minutes!

## Choose Your Setup Method

### Option 1: Docker Compose (Easiest - Recommended)

```bash
# 1. Clone and build
git clone https://github.com/TheApeMachine/lumiere.git
cd lumiere

# 2. Start everything with Docker
docker-compose up -d

# 3. Check that services are running
docker-compose ps

# 4. Test the API
curl http://localhost:8080/health
```

**That's it!** All AI services and the main server are now running.

### Option 2: Local Installation (More Control)

```bash
# 1. Clone the repository
git clone https://github.com/TheApeMachine/lumiere.git
cd lumiere

# 2. Build the Go server
go mod download
go build -o lumiere .

# 3. Setup Python AI services (one-time)
./setup-ai-services.sh

# 4. Start AI services (in background)
source venv/bin/activate
python services/image-generator/server.py &
python services/video-generator/server.py &
python services/audio-analyzer/server.py &

# 5. Start main server with AI enabled
USE_AI_SERVICES=true ./lumiere
```

## Your First Music Video

### Step 1: Prepare Your Files

```bash
# You need:
# - An MP3 audio file
# - (Optional) Character images for consistency

# Example:
cp ~/Music/song.mp3 ./test_audio.mp3
```

### Step 2: Create a Project

```bash
curl -X POST http://localhost:8080/api/v1/projects \
  -F "audio=@test_audio.mp3" \
  -F "prompt=Epic cinematic journey through mystical forests with ethereal light"
```

**Response:**
```json
{
  "id": "abc123...",
  "status": "created",
  ...
}
```

Copy the project ID from the response.

### Step 3: Start Processing

```bash
# Replace {PROJECT_ID} with your actual project ID
curl -X POST http://localhost:8080/api/v1/projects/{PROJECT_ID}/process
```

### Step 4: Monitor Progress

```bash
# Check status
curl http://localhost:8080/api/v1/projects/{PROJECT_ID}

# You'll see status change:
# created -> processing -> concept_generated -> seeds_generated -> animations_generated -> completed
```

### Step 5: Get Your Video

Once status is `completed`:

```bash
# View project details to find output paths
curl http://localhost:8080/api/v1/projects/{PROJECT_ID}

# Your video is in the outputs directory:
# outputs/{PROJECT_ID}/final_video.mp4
```

## Performance Expectations

### With GPU (NVIDIA or Apple Silicon)
- **Audio Analysis**: ~2-5 seconds
- **Per Image (7 total)**: ~3-5 seconds each = ~30 seconds
- **Per Video Segment (6 total)**: ~30-60 seconds each = ~5 minutes
- **Total Time**: ~6-7 minutes for a 3-minute song

### With CPU Only
- **Audio Analysis**: ~2-5 seconds
- **Per Image (7 total)**: ~30-60 seconds each = ~5 minutes
- **Per Video Segment (6 total)**: ~10 seconds each (interpolation) = ~1 minute
- **Total Time**: ~7-8 minutes for a 3-minute song

## Troubleshooting

### Services Not Starting

**Check logs:**
```bash
# Docker
docker-compose logs image-generator

# Local
tail -f services/image-generator/server.py.log
```

**Common issues:**
- Out of memory: Reduce image size in settings
- CUDA not found: Check `nvidia-smi` output
- Port already in use: Change ports in `.env`

### Slow Performance

**For GPU users:**
```bash
# Check GPU is being used
nvidia-smi  # For NVIDIA
python -c "import torch; print(torch.backends.mps.is_available())"  # For Apple Silicon
```

**If GPU not detected:**
- Verify CUDA installation: `nvidia-smi`
- Verify PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
- For Apple Silicon, ensure PyTorch 2.0+

### Out of Memory

**Reduce memory usage:**

1. Edit `services/image-generator/server.py`:
   ```python
   # Change from 512x512 to 256x256
   width = data.get('width', 256)
   height = data.get('height', 256)
   ```

2. Reduce inference steps:
   ```python
   num_inference_steps = data.get('num_inference_steps', 15)  # Was 20
   ```

3. Restart services

### API Not Responding

**Check service health:**
```bash
curl http://localhost:5001/health  # Image generator
curl http://localhost:5002/health  # Video generator
curl http://localhost:5003/health  # Audio analyzer
curl http://localhost:8080/health  # Main server
```

**If services are down:**
```bash
# Docker
docker-compose restart

# Local
pkill -f "python services"  # Kill all Python services
# Then restart them
```

## Next Steps

- **Experiment with prompts**: Try different creative descriptions
- **Add character images**: Upload reference images for consistent characters
- **Adjust settings**: Modify image size, inference steps, etc.
- **Read the docs**: See README.md for detailed configuration options
- **Check examples**: Run `./examples/demo.sh` for a full demonstration

## Getting Help

- **Documentation**: See README.md and services/README.md
- **Issues**: Report bugs on GitHub
- **Architecture**: See ARCHITECTURE.md for technical details

## Hardware-Specific Notes

### NVIDIA GPUs
- Works with CUDA 11.0 or higher
- Recommended: 4GB+ VRAM
- Models automatically use CUDA when available

### Apple Silicon (M1/M2/M3)
- Automatically uses Metal Performance Shaders (MPS)
- Recommended: 16GB+ unified memory
- Performance comparable to entry-level NVIDIA GPUs

### CPU Only
- Works on any modern CPU
- Significantly slower than GPU
- Uses optimized CPU operations
- Frame interpolation for video (fast but lower quality)

## Production Tips

1. **Use GPU**: Essential for reasonable performance
2. **Monitor memory**: Watch GPU/RAM usage
3. **Batch process**: Queue multiple projects
4. **Adjust quality**: Balance quality vs. speed
5. **Regular updates**: Pull latest improvements

Enjoy creating AI music videos! 🎵🎨🎬
