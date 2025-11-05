# Lumiere - AI Music Video Generator 🎥

An AI-driven music video generator that transforms your music into stunning visual experiences using **local AI models** that run on consumer hardware.

## Overview 🧭

Lumiere uses a sophisticated AI pipeline to create music videos:

1. **Upload**: Provide MP3 audio, optional character images, and text prompts
2. **Concept Generation**: AI analyzes music (intensity, rhythm, lyrics) and creates video concepts
3. **Visual Seeding**: AI generates keyframe images at strategic moments
4. **Animation**: AI animates between keyframes with smooth transitions
5. **Final Composition**: Combines all segments into a complete music video

## Features ✨

### Core Features 🔧
- 🎵 **Audio Analysis**: Uses librosa to detect tempo, beats, and intensity patterns
- 🎨 **Local AI Image Generation**: Stable Diffusion v1.5 running on your hardware
- 🎬 **Local AI Video Animation**: LTX-Video model with frame interpolation fallback
- 💻 **Consumer Hardware**: Optimized for CUDA (NVIDIA) and Metal (Apple Silicon)
- 🚀 **CPU Fallback**: Works on any hardware with graceful degradation
- 👤 **Character Consistency**: Optional character images for consistent character generation
- 🔄 **RESTful API**: Easy integration with web and mobile applications

### Enhanced Features (New!) 🚀
- 🎼 **Beat Detection & Audio Analysis**: Precise beat and downbeat timestamps for perfect sync
- 🎬 **Automatic Transition Parameters**: Smart selection of cuts, fades, and camera movements
- 📊 **Variable Seed Density**: More visual cuts in high-energy sections, fewer in calm parts
- 👁️ **Character Face Preservation**: Maintain character likeness across all frames
- ✅ **Quality Control**: Automated validation of generated images and videos
- ⚙️ **Configurable Output**: Flexible resolution, FPS, and quality settings

See [FEATURES.md](FEATURES.md) for detailed documentation on all enhanced features.

## Installation 🛠️

### Prerequisites 📋

- Go 1.21 or higher
- Python 3.8+ (for AI services)
- NVIDIA GPU with CUDA (optional, recommended)
- OR Apple Silicon Mac (Metal support)
- OR CPU only (works but slower)

### Quick Start ⚡

```bash
git clone https://github.com/TheApeMachine/lumiere.git
cd lumiere

# Build Go server
go mod download
go build -o lumiere .

# Setup AI services (one-time)
./setup-ai-services.sh
```

### Option 1: Docker Compose (Easiest) 🐳

```bash
# Start all services including AI models
docker-compose up -d

# Check status
docker-compose logs -f
```

### Option 2: Manual Setup 🧰

```bash
# Terminal 1: Start AI services
source venv/bin/activate
python services/image-generator/server.py &
python services/video-generator/server.py &
python services/audio-analyzer/server.py &

# Terminal 2: Start main server with AI enabled
USE_AI_SERVICES=true ./lumiere
```

## Usage ▶️

### Starting the Server 🟢

**With AI Services (Recommended):**
```bash
USE_AI_SERVICES=true ./lumiere
```

**Without AI Services (Simulation Mode):**
```bash
./lumiere
```

The server will start on port 8080 (configurable via `PORT` environment variable).

### Environment Variables ⚙️

#### Main Server 🖥️
- `PORT`: Server port (default: 8080)
- `USE_AI_SERVICES`: Enable local AI models (default: true)
- `IMAGE_SERVICE_URL`: Image service URL (default: http://localhost:5001)
- `VIDEO_SERVICE_URL`: Video service URL (default: http://localhost:5002)
- `AUDIO_SERVICE_URL`: Audio service URL (default: http://localhost:5003)
- `UPLOAD_DIR`: Directory for uploaded files (default: ./uploads)
- `OUTPUT_DIR`: Directory for generated outputs (default: ./outputs)
- `MAX_UPLOAD_SIZE_MB`: Maximum upload size in megabytes (default: 100)
- `CONCEPT_GENERATION_ENABLED`: Enable concept generation (default: true)
- `VISUAL_SEEDING_ENABLED`: Enable visual seeding (default: true)
- `ANIMATION_ENABLED`: Enable animation (default: true)

#### AI Services 🤖
- `IMAGE_SERVICE_PORT`: Image generator port (default: 5001)
- `VIDEO_SERVICE_PORT`: Video generator port (default: 5002)
- `AUDIO_SERVICE_PORT`: Audio analyzer port (default: 5003)

### API Endpoints 🔗

#### Health Check 🩺

```bash
curl http://localhost:8080/health
```

#### Create Project 🆕

```bash
curl -X POST http://localhost:8080/api/v1/projects \
  -F "audio=@your-music.mp3" \
  -F "prompt=cyberpunk cityscape at night with neon lights" \
  -F "character_images=@character1.png" \
  -F "character_images=@character2.png"
```

Response:
```json
{
  "id": "project-uuid",
  "audio_file": "/path/to/audio.mp3",
  "prompt": "cyberpunk cityscape at night with neon lights",
  "status": "created",
  "created_at": "2025-11-03T21:00:00Z",
  "character_images": ["/path/to/character1.png", "/path/to/character2.png"]
}
```

#### Get Project Status 📊

```bash
curl http://localhost:8080/api/v1/projects/{project-id}
```

#### List All Projects 📋

```bash
curl http://localhost:8080/api/v1/projects
```

#### Process Project ⚙️

Start the AI pipeline for a project:

```bash
curl -X POST http://localhost:8080/api/v1/projects/{project-id}/process
```

## Hardware Requirements 🖥️

### Minimum (CPU Only) 🧠
- **CPU**: Any modern CPU
- **RAM**: 8GB
- **Disk**: 10GB free space
- **Speed**: ~60s per image, frame interpolation for video
- **Use Case**: Testing and development

### Recommended (GPU Acceleration) ⚡
- **GPU**: NVIDIA GPU with 4GB+ VRAM (CUDA 11+)
  - OR Apple Silicon (M1/M2/M3 with Metal)
- **RAM**: 16GB
- **Disk**: 15GB free space
- **Speed**: ~3s per image, ~30-60s per video segment
- **Use Case**: Production use

### Optimal (High Performance) 🚀
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **RAM**: 32GB
- **Disk**: 20GB free space
- **Speed**: ~2s per image, ~20-30s per video segment
- **Use Case**: High-volume production

## Pipeline Architecture 🏗️

### 1. Concept Generation (Audio Analyzer Service) 🎼

**Technology**: librosa (Python)

The AI analyzes the audio file to:
- Detect real tempo and beats using librosa
- Calculate intensity curves with 1-second granularity
- Identify peak moments in the music
- Select 7 strategic key moments throughout the song
- Generate creative visual prompts for each moment

### 2. Visual Seeding (Image Generator Service) 🖼️

**Technology**: Stable Diffusion v1.5

For each key moment, the AI generates:
- 512x512 keyframe images (customizable)
- Consistent visual style across all frames
- Character consistency (if character images provided)
- Scene composition based on music intensity
- ~20 inference steps for fast generation

### 3. Animation (Video Generator Service) 🎞️

**Technology**: Stable Video Diffusion / Frame Interpolation

The AI creates:
- Smooth video transitions between keyframes
- 14 frames per segment at 7 FPS (adjustable)
- Synchronized motion with audio timing
- GPU-accelerated when available
- CPU fallback with simple interpolation

### 4. Composition 🎛️

Final video assembly:
- Concatenates all animation segments
- Adds original audio track
- Applies timing synchronization
- Exports in MP4 format

## Project Structure 🗂️

```
lumiere/
├── main.go              # Application entry point
├── config/              # Configuration management
│   └── config.go
├── models/              # Data models
│   └── models.go
├── pipeline/            # AI pipeline components
│   ├── pipeline.go      # Pipeline orchestration
│   ├── concept.go       # Concept generation
│   ├── visual.go        # Visual seeding
│   └── animation.go     # Animation generation
└── api/                 # REST API
    └── api.go
```

## Integration Notes 🔌

### Current Implementation 📦

This is a **framework implementation** that provides:
- Complete API structure
- Pipeline architecture
- Data models and workflows
- File management

### Production Integration 🏭

For production use, integrate with:

**Image Generation:**
- Stable Diffusion API
- DALL-E API
- Midjourney API

**Video Generation:**
- Runway Gen-2
- Pika Labs API
- Stable Video Diffusion

**Audio Analysis:**
- Librosa (Python)
- Essentia
- FFmpeg audio analysis

**Video Composition:**
- FFmpeg for video concatenation
- Video editing libraries

## Development 🧑‍💻

### Running Tests 🧪

```bash
go test ./...
```

### Code Formatting 🧹

```bash
go fmt ./...
```

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

MIT License

## Roadmap 🗺️

- [ ] Integrate actual AI image generation
      - [x] Local image generation
      - [ ] Remote service image generation
- [ ] Integrate actual AI video generation
      - [x] Local video generation
      - [ ] Remote service video generation
- [x] Add audio analysis with beat detection
- [ ] Implement lyrics extraction
- [ ] Add web UI for easier interaction
- [ ] Support multiple video styles
- [ ] Add batch processing
- [ ] Implement caching for faster regeneration
- [ ] Add preview generation
- [ ] Support custom transitions
