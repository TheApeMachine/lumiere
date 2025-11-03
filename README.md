# Lumiere - AI Music Video Generator

An AI-driven music video generator that transforms your music into stunning visual experiences.

## Overview

Lumiere uses a sophisticated AI pipeline to create music videos:

1. **Upload**: Provide MP3 audio, optional character images, and text prompts
2. **Concept Generation**: AI analyzes music (intensity, rhythm, lyrics) and creates video concepts
3. **Visual Seeding**: AI generates keyframe images at strategic moments
4. **Animation**: AI animates between keyframes with smooth transitions
5. **Final Composition**: Combines all segments into a complete music video

## Features

- 🎵 **Audio Analysis**: Automatically detects intensity, tempo, and rhythm patterns
- 🎨 **AI Image Generation**: Creates visually coherent keyframes based on music and prompts
- 🎬 **AI Video Animation**: Smooth transitions between keyframes synchronized with music
- 👤 **Character Consistency**: Optional character images for consistent character generation
- 🔄 **RESTful API**: Easy integration with web and mobile applications

## Installation

### Prerequisites

- Go 1.21 or higher

### Build from Source

```bash
git clone https://github.com/TheApeMachine/lumiere.git
cd lumiere
go mod download
go build -o lumiere .
```

## Usage

### Starting the Server

```bash
./lumiere
```

The server will start on port 8080 (configurable via `PORT` environment variable).

### Environment Variables

- `PORT`: Server port (default: 8080)
- `UPLOAD_DIR`: Directory for uploaded files (default: ./uploads)
- `OUTPUT_DIR`: Directory for generated outputs (default: ./outputs)
- `CONCEPT_GENERATION_ENABLED`: Enable concept generation (default: true)
- `VISUAL_SEEDING_ENABLED`: Enable visual seeding (default: true)
- `ANIMATION_ENABLED`: Enable animation (default: true)

### API Endpoints

#### Health Check

```bash
curl http://localhost:8080/health
```

#### Create Project

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

#### Get Project Status

```bash
curl http://localhost:8080/api/v1/projects/{project-id}
```

#### List All Projects

```bash
curl http://localhost:8080/api/v1/projects
```

#### Process Project

Start the AI pipeline for a project:

```bash
curl -X POST http://localhost:8080/api/v1/projects/{project-id}/process
```

## Pipeline Architecture

### 1. Concept Generation

The AI analyzes the audio file to:
- Detect intensity curves and tempo changes
- Identify key moments (intro, build-ups, climaxes, bridges)
- Extract lyrics (if present)
- Generate creative visual prompts for each moment

### 2. Visual Seeding

For each key moment, the AI generates:
- High-quality keyframe images
- Consistent visual style
- Character consistency (if character images provided)
- Scene composition based on intensity

### 3. Animation

The AI creates:
- Smooth video transitions between keyframes
- Synchronized motion with audio
- Dynamic camera movements
- Visual effects matching intensity

### 4. Composition

Final video assembly:
- Concatenates all animation segments
- Adds original audio track
- Applies color grading and effects
- Exports in standard video format

## Project Structure

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

## Integration Notes

### Current Implementation

This is a **framework implementation** that provides:
- Complete API structure
- Pipeline architecture
- Data models and workflows
- File management

### Production Integration

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

## Development

### Running Tests

```bash
go test ./...
```

### Code Formatting

```bash
go fmt ./...
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Roadmap

- [ ] Integrate actual AI image generation APIs
- [ ] Integrate actual AI video generation APIs
- [ ] Add audio analysis with beat detection
- [ ] Implement lyrics extraction
- [ ] Add web UI for easier interaction
- [ ] Support multiple video styles
- [ ] Add batch processing
- [ ] Implement caching for faster regeneration
- [ ] Add preview generation
- [ ] Support custom transitions
