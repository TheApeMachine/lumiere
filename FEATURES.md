# Lumiere Enhanced Features

This document describes the enhanced features added to the Lumiere video generation pipeline based on open-source best practices and recommendations from the video generation community.

## Overview

Lumiere now includes several advanced features to improve video quality, audio-visual synchronization, and user control over the generation process. These enhancements align with approaches used by leading open-source video generation tools like Open-Sora and Wan 2.1.

## Key Features

### 1. Beat Detection and Audio Analysis

The pipeline now includes detailed audio analysis with beat detection capabilities:

**Features:**
- **Beat Timestamps**: Precise detection of beat locations throughout the audio
- **Downbeat Detection**: Identification of strong beats (typically first beat in a measure)
- **Onset Strengths**: Intensity measurement for each detected beat
- **Tempo Detection**: BPM (beats per minute) calculation
- **Spectral Analysis**: Optional spectral centroid and zero-crossing rate analysis

**Configuration:**
The audio analysis is automatically generated when creating a concept. No additional configuration is required.

**Benefits:**
- Better synchronization between visual cuts and musical beats
- Scene changes aligned with downbeats for more natural transitions
- Intensity-aware key moment generation

**Example Output:**
```json
{
  "audio_analysis": {
    "duration": 180.0,
    "tempo": 120.0,
    "beats": [0.0, 0.5, 1.0, 1.5, ...],
    "downbeats": [0.0, 2.0, 4.0, 6.0, ...],
    "onset_strengths": [0.8, 0.6, 0.9, ...]
  }
}
```

### 2. Transition Parameters

Key moments and animations now include detailed transition parameters:

**Key Moment Parameters:**
- **Transition Style**: Type of transition (cut, fade, dissolve, zoom)
- **Cut Frequency**: How often cuts occur (slow, medium, fast)
- **Motion Intensity**: Level of motion in the segment (low, medium, high)
- **Camera Movement**: Camera motion style (static, pan, zoom, dolly)

**Automatic Determination:**
Transition parameters are automatically determined based on audio intensity:
- **Low Intensity** (< 0.3): Slow fades, low motion, static camera
- **Medium Intensity** (0.3-0.6): Dissolves, moderate motion, panning camera
- **High Intensity** (> 0.6): Fast cuts, high motion, dynamic camera movements

**Example:**
```json
{
  "timestamp": 45.0,
  "intensity": 0.75,
  "transition_style": "cut",
  "cut_frequency": "fast",
  "motion_intensity": "high",
  "camera_movement": "zoom"
}
```

### 3. Variable Seed Density Based on Intensity

The visual seeding process now adapts to audio intensity:

**How It Works:**
- Base seed density (default: 2 seeds per minute)
- Intensity multiplier (default: 2.0x at maximum intensity)
- High-intensity sections get more seeds → more visual variety
- Low-intensity sections get fewer seeds → longer, contemplative shots

**Configuration:**
```bash
# Environment variables
BASE_SEED_DENSITY=2.0           # Base seeds per minute
INTENSITY_MULTIPLIER=2.0        # Multiplier for high-intensity sections
```

**Benefits:**
- Chorus sections get rapid cuts and visual variety
- Verse sections have longer, more stable shots
- Better matches the energy of the music

### 4. Character Face Preservation

When character images are uploaded, the system now supports face preservation:

**Features:**
- Primary character image used as reference
- Face features embedded in generation prompts
- Character reference tracking for all visual seeds
- Consistent character appearance across frames

**Configuration:**
```bash
ENABLE_FACE_PRESERVATION=true
```

**Usage:**
Upload character images when creating a project. The first image is used as the primary reference:
```bash
curl -X POST http://localhost:8080/api/v1/projects \
  -F "prompt=Epic music video" \
  -F "audio=@song.mp3" \
  -F "character_images=@character.jpg"
```

**Implementation Note:**
In production, this would integrate with:
- Face recognition models (e.g., InsightFace)
- ControlNet for face preservation in Stable Diffusion
- Character consistency models

### 5. Quality Control and Validation

Automated quality control checks for generated content:

**Image Quality Checks:**
- Blur/sharpness detection
- Artifact detection
- Composition validation
- Color balance verification

**Video Quality Checks:**
- Frozen frame detection
- Extreme artifact detection
- Motion smoothness verification
- Audio-visual sync validation

**Configuration:**
```bash
ENABLE_QUALITY_CONTROL=true
MIN_QUALITY_SCORE=0.5          # Minimum acceptable score (0-1)
```

**Quality Scores:**
Each generated seed and animation receives a quality score (0-1) and validation status:
- `pass`: Quality score above minimum
- `needs_review`: Quality score below minimum
- `fail`: Critical quality issues detected

**Example Output:**
```json
{
  "id": "seed_123",
  "quality_score": 0.85,
  "validation_status": "pass"
}
```

### 6. Configurable Video Output

Flexible configuration for output video parameters:

**Video Settings:**
```bash
# Resolution presets
VIDEO_RESOLUTION=720p           # Options: 720p, 1080p, 512x512
VIDEO_WIDTH=1280               # Custom width in pixels
VIDEO_HEIGHT=720               # Custom height in pixels
VIDEO_FPS=24                   # Frames per second

# Quality settings
ENABLE_QUALITY_CONTROL=true
MIN_QUALITY_SCORE=0.5
```

**Common Configurations:**

**Fast Draft Mode (Consumer Hardware):**
```bash
VIDEO_WIDTH=512
VIDEO_HEIGHT=512
VIDEO_FPS=7
ENABLE_QUALITY_CONTROL=false
```

**High Quality (Production):**
```bash
VIDEO_WIDTH=1920
VIDEO_HEIGHT=1080
VIDEO_FPS=30
ENABLE_QUALITY_CONTROL=true
MIN_QUALITY_SCORE=0.7
```

**Balanced (Recommended):**
```bash
VIDEO_WIDTH=1280
VIDEO_HEIGHT=720
VIDEO_FPS=24
ENABLE_QUALITY_CONTROL=true
MIN_QUALITY_SCORE=0.5
```

## Configuration Reference

### Complete Environment Variables

```bash
# Server Configuration
PORT=8080
UPLOAD_DIR=./uploads
OUTPUT_DIR=./outputs
MAX_UPLOAD_SIZE_MB=100

# Pipeline Stages
CONCEPT_GENERATION_ENABLED=true
VISUAL_SEEDING_ENABLED=true
ANIMATION_ENABLED=true

# Video Generation Settings
VIDEO_RESOLUTION=720p
VIDEO_FPS=24
VIDEO_WIDTH=1280
VIDEO_HEIGHT=720

# Quality Control
ENABLE_QUALITY_CONTROL=true
MIN_QUALITY_SCORE=0.5
ENABLE_FACE_PRESERVATION=true

# Seed Density
BASE_SEED_DENSITY=2.0
INTENSITY_MULTIPLIER=2.0

# AI Services (if using external services)
USE_AI_SERVICES=false
IMAGE_SERVICE_URL=http://localhost:5001
VIDEO_SERVICE_URL=http://localhost:5002
AUDIO_SERVICE_URL=http://localhost:5003
```

## Usage Examples

### Basic Usage with Enhanced Features

```bash
# 1. Start the server
./lumiere

# 2. Create a project with character image
curl -X POST http://localhost:8080/api/v1/projects \
  -F "prompt=Cinematic music video with epic landscapes" \
  -F "audio=@mysong.mp3" \
  -F "character_images=@myface.jpg"

# 3. Process the project
curl -X POST http://localhost:8080/api/v1/projects/{project_id}/process

# 4. Check status
curl http://localhost:8080/api/v1/projects/{project_id}
```

### Response with Enhanced Features

```json
{
  "id": "project_123",
  "status": "completed",
  "concept": {
    "description": "AI-generated music video concept...",
    "key_moments": [
      {
        "timestamp": 0.0,
        "intensity": 0.3,
        "transition_style": "fade",
        "cut_frequency": "slow",
        "motion_intensity": "low",
        "camera_movement": "static"
      }
    ],
    "audio_analysis": {
      "duration": 180.0,
      "tempo": 128.5,
      "beats": [0.0, 0.468, 0.936, ...],
      "downbeats": [0.0, 1.872, 3.744, ...]
    }
  },
  "visual_seeds": [
    {
      "id": "seed_1",
      "timestamp": 0.0,
      "character_reference": "/uploads/project_123/character_0.png",
      "quality_score": 0.87,
      "validation_status": "pass"
    }
  ],
  "animations": [
    {
      "id": "anim_1",
      "start_time": 0.0,
      "end_time": 5.0,
      "transition_style": "fade",
      "motion_intensity": "low",
      "camera_movement": "static",
      "quality_score": 0.92,
      "validation_status": "pass"
    }
  ]
}
```

## Integration with External Services

The enhanced features work seamlessly with external AI services:

### Audio Analysis Service

```python
# Expected API endpoint: POST /analyze
{
  "audio_path": "/path/to/audio.mp3"
}

# Response:
{
  "success": true,
  "analysis": {
    "duration": 180.0,
    "tempo": 120.0,
    "beats": [...],
    "downbeats": [...],
    "intensity_curve": [...],
    "key_moments": [...]
  }
}
```

### Image Generation Service

```python
# Expected API endpoint: POST /generate
{
  "prompt": "cinematic landscape...",
  "output_path": "/path/to/output.png",
  "width": 1280,
  "height": 720,
  "character_reference": "/path/to/character.jpg"  # Optional
}

# Response:
{
  "success": true,
  "output_path": "/path/to/output.png"
}
```

### Video Generation Service

```python
# Expected API endpoint: POST /generate
{
  "start_frame": "/path/to/start.png",
  "end_frame": "/path/to/end.png",
  "output_path": "/path/to/output.mp4",
  "num_frames": 48,
  "fps": 24,
  "transition": "fade",
  "motion": "low",
  "camera": "static"
}

# Response:
{
  "success": true,
  "output_path": "/path/to/output.mp4"
}
```

## Performance Considerations

### Resource Requirements

**Minimum (Draft Quality):**
- CPU: 4 cores
- RAM: 8 GB
- GPU: Optional (CPU fallback available)
- Disk: 10 GB free space

**Recommended (High Quality):**
- CPU: 8+ cores
- RAM: 16 GB
- GPU: 10+ GB VRAM (RTX 3080 or better)
- Disk: 50 GB free space

### Processing Time Estimates

For a 3-minute song:
- **Concept Generation**: < 1 second (simulated), 5-10 seconds (with AI audio analysis)
- **Visual Seed Generation**: 2-5 minutes (with AI, depends on seed count)
- **Animation Generation**: 10-30 minutes (with AI, depends on quality settings)
- **Final Composition**: < 1 minute

**Note**: Times vary significantly based on hardware and quality settings.

## Future Enhancements

Planned improvements:
1. Advanced character motion (walking, dancing, facial expressions)
2. Multi-character support with relationship tracking
3. Style consistency across frames using LoRA
4. Audio reactivity for particle effects and color grading
5. Scene understanding and semantic transitions
6. Real-time preview generation
7. GPU-optimized batch processing

## Troubleshooting

### Low Quality Scores

If many seeds/animations fail quality control:
- Lower MIN_QUALITY_SCORE threshold
- Adjust generation parameters
- Use higher resolution settings
- Increase inference steps (if using AI services)

### Performance Issues

If generation is too slow:
- Lower resolution (e.g., 512x512)
- Reduce FPS (e.g., 7 fps)
- Decrease BASE_SEED_DENSITY
- Disable ENABLE_QUALITY_CONTROL for faster processing

### Character Inconsistency

If character appearance varies:
- Ensure character reference image is high quality
- Use frontal face photos for best results
- Enable ENABLE_FACE_PRESERVATION
- Consider using ControlNet or similar for better preservation

## Contributing

To contribute enhancements:
1. Follow existing code patterns
2. Add tests for new features
3. Update documentation
4. Ensure backward compatibility

## References

- [Open-Sora Documentation](https://github.com/hpcaitech/Open-Sora)
- [Stable Video Diffusion Paper](https://stability.ai/research/stable-video-diffusion)
- [Librosa Audio Analysis](https://librosa.org/)
- [ControlNet for Character Preservation](https://github.com/lllyasviel/ControlNet)
