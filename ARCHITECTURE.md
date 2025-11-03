# Lumiere Architecture

## Overview

Lumiere is an AI-driven music video generator built in Go that provides a complete framework for transforming music into visual experiences. The architecture is designed to be modular, extensible, and ready for integration with production AI services.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                          │
│  (Web UI, Mobile Apps, CLI Tools, Third-party Services)     │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP/REST API
┌──────────────────────▼──────────────────────────────────────┐
│                      API Server (Gin)                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Handlers: Upload, Status, Process, List Projects     │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   Pipeline Orchestrator                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Coordinates all pipeline stages                       │ │
│  │  Manages project state and workflow                    │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────┬───────────┬───────────┬──────────────────────┘
               │           │           │
       ┌───────▼──┐  ┌────▼────┐  ┌──▼──────┐
       │ Concept  │  │ Visual  │  │Animation│
       │Generator │  │ Seeder  │  │Generator│
       └──────────┘  └─────────┘  └─────────┘
```

## Component Details

### 1. API Server (`api/api.go`)

**Responsibilities:**
- Handle HTTP requests and responses
- File upload management (audio, images)
- Project lifecycle management
- Asynchronous processing coordination

**Endpoints:**
- `GET /health` - Health check
- `POST /api/v1/projects` - Create new project with uploads
- `GET /api/v1/projects/:id` - Get project status and details
- `GET /api/v1/projects` - List all projects
- `POST /api/v1/projects/:id/process` - Start pipeline processing

**Key Features:**
- Multipart form handling for file uploads
- Configurable upload size limits
- Thread-safe project storage with sync.RWMutex
- Asynchronous pipeline execution

### 2. Pipeline Orchestrator (`pipeline/pipeline.go`)

**Responsibilities:**
- Coordinate execution of pipeline stages
- Manage project state transitions
- Error handling and recovery
- Progress tracking

**Workflow:**
```
Created → Concept Generated → Seeds Generated → Animations Generated → Completed
                    ↓              ↓                   ↓
                 Failed         Failed              Failed
```

**State Management:**
- `created` - Initial state after upload
- `processing` - Pipeline is running
- `concept_generated` - Step 1 complete
- `seeds_generated` - Step 2 complete
- `animations_generated` - Step 3 complete
- `completed` - All steps successful
- `failed` - Error occurred

### 3. Concept Generator (`pipeline/concept.go`)

**Responsibilities:**
- Analyze audio characteristics
- Generate intensity curves
- Identify key moments
- Create visual prompts

**Output:**
- Video concept with description
- 7 key moments with timestamps and prompts
- Continuous intensity curve (1-second intervals)
- Optional lyrics segments

**Algorithm:**
- Simulates audio analysis (production: use audio processing libs)
- Creates realistic intensity curves using sine waves + noise
- Strategic keyframe placement (intro, build, climax, bridge, peak, resolution)
- Combines user prompt with scene descriptions

### 4. Visual Seeder (`pipeline/visual.go`)

**Responsibilities:**
- Generate keyframe images for each key moment
- Maintain visual consistency
- Handle character references

**Output:**
- One seed image per key moment
- Image metadata (prompt, timestamp, ID)
- File storage management

**Integration Points:**
- Stable Diffusion API
- DALL-E API
- Midjourney API
- Custom image generation services

### 5. Animation Generator (`pipeline/animation.go`)

**Responsibilities:**
- Create video segments between keyframes
- Ensure smooth transitions
- Synchronize with audio
- Compose final video

**Output:**
- Animation segments for each transition
- Metadata (duration, frame references)
- Final composed video

**Integration Points:**
- Runway Gen-2 API
- Pika Labs API
- Stable Video Diffusion
- Custom video generation services

## Data Models (`models/models.go`)

### Project
The main entity that tracks a video generation job:
```go
- ID: Unique identifier
- AudioFile: Path to uploaded audio
- Prompt: User's creative direction
- Status: Current pipeline state
- CharacterImages: Optional character references
- Concept: Generated concept
- VisualSeeds: Generated keyframes
- Animations: Video segments
- FinalVideo: Composed output
```

### Concept
AI-generated video plan:
```go
- Description: Overall concept
- KeyMoments: Strategic points in timeline
- Intensity: Audio energy curve
- Lyrics: Optional lyric segments
- Metadata: Additional info
```

### VisualSeed
Keyframe image:
```go
- ID: Unique identifier
- Timestamp: Position in audio
- Prompt: Generation prompt
- ImagePath: Stored file location
- KeyMomentID: Reference to concept
```

### Animation
Video segment:
```go
- ID: Unique identifier
- StartTime/EndTime: Duration
- VideoPath: Stored file location
- FirstFrameID/LastFrameID: Transition frames
```

## Configuration (`config/config.go`)

### Constants
```go
DefaultFilePerms = 0644  // File permissions
DefaultDirPerms = 0755   // Directory permissions
DefaultMaxUploadSizeMB = 100  // Upload limit
```

### Environment Variables
All configuration is environment-driven for 12-factor app compliance:
- Server settings (port)
- Storage paths (uploads, outputs)
- Upload limits
- Feature flags (enable/disable pipeline stages)

## File Organization

```
lumiere/
├── main.go              # Entry point, server initialization
├── config/
│   └── config.go        # Configuration management
├── api/
│   └── api.go          # HTTP handlers and routing
├── models/
│   └── models.go       # Data structures
├── pipeline/
│   ├── pipeline.go     # Orchestration
│   ├── concept.go      # Audio analysis & concept generation
│   ├── visual.go       # Image generation
│   └── animation.go    # Video generation & composition
└── examples/
    └── test_api.sh     # API testing script
```

## Storage Structure

```
uploads/
└── {project-id}/
    ├── audio.mp3
    ├── character_0.png
    └── character_1.png

outputs/
└── {project-id}/
    ├── seeds/
    │   ├── seed_0_{uuid}.png
    │   ├── seed_1_{uuid}.png
    │   └── ...
    ├── animations/
    │   ├── anim_0_{uuid}.mp4
    │   ├── anim_1_{uuid}.mp4
    │   └── ...
    └── final_video.mp4
```

## Extensibility

### Adding New Pipeline Stages
1. Create new generator in `pipeline/`
2. Add to `Pipeline.Process()` workflow
3. Update `Project` model with new fields
4. Add corresponding status states

### Integrating AI Services
Replace placeholder functions with actual API calls:
- `ConceptGenerator.Generate()` → Audio analysis API
- `VisualSeeder.generateImage()` → Image generation API
- `Animator.generateAnimation()` → Video generation API
- `Animator.ComposeVideo()` → Video editing tools (ffmpeg)

### Custom Handlers
Add new API endpoints in `api/api.go`:
```go
router.GET("/api/v1/projects/:id/preview", server.generatePreview)
router.POST("/api/v1/projects/:id/regenerate", server.regenerateSegment)
```

## Performance Considerations

### Asynchronous Processing
- Pipeline execution runs in goroutines
- Non-blocking API responses
- Poll status endpoint for progress

### Scalability
- Stateless API server (can run multiple instances)
- Project storage can be replaced with database
- File storage can be replaced with object storage (S3, etc.)
- Pipeline stages can be distributed to workers

### Resource Management
- Configurable upload limits
- Cleanup old projects periodically
- Stream large files instead of loading in memory

## Security

### File Upload Safety
- Size limits enforced
- File type validation (can be added)
- Sandboxed storage paths
- No code execution from uploads

### API Security (To Add)
- Authentication/authorization
- Rate limiting
- Input validation
- CORS configuration

## Testing Strategy

### Unit Tests
Test individual components:
- Config loading
- Data model validation
- Pipeline stage logic

### Integration Tests
Test component interactions:
- API endpoints
- Pipeline orchestration
- File operations

### End-to-End Tests
Test complete workflows:
- Full video generation
- Error handling
- State transitions

## Deployment

### Development
```bash
go run main.go
```

### Production Build
```bash
go build -ldflags="-s -w" -o lumiere
./lumiere
```

### Docker (Example)
```dockerfile
FROM golang:1.21 AS builder
WORKDIR /app
COPY . .
RUN go build -o lumiere

FROM ubuntu:22.04
RUN apt-get update && apt-get install -y ffmpeg
COPY --from=builder /app/lumiere /usr/local/bin/
CMD ["lumiere"]
```

## Future Enhancements

1. **Database Integration**: Replace in-memory storage with PostgreSQL/MongoDB
2. **Queue System**: Use RabbitMQ/Redis for job management
3. **WebSocket Support**: Real-time progress updates
4. **Caching**: Cache intermediate results for faster iterations
5. **Multi-tenancy**: User accounts and project isolation
6. **Preview Generation**: Quick low-res previews before full generation
7. **Style Presets**: Pre-configured visual styles
8. **Batch Processing**: Process multiple songs simultaneously
9. **API Rate Limiting**: Protect against abuse
10. **Monitoring**: Prometheus metrics and health checks
