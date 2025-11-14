package models

import "time"

// Project represents a music video generation project
type Project struct {
	ID          string    `json:"id"`
	AudioFile   string    `json:"audio_file"`
	Prompt      string    `json:"prompt"`
	Status      string    `json:"status"`
	CreatedAt   time.Time `json:"created_at"`
	CompletedAt *time.Time `json:"completed_at,omitempty"`

	// Optional character images
	CharacterImages []string `json:"character_images,omitempty"`

	// Pipeline outputs
	Concept      *Concept      `json:"concept,omitempty"`
	VisualSeeds  []VisualSeed  `json:"visual_seeds,omitempty"`
	Animations   []Animation   `json:"animations,omitempty"`
	FinalVideo   string        `json:"final_video,omitempty"`
}

// Concept represents the AI-generated video concept
type Concept struct {
    Description   string                 `json:"description"`
    KeyMoments    []KeyMoment            `json:"key_moments"`
    Intensity     []IntensityPoint       `json:"intensity"`
    Lyrics        []LyricSegment         `json:"lyrics,omitempty"`
    AudioAnalysis *AudioAnalysis         `json:"audio_analysis,omitempty"`
    Metadata      map[string]interface{} `json:"metadata,omitempty"`
    // Creative Director additions
    Theme         string                 `json:"theme,omitempty"`
    Mood          string                 `json:"mood,omitempty"`
    VisualStyle   string                 `json:"visual_style,omitempty"`
    Characters    []Character            `json:"characters,omitempty"`
}

// AudioAnalysis contains detailed audio analysis results
type AudioAnalysis struct {
	Duration       float64        `json:"duration"`
	Tempo          float64        `json:"tempo"`
	Beats          []float64      `json:"beats"`           // Beat timestamps in seconds
	Downbeats      []float64      `json:"downbeats"`       // Downbeat timestamps in seconds
	OnsetStrengths []float64      `json:"onset_strengths"` // Strength of each onset
	SpectralCentroid []float64    `json:"spectral_centroid,omitempty"`
	ZeroCrossingRate []float64    `json:"zero_crossing_rate,omitempty"`
}

// KeyMoment represents a significant moment in the music
type KeyMoment struct {
	Timestamp        float64            `json:"timestamp"`
	Description      string             `json:"description"`
	Prompt           string             `json:"prompt"`
	Intensity        float64            `json:"intensity"`
	TransitionStyle  string             `json:"transition_style,omitempty"`  // e.g., "cut", "fade", "zoom"
	CutFrequency     string             `json:"cut_frequency,omitempty"`     // e.g., "slow", "medium", "fast"
	MotionIntensity  string             `json:"motion_intensity,omitempty"`  // e.g., "low", "medium", "high"
	CameraMovement   string             `json:"camera_movement,omitempty"`   // e.g., "static", "pan", "zoom"
}

// IntensityPoint represents audio intensity at a point in time
type IntensityPoint struct {
	Timestamp float64 `json:"timestamp"`
	Value     float64 `json:"value"`
}

// LyricSegment represents a segment of lyrics
type LyricSegment struct {
	StartTime float64 `json:"start_time"`
	EndTime   float64 `json:"end_time"`
	Text      string  `json:"text"`
}

// VisualSeed represents a generated keyframe image
type VisualSeed struct {
	ID                 string   `json:"id"`
	Timestamp          float64  `json:"timestamp"`
	Prompt             string   `json:"prompt"`
	ImagePath          string   `json:"image_path"`
	KeyMomentID        int      `json:"key_moment_id"`
	CharacterReference string   `json:"character_reference,omitempty"` // Path to character image for face preservation
	QualityScore       float64  `json:"quality_score,omitempty"`       // 0-1 score for image quality
	ValidationStatus   string   `json:"validation_status,omitempty"`   // e.g., "pass", "fail", "needs_review"
}

// Animation represents an animated video segment
type Animation struct {
	ID               string            `json:"id"`
	StartTime        float64           `json:"start_time"`
	EndTime          float64           `json:"end_time"`
	VideoPath        string            `json:"video_path"`
	FirstFrameID     string            `json:"first_frame_id"`
	LastFrameID      string            `json:"last_frame_id"`
	TransitionStyle  string            `json:"transition_style,omitempty"`  // Transition effect used
	MotionIntensity  string            `json:"motion_intensity,omitempty"`  // Motion level in this segment
	CameraMovement   string            `json:"camera_movement,omitempty"`   // Camera movement style
	QualityScore     float64           `json:"quality_score,omitempty"`     // 0-1 score for video quality
	ValidationStatus string            `json:"validation_status,omitempty"` // Validation result
	GenerationParams map[string]interface{} `json:"generation_params,omitempty"` // Parameters used for generation
}

// Character represents an actor/identity to maintain across prompts
type Character struct {
    ID              string   `json:"id"`
    Name            string   `json:"name,omitempty"`
    Role            string   `json:"role,omitempty"`
    Descriptors     []string `json:"descriptors,omitempty"`
    ReferenceImages []string `json:"reference_images,omitempty"`
    IdentityToken   string   `json:"identity_token,omitempty"`
}

// UploadRequest represents the initial upload request
type UploadRequest struct {
	Prompt string `json:"prompt" binding:"required"`
}
