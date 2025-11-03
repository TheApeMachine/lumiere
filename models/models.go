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
	Description string                 `json:"description"`
	KeyMoments  []KeyMoment            `json:"key_moments"`
	Intensity   []IntensityPoint       `json:"intensity"`
	Lyrics      []LyricSegment         `json:"lyrics,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// KeyMoment represents a significant moment in the music
type KeyMoment struct {
	Timestamp   float64 `json:"timestamp"`
	Description string  `json:"description"`
	Prompt      string  `json:"prompt"`
	Intensity   float64 `json:"intensity"`
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
	ID          string  `json:"id"`
	Timestamp   float64 `json:"timestamp"`
	Prompt      string  `json:"prompt"`
	ImagePath   string  `json:"image_path"`
	KeyMomentID int     `json:"key_moment_id"`
}

// Animation represents an animated video segment
type Animation struct {
	ID            string  `json:"id"`
	StartTime     float64 `json:"start_time"`
	EndTime       float64 `json:"end_time"`
	VideoPath     string  `json:"video_path"`
	FirstFrameID  string  `json:"first_frame_id"`
	LastFrameID   string  `json:"last_frame_id"`
}

// UploadRequest represents the initial upload request
type UploadRequest struct {
	Prompt string `json:"prompt" binding:"required"`
}
