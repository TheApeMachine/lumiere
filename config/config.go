package config

import (
	"os"
	"strconv"
)

const (
	// File and directory permissions
	DefaultFilePerms = 0644
	DefaultDirPerms  = 0755
	
	// Default upload size limit in MB
	DefaultMaxUploadSizeMB = 100
)

// Config holds the application configuration
type Config struct {
	ServerPort string
	UploadDir  string
	OutputDir  string
	
	// File upload configuration
	MaxUploadSizeMB int64
	
	// AI Service Configuration
	ConceptGenerationEnabled bool
	VisualSeedingEnabled     bool
	AnimationEnabled         bool
	
	// Video Generation Settings
	VideoResolution string  // e.g., "720p", "1080p", "512x512"
	VideoFPS        int     // Frames per second
	VideoWidth      int     // Width in pixels
	VideoHeight     int     // Height in pixels
	
	// Quality Control Settings
	EnableQualityControl   bool
	MinQualityScore        float64 // Minimum acceptable quality score (0-1)
	EnableFacePreservation bool    // Enable character face preservation
	
	// Seed Density Settings
	BaseSeedDensity     float64 // Base seeds per minute
	IntensityMultiplier float64 // How much intensity affects seed density
}

// Load reads configuration from environment variables with defaults
func Load() (*Config, error) {
	cfg := &Config{
		ServerPort:      getEnv("PORT", "8080"),
		UploadDir:       getEnv("UPLOAD_DIR", "./uploads"),
		OutputDir:       getEnv("OUTPUT_DIR", "./outputs"),
		MaxUploadSizeMB: getEnvInt64("MAX_UPLOAD_SIZE_MB", DefaultMaxUploadSizeMB),
		
		ConceptGenerationEnabled: getEnvBool("CONCEPT_GENERATION_ENABLED", true),
		VisualSeedingEnabled:     getEnvBool("VISUAL_SEEDING_ENABLED", true),
		AnimationEnabled:         getEnvBool("ANIMATION_ENABLED", true),
		
		// Video Generation Settings
		VideoResolution: getEnv("VIDEO_RESOLUTION", "720p"),
		VideoFPS:        getEnvInt("VIDEO_FPS", 24),
		VideoWidth:      getEnvInt("VIDEO_WIDTH", 1280),
		VideoHeight:     getEnvInt("VIDEO_HEIGHT", 720),
		
		// Quality Control Settings
		EnableQualityControl:   getEnvBool("ENABLE_QUALITY_CONTROL", true),
		MinQualityScore:        getEnvFloat64("MIN_QUALITY_SCORE", 0.5),
		EnableFacePreservation: getEnvBool("ENABLE_FACE_PRESERVATION", true),
		
		// Seed Density Settings
		BaseSeedDensity:     getEnvFloat64("BASE_SEED_DENSITY", 2.0),      // 2 seeds per minute by default
		IntensityMultiplier: getEnvFloat64("INTENSITY_MULTIPLIER", 2.0), // Up to 3x more seeds at high intensity
	}
	
	return cfg, nil
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvBool(key string, defaultValue bool) bool {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	return value == "true" || value == "1"
}

func getEnvInt64(key string, defaultValue int64) int64 {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	if intVal, err := strconv.ParseInt(value, 10, 64); err == nil {
		return intVal
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	if intVal, err := strconv.Atoi(value); err == nil {
		return intVal
	}
	return defaultValue
}

func getEnvFloat64(key string, defaultValue float64) float64 {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	if floatVal, err := strconv.ParseFloat(value, 64); err == nil {
		return floatVal
	}
	return defaultValue
}
