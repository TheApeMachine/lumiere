package config

import (
	"os"
)

// Config holds the application configuration
type Config struct {
	ServerPort string
	UploadDir  string
	OutputDir  string
	
	// AI Service Configuration
	ConceptGenerationEnabled bool
	VisualSeedingEnabled     bool
	AnimationEnabled         bool
}

// Load reads configuration from environment variables with defaults
func Load() (*Config, error) {
	cfg := &Config{
		ServerPort: getEnv("PORT", "8080"),
		UploadDir:  getEnv("UPLOAD_DIR", "./uploads"),
		OutputDir:  getEnv("OUTPUT_DIR", "./outputs"),
		
		ConceptGenerationEnabled: getEnvBool("CONCEPT_GENERATION_ENABLED", true),
		VisualSeedingEnabled:     getEnvBool("VISUAL_SEEDING_ENABLED", true),
		AnimationEnabled:         getEnvBool("ANIMATION_ENABLED", true),
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
