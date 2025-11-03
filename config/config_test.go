package config

import (
	"os"
	"testing"
)

func TestLoadDefault(t *testing.T) {
	// Clear environment variables
	os.Clearenv()
	
	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load() failed: %v", err)
	}
	
	// Test default values
	if cfg.ServerPort != "8080" {
		t.Errorf("Expected default port 8080, got %s", cfg.ServerPort)
	}
	
	if cfg.UploadDir != "./uploads" {
		t.Errorf("Expected default upload dir ./uploads, got %s", cfg.UploadDir)
	}
	
	if cfg.OutputDir != "./outputs" {
		t.Errorf("Expected default output dir ./outputs, got %s", cfg.OutputDir)
	}
	
	if cfg.MaxUploadSizeMB != DefaultMaxUploadSizeMB {
		t.Errorf("Expected default max upload %d MB, got %d MB", DefaultMaxUploadSizeMB, cfg.MaxUploadSizeMB)
	}
	
	if !cfg.ConceptGenerationEnabled {
		t.Error("Expected concept generation to be enabled by default")
	}
	
	if !cfg.VisualSeedingEnabled {
		t.Error("Expected visual seeding to be enabled by default")
	}
	
	if !cfg.AnimationEnabled {
		t.Error("Expected animation to be enabled by default")
	}
}

func TestLoadFromEnv(t *testing.T) {
	// Set environment variables
	os.Setenv("PORT", "9000")
	os.Setenv("UPLOAD_DIR", "/tmp/uploads")
	os.Setenv("OUTPUT_DIR", "/tmp/outputs")
	os.Setenv("MAX_UPLOAD_SIZE_MB", "200")
	os.Setenv("CONCEPT_GENERATION_ENABLED", "false")
	defer os.Clearenv()
	
	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load() failed: %v", err)
	}
	
	if cfg.ServerPort != "9000" {
		t.Errorf("Expected port 9000, got %s", cfg.ServerPort)
	}
	
	if cfg.UploadDir != "/tmp/uploads" {
		t.Errorf("Expected upload dir /tmp/uploads, got %s", cfg.UploadDir)
	}
	
	if cfg.OutputDir != "/tmp/outputs" {
		t.Errorf("Expected output dir /tmp/outputs, got %s", cfg.OutputDir)
	}
	
	if cfg.MaxUploadSizeMB != 200 {
		t.Errorf("Expected max upload 200 MB, got %d MB", cfg.MaxUploadSizeMB)
	}
	
	if cfg.ConceptGenerationEnabled {
		t.Error("Expected concept generation to be disabled")
	}
}

func TestGetEnvBool(t *testing.T) {
	tests := []struct {
		value    string
		expected bool
	}{
		{"true", true},
		{"1", true},
		{"false", false},
		{"0", false},
		{"", false}, // default when using false as default
	}
	
	for _, tt := range tests {
		os.Setenv("TEST_BOOL", tt.value)
		result := getEnvBool("TEST_BOOL", false)
		if result != tt.expected {
			t.Errorf("getEnvBool(%q) = %v, want %v", tt.value, result, tt.expected)
		}
		os.Unsetenv("TEST_BOOL")
	}
}

func TestGetEnvInt64(t *testing.T) {
	tests := []struct {
		value    string
		expected int64
	}{
		{"100", 100},
		{"0", 0},
		{"999", 999},
		{"invalid", 50}, // falls back to default
		{"", 50},        // falls back to default
	}
	
	for _, tt := range tests {
		os.Setenv("TEST_INT", tt.value)
		result := getEnvInt64("TEST_INT", 50)
		if result != tt.expected {
			t.Errorf("getEnvInt64(%q, 50) = %v, want %v", tt.value, result, tt.expected)
		}
		os.Unsetenv("TEST_INT")
	}
}
