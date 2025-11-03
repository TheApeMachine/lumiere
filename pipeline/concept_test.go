package pipeline

import (
	"testing"
)

func TestNewConceptGenerator(t *testing.T) {
	cg := NewConceptGenerator()
	if cg == nil {
		t.Fatal("NewConceptGenerator() returned nil")
	}
}

func TestGenerate(t *testing.T) {
	cg := NewConceptGenerator()
	
	audioPath := "/tmp/test_audio.mp3"
	userPrompt := "Epic landscape with mountains"
	
	concept, err := cg.Generate(audioPath, userPrompt)
	if err != nil {
		t.Fatalf("Generate() failed: %v", err)
	}
	
	if concept == nil {
		t.Fatal("Generate() returned nil concept")
	}
	
	// Check description
	if concept.Description == "" {
		t.Error("Expected non-empty description")
	}
	
	// Check key moments
	if len(concept.KeyMoments) != 7 {
		t.Errorf("Expected 7 key moments, got %d", len(concept.KeyMoments))
	}
	
	// Verify key moments are in chronological order
	for i := 1; i < len(concept.KeyMoments); i++ {
		if concept.KeyMoments[i].Timestamp <= concept.KeyMoments[i-1].Timestamp {
			t.Errorf("Key moments not in chronological order at index %d", i)
		}
	}
	
	// Check intensity curve
	if len(concept.Intensity) == 0 {
		t.Error("Expected intensity curve to be populated")
	}
	
	// Verify intensity values are in valid range [0, 1]
	for i, point := range concept.Intensity {
		if point.Value < 0 || point.Value > 1 {
			t.Errorf("Intensity point %d has invalid value %f (must be 0-1)", i, point.Value)
		}
	}
	
	// Check metadata
	if concept.Metadata == nil {
		t.Error("Expected metadata to be populated")
	}
	
	if _, ok := concept.Metadata["duration"]; !ok {
		t.Error("Expected duration in metadata")
	}
	
	if _, ok := concept.Metadata["user_prompt"]; !ok {
		t.Error("Expected user_prompt in metadata")
	}
}

func TestCalculateIntensity(t *testing.T) {
	tests := []struct {
		time float64
	}{
		{0.0},
		{0.25},
		{0.5},
		{0.75},
		{1.0},
	}
	
	for _, tt := range tests {
		intensity := calculateIntensity(tt.time)
		if intensity < 0 || intensity > 1 {
			t.Errorf("calculateIntensity(%f) = %f, want value in range [0, 1]", tt.time, intensity)
		}
	}
}

func TestGenerateKeyMoments(t *testing.T) {
	duration := 180.0
	userPrompt := "Test prompt"
	
	moments := generateKeyMoments(duration, userPrompt)
	
	if len(moments) != 7 {
		t.Errorf("Expected 7 key moments, got %d", len(moments))
	}
	
	// First moment should be at start
	if moments[0].Timestamp != 0.0 {
		t.Errorf("First moment should be at timestamp 0, got %f", moments[0].Timestamp)
	}
	
	// Last moment should be at end
	if moments[len(moments)-1].Timestamp != duration {
		t.Errorf("Last moment should be at timestamp %f, got %f", duration, moments[len(moments)-1].Timestamp)
	}
	
	// All moments should have prompts
	for i, moment := range moments {
		if moment.Prompt == "" {
			t.Errorf("Moment %d has empty prompt", i)
		}
		if moment.Description == "" {
			t.Errorf("Moment %d has empty description", i)
		}
	}
}

func TestGenerateIntensityCurve(t *testing.T) {
	duration := 10.0 // 10 seconds for quick test
	
	points := generateIntensityCurve(duration)
	
	// Should have approximately 10 points (one per second)
	if len(points) < 10 || len(points) > 11 {
		t.Errorf("Expected ~10 intensity points for 10 seconds, got %d", len(points))
	}
	
	// All points should have valid timestamps and values
	for i, point := range points {
		if point.Timestamp < 0 || point.Timestamp > duration {
			t.Errorf("Point %d has invalid timestamp %f", i, point.Timestamp)
		}
		if point.Value < 0 || point.Value > 1 {
			t.Errorf("Point %d has invalid value %f", i, point.Value)
		}
	}
}
