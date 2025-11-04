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
	
	// Verify key moments have transition parameters
	for i, moment := range concept.KeyMoments {
		if moment.TransitionStyle == "" {
			t.Errorf("Key moment %d missing transition style", i)
		}
		if moment.CutFrequency == "" {
			t.Errorf("Key moment %d missing cut frequency", i)
		}
		if moment.MotionIntensity == "" {
			t.Errorf("Key moment %d missing motion intensity", i)
		}
		if moment.CameraMovement == "" {
			t.Errorf("Key moment %d missing camera movement", i)
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
	
	// Check audio analysis
	if concept.AudioAnalysis == nil {
		t.Error("Expected audio analysis to be populated")
	} else {
		if concept.AudioAnalysis.Duration <= 0 {
			t.Error("Expected positive duration in audio analysis")
		}
		if concept.AudioAnalysis.Tempo <= 0 {
			t.Error("Expected positive tempo in audio analysis")
		}
		if len(concept.AudioAnalysis.Beats) == 0 {
			t.Error("Expected beats to be populated")
		}
		if len(concept.AudioAnalysis.Downbeats) == 0 {
			t.Error("Expected downbeats to be populated")
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
	
	if _, ok := concept.Metadata["tempo"]; !ok {
		t.Error("Expected tempo in metadata")
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

func TestGenerateAudioAnalysis(t *testing.T) {
	duration := 60.0 // 1 minute test
	
	analysis := generateAudioAnalysis(duration)
	
	if analysis == nil {
		t.Fatal("generateAudioAnalysis returned nil")
	}
	
	// Check duration
	if analysis.Duration != duration {
		t.Errorf("Expected duration %f, got %f", duration, analysis.Duration)
	}
	
	// Check tempo is in reasonable range
	if analysis.Tempo < 80 || analysis.Tempo > 160 {
		t.Errorf("Tempo %f is outside expected range 80-160 BPM", analysis.Tempo)
	}
	
	// Check beats are populated
	if len(analysis.Beats) == 0 {
		t.Error("Expected beats to be populated")
	}
	
	// Check downbeats are populated
	if len(analysis.Downbeats) == 0 {
		t.Error("Expected downbeats to be populated")
	}
	
	// Check downbeats are subset of beats
	if len(analysis.Downbeats) > len(analysis.Beats) {
		t.Error("More downbeats than beats, which is invalid")
	}
	
	// Verify beats are in chronological order
	for i := 1; i < len(analysis.Beats); i++ {
		if analysis.Beats[i] <= analysis.Beats[i-1] {
			t.Errorf("Beats not in chronological order at index %d", i)
		}
	}
	
	// Verify onset strengths match beat count
	if len(analysis.OnsetStrengths) != len(analysis.Beats) {
		t.Errorf("Expected %d onset strengths, got %d", len(analysis.Beats), len(analysis.OnsetStrengths))
	}
}

func TestFindNearestBeat(t *testing.T) {
	beats := []float64{0.0, 0.5, 1.0, 1.5, 2.0}
	
	tests := []struct {
		target   float64
		expected float64
	}{
		{0.0, 0.0},
		{0.2, 0.0},   // Closer to 0.0 than 0.5
		{0.3, 0.5},   // Closer to 0.5 than 0.0
		{0.7, 0.5},   // Closer to 0.5 than 1.0
		{0.8, 1.0},   // Closer to 1.0 than 0.5
		{1.25, 1.0},  // Closer to 1.0 than 1.5
		{1.3, 1.5},   // Closer to 1.5 than 1.0
		{1.8, 2.0},   // Closer to 2.0 than 1.5
	}
	
	for _, tt := range tests {
		result := findNearestBeat(tt.target, beats)
		if result != tt.expected {
			t.Errorf("findNearestBeat(%f) = %f, want %f", tt.target, result, tt.expected)
		}
	}
}

func TestDetermineTransitionParameters(t *testing.T) {
	tests := []struct {
		intensity          float64
		expectedTransition string
		expectedCutFreq    string
		expectedMotion     string
		expectedCamera     string
	}{
		{0.1, "fade", "slow", "low", "static"},
		{0.4, "dissolve", "medium", "medium", "pan"},
		{0.7, "cut", "fast", "high", "zoom"},
	}
	
	for _, tt := range tests {
		transition, cutFreq, motion, camera := determineTransitionParameters(tt.intensity)
		
		if transition != tt.expectedTransition {
			t.Errorf("intensity %f: transition = %s, want %s", tt.intensity, transition, tt.expectedTransition)
		}
		if cutFreq != tt.expectedCutFreq {
			t.Errorf("intensity %f: cutFreq = %s, want %s", tt.intensity, cutFreq, tt.expectedCutFreq)
		}
		if motion != tt.expectedMotion {
			t.Errorf("intensity %f: motion = %s, want %s", tt.intensity, motion, tt.expectedMotion)
		}
		if camera != tt.expectedCamera {
			t.Errorf("intensity %f: camera = %s, want %s", tt.intensity, camera, tt.expectedCamera)
		}
	}
}
