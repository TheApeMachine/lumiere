package pipeline

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/TheApeMachine/lumiere/models"
)

var rng = rand.New(rand.NewSource(time.Now().UnixNano()))

// ConceptGenerator analyzes audio and generates video concepts
type ConceptGenerator struct{}

// NewConceptGenerator creates a new concept generator
func NewConceptGenerator() *ConceptGenerator {
	return &ConceptGenerator{}
}

// Generate analyzes the audio file and creates a video concept
// In a real implementation, this would use ML models to:
// - Analyze audio intensity/tempo
// - Detect beats and rhythm patterns
// - Extract lyrics (if present)
// - Generate creative prompts for visuals
func (cg *ConceptGenerator) Generate(audioPath, userPrompt string) (*models.Concept, error) {
	// Simulate audio analysis (in production, use actual audio processing libraries)
	duration := estimateAudioDuration(audioPath)
	
	// Generate detailed audio analysis
	audioAnalysis := generateAudioAnalysis(duration)
	
	// Generate key moments aligned with beats and intensity
	keyMoments := generateKeyMomentsWithBeats(duration, userPrompt, audioAnalysis)
	
	concept := &models.Concept{
		Description:   fmt.Sprintf("AI-generated music video concept based on prompt: %s", userPrompt),
		KeyMoments:    keyMoments,
		Intensity:     generateIntensityCurve(duration),
		Lyrics:        []models.LyricSegment{},
		AudioAnalysis: audioAnalysis,
		Metadata: map[string]interface{}{
			"duration":     duration,
			"generated_at": time.Now().Format(time.RFC3339),
			"user_prompt":  userPrompt,
			"tempo":        audioAnalysis.Tempo,
			"total_beats":  len(audioAnalysis.Beats),
		},
	}
	
	return concept, nil
}

// estimateAudioDuration simulates getting audio duration
// In production, use a library like go-mp3 or ffmpeg
func estimateAudioDuration(audioPath string) float64 {
	// For demonstration, return a typical song duration
	// Real implementation would parse the MP3 file
	return 180.0 // 3 minutes
}

// generateKeyMoments creates significant moments for the video
func generateKeyMoments(duration float64, userPrompt string) []models.KeyMoment {
	moments := []models.KeyMoment{}
	
	// Generate key moments at strategic intervals
	intervals := []float64{0.0, 0.15, 0.33, 0.50, 0.67, 0.85, 1.0}
	themes := []string{
		"opening scene, establishing atmosphere",
		"building energy, introducing elements",
		"first climax, dynamic action",
		"bridge, contemplative moment",
		"building to crescendo",
		"peak moment, maximum intensity",
		"resolution, closing scene",
	}
	
	for i, interval := range intervals {
		timestamp := duration * interval
		intensity := calculateIntensity(interval)
		
		moment := models.KeyMoment{
			Timestamp:   timestamp,
			Description: themes[i],
			Prompt:      fmt.Sprintf("%s, %s", userPrompt, themes[i]),
			Intensity:   intensity,
		}
		moments = append(moments, moment)
	}
	
	return moments
}

// generateIntensityCurve creates an intensity profile for the audio
func generateIntensityCurve(duration float64) []models.IntensityPoint {
	points := []models.IntensityPoint{}
	
	// Sample at 1-second intervals
	for t := 0.0; t < duration; t += 1.0 {
		normalized := t / duration
		intensity := calculateIntensity(normalized)
		
		points = append(points, models.IntensityPoint{
			Timestamp: t,
			Value:     intensity,
		})
	}
	
	return points
}

// calculateIntensity generates a realistic intensity curve
func calculateIntensity(normalizedTime float64) float64 {
	// Create a dynamic intensity curve typical of music
	// Start low, build up, peak around 2/3, then resolve
	base := 0.3 + 0.4*math.Sin(normalizedTime*math.Pi)
	variation := 0.2 * math.Sin(normalizedTime*math.Pi*4)
	noise := (rng.Float64() - 0.5) * 0.1
	
	intensity := base + variation + noise
	
	// Clamp between 0 and 1
	if intensity < 0 {
		intensity = 0
	}
	if intensity > 1 {
		intensity = 1
	}
	
	return intensity
}

// generateAudioAnalysis creates detailed audio analysis with beat detection
func generateAudioAnalysis(duration float64) *models.AudioAnalysis {
	// Simulate tempo detection (in production, use librosa or similar)
	tempo := 120.0 + (rng.Float64()-0.5)*40.0 // 100-140 BPM range
	
	// Generate beat timestamps based on tempo
	beatInterval := 60.0 / tempo // seconds per beat
	beats := []float64{}
	downbeats := []float64{}
	onsetStrengths := []float64{}
	
	for t := 0.0; t < duration; t += beatInterval {
		beats = append(beats, t)
		onsetStrengths = append(onsetStrengths, 0.5+rng.Float64()*0.5)
		
		// Every 4th beat is a downbeat
		if len(beats)%4 == 1 {
			downbeats = append(downbeats, t)
		}
	}
	
	return &models.AudioAnalysis{
		Duration:       duration,
		Tempo:          tempo,
		Beats:          beats,
		Downbeats:      downbeats,
		OnsetStrengths: onsetStrengths,
	}
}

// generateKeyMomentsWithBeats creates key moments aligned with beats and downbeats
func generateKeyMomentsWithBeats(duration float64, userPrompt string, analysis *models.AudioAnalysis) []models.KeyMoment {
	moments := []models.KeyMoment{}
	
	// Strategic intervals aligned with musical structure
	intervals := []float64{0.0, 0.15, 0.33, 0.50, 0.67, 0.85, 1.0}
	themes := []string{
		"opening scene, establishing atmosphere",
		"building energy, introducing elements",
		"first climax, dynamic action",
		"bridge, contemplative moment",
		"building to crescendo",
		"peak moment, maximum intensity",
		"resolution, closing scene",
	}
	
	for i, interval := range intervals {
		timestamp := duration * interval
		
		// Align timestamp to nearest downbeat for better sync
		if len(analysis.Downbeats) > 0 {
			timestamp = findNearestBeat(timestamp, analysis.Downbeats)
		}
		
		intensity := calculateIntensity(interval)
		
		// Determine transition style based on intensity
		transitionStyle, cutFreq, motionInt, cameraMove := determineTransitionParameters(intensity)
		
		moment := models.KeyMoment{
			Timestamp:       timestamp,
			Description:     themes[i],
			Prompt:          fmt.Sprintf("%s, %s", userPrompt, themes[i]),
			Intensity:       intensity,
			TransitionStyle: transitionStyle,
			CutFrequency:    cutFreq,
			MotionIntensity: motionInt,
			CameraMovement:  cameraMove,
		}
		moments = append(moments, moment)
	}
	
	return moments
}

// findNearestBeat finds the nearest beat timestamp to a given time
func findNearestBeat(target float64, beats []float64) float64 {
	if len(beats) == 0 {
		return target
	}
	
	nearest := beats[0]
	minDiff := math.Abs(target - nearest)
	
	for _, beat := range beats {
		diff := math.Abs(target - beat)
		if diff < minDiff {
			minDiff = diff
			nearest = beat
		}
	}
	
	return nearest
}

// determineTransitionParameters determines transition parameters based on intensity
func determineTransitionParameters(intensity float64) (transitionStyle, cutFreq, motionInt, cameraMove string) {
	if intensity < 0.3 {
		// Low intensity - slow, contemplative
		transitionStyle = "fade"
		cutFreq = "slow"
		motionInt = "low"
		cameraMove = "static"
	} else if intensity < 0.6 {
		// Medium intensity - moderate movement
		transitionStyle = "dissolve"
		cutFreq = "medium"
		motionInt = "medium"
		cameraMove = "pan"
	} else {
		// High intensity - fast, dynamic
		transitionStyle = "cut"
		cutFreq = "fast"
		motionInt = "high"
		cameraMove = "zoom"
	}
	
	return
}
