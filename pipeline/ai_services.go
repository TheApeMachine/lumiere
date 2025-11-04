package pipeline

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/TheApeMachine/lumiere/models"
)

// AIServiceConfig holds configuration for AI services
type AIServiceConfig struct {
	ImageServiceURL string
	VideoServiceURL string
	AudioServiceURL string
	Timeout         time.Duration
}

// DefaultAIServiceConfig returns default configuration
func DefaultAIServiceConfig() *AIServiceConfig {
	return &AIServiceConfig{
		ImageServiceURL: getEnv("IMAGE_SERVICE_URL", "http://localhost:5001"),
		VideoServiceURL: getEnv("VIDEO_SERVICE_URL", "http://localhost:5002"),
		AudioServiceURL: getEnv("AUDIO_SERVICE_URL", "http://localhost:5003"),
		Timeout:         30 * time.Second,
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// AIConceptGenerator uses Python service for audio analysis
type AIConceptGenerator struct {
	config *AIServiceConfig
	client *http.Client
}

// NewAIConceptGenerator creates a new AI-powered concept generator
func NewAIConceptGenerator(config *AIServiceConfig) *AIConceptGenerator {
	if config == nil {
		config = DefaultAIServiceConfig()
	}
	return &AIConceptGenerator{
		config: config,
		client: &http.Client{Timeout: config.Timeout},
	}
}

// Generate analyzes audio using Python service and creates a video concept
func (cg *AIConceptGenerator) Generate(audioPath, userPrompt string) (*models.Concept, error) {
	// Check if service is available
	healthURL := cg.config.AudioServiceURL + "/health"
	resp, err := cg.client.Get(healthURL)
	if err != nil || resp.StatusCode != 200 {
		// Fall back to simulated analysis
		fmt.Printf("Audio service not available, using fallback: %v\n", err)
		return NewConceptGenerator().Generate(audioPath, userPrompt)
	}
	resp.Body.Close()

	// Call audio analysis service
	reqData := map[string]interface{}{
		"audio_path": audioPath,
	}

	jsonData, err := json.Marshal(reqData)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	analyzeURL := cg.config.AudioServiceURL + "/analyze"
	resp, err = cg.client.Post(analyzeURL, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to call audio service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("audio service error: %s", string(body))
	}

	var result struct {
		Success  bool `json:"success"`
		Analysis struct {
			Duration       float64 `json:"duration"`
			Tempo          float64 `json:"tempo"`
			IntensityCurve []models.IntensityPoint `json:"intensity_curve"`
			KeyMoments     []struct {
				Timestamp   float64 `json:"timestamp"`
				Description string  `json:"description"`
				Intensity   float64 `json:"intensity"`
			} `json:"key_moments"`
		} `json:"analysis"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if !result.Success {
		return nil, fmt.Errorf("audio analysis failed")
	}

	// Convert to our models
	keyMoments := []models.KeyMoment{}
	for _, km := range result.Analysis.KeyMoments {
		keyMoments = append(keyMoments, models.KeyMoment{
			Timestamp:   km.Timestamp,
			Description: km.Description,
			Prompt:      fmt.Sprintf("%s, %s", userPrompt, km.Description),
			Intensity:   km.Intensity,
		})
	}

	concept := &models.Concept{
		Description: fmt.Sprintf("AI-generated music video concept based on prompt: %s", userPrompt),
		KeyMoments:  keyMoments,
		Intensity:   result.Analysis.IntensityCurve,
		Lyrics:      []models.LyricSegment{},
		Metadata: map[string]interface{}{
			"duration":    result.Analysis.Duration,
			"tempo":       result.Analysis.Tempo,
			"generated_at": time.Now().Format(time.RFC3339),
			"user_prompt": userPrompt,
		},
	}

	return concept, nil
}

// AIVisualSeeder uses Python service for image generation
type AIVisualSeeder struct {
	config    *AIServiceConfig
	client    *http.Client
	outputDir string
}

// NewAIVisualSeeder creates a new AI-powered visual seeder
func NewAIVisualSeeder(outputDir string, config *AIServiceConfig) *AIVisualSeeder {
	if config == nil {
		config = DefaultAIServiceConfig()
	}
	return &AIVisualSeeder{
		config:    config,
		client:    &http.Client{Timeout: 120 * time.Second}, // Longer timeout for image gen
		outputDir: outputDir,
	}
}

// GenerateImage generates an image using the Python service
func (vs *AIVisualSeeder) GenerateImage(prompt, outputPath string) error {
	// Check if service is available
	healthURL := vs.config.ImageServiceURL + "/health"
	resp, err := vs.client.Get(healthURL)
	if err != nil || resp.StatusCode != 200 {
		// Fall back to placeholder
		fmt.Printf("Image service not available, using placeholder: %v\n", err)
		return createPlaceholderImage(prompt, outputPath)
	}
	resp.Body.Close()

	// Call image generation service
	reqData := map[string]interface{}{
		"prompt":              prompt,
		"output_path":         outputPath,
		"num_inference_steps": 20, // Fast generation for consumer hardware
		"width":               512,
		"height":              512,
	}

	jsonData, err := json.Marshal(reqData)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	generateURL := vs.config.ImageServiceURL + "/generate"
	resp, err = vs.client.Post(generateURL, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to call image service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("image service error: %s", string(body))
	}

	var result struct {
		Success    bool   `json:"success"`
		OutputPath string `json:"output_path"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return fmt.Errorf("failed to decode response: %w", err)
	}

	if !result.Success {
		return fmt.Errorf("image generation failed")
	}

	return nil
}

func createPlaceholderImage(prompt, outputPath string) error {
	// Create a simple text placeholder
	content := fmt.Sprintf("# AI Generated Image\nPrompt: %s\n", prompt)
	return os.WriteFile(outputPath, []byte(content), 0644)
}

// AIAnimator uses Python service for video generation
type AIAnimator struct {
	config    *AIServiceConfig
	client    *http.Client
	outputDir string
}

// NewAIAnimator creates a new AI-powered animator
func NewAIAnimator(outputDir string, config *AIServiceConfig) *AIAnimator {
	if config == nil {
		config = DefaultAIServiceConfig()
	}
	return &AIAnimator{
		config:    config,
		client:    &http.Client{Timeout: 300 * time.Second}, // Very long timeout for video gen
		outputDir: outputDir,
	}
}

// GenerateAnimation generates a video using the Python service
func (a *AIAnimator) GenerateAnimation(startFrame, endFrame, outputPath string, duration float64) error {
	// Check if service is available
	healthURL := a.config.VideoServiceURL + "/health"
	resp, err := a.client.Get(healthURL)
	if err != nil || resp.StatusCode != 200 {
		// Fall back to placeholder
		fmt.Printf("Video service not available, using placeholder: %v\n", err)
		return createPlaceholderVideo(startFrame, endFrame, outputPath, duration)
	}
	resp.Body.Close()

	// Calculate frames and fps
	fps := 7.0
	numFrames := int(duration * fps)
	if numFrames < 7 {
		numFrames = 7
	}
	if numFrames > 30 {
		numFrames = 30
	}

	// Call video generation service
	reqData := map[string]interface{}{
		"start_frame": startFrame,
		"end_frame":   endFrame,
		"output_path": outputPath,
		"num_frames":  numFrames,
		"fps":         int(fps),
	}

	jsonData, err := json.Marshal(reqData)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	generateURL := a.config.VideoServiceURL + "/generate"
	resp, err = a.client.Post(generateURL, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to call video service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("video service error: %s", string(body))
	}

	var result struct {
		Success    bool   `json:"success"`
		OutputPath string `json:"output_path"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return fmt.Errorf("failed to decode response: %w", err)
	}

	if !result.Success {
		return fmt.Errorf("video generation failed")
	}

	return nil
}

func createPlaceholderVideo(startFrame, endFrame, outputPath string, duration float64) error {
	// Create a simple text placeholder
	content := fmt.Sprintf("# AI Generated Animation\nStart: %s\nEnd: %s\nDuration: %.2f seconds\n",
		startFrame, endFrame, duration)
	return os.WriteFile(outputPath, []byte(content), 0644)
}
