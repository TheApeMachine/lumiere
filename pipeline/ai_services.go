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

// Service health models (subset of fields we care about)
type videoHealthResponse struct {
	PipelineLoaded bool   `json:"pipeline_loaded"`
	LTXAvailable   bool   `json:"ltx_available"`
	Mode           string `json:"mode"`
	Device         string `json:"device"`
}

// ensureVideoReady verifies the video service can generate videos (not just be up)
func ensureVideoReady(cfg *AIServiceConfig) error {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(cfg.VideoServiceURL + "/health")
	if err != nil {
		return fmt.Errorf("video service health check failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("video service unhealthy: %s", string(body))
	}
	var vh videoHealthResponse
	if err := json.NewDecoder(resp.Body).Decode(&vh); err != nil {
		return fmt.Errorf("invalid video health response: %w", err)
	}
	if !vh.LTXAvailable || !vh.PipelineLoaded || vh.Mode != "ltx-video" {
		return fmt.Errorf("video generator not ready (device=%s, mode=%s, ltx=%v, loaded=%v)", vh.Device, vh.Mode, vh.LTXAvailable, vh.PipelineLoaded)
	}
	return nil
}

// AIServiceConfig holds configuration for AI services
type AIServiceConfig struct {
	ImageServiceURL     string
	VideoServiceURL     string
	AudioServiceURL     string
	CreativeDirectorURL string
	Timeout             time.Duration
}

// DefaultAIServiceConfig returns default configuration
func DefaultAIServiceConfig() *AIServiceConfig {
	return &AIServiceConfig{
		ImageServiceURL:     getEnv("IMAGE_SERVICE_URL", "http://localhost:5001"),
		VideoServiceURL:     getEnv("VIDEO_SERVICE_URL", "http://localhost:5002"),
		AudioServiceURL:     getEnv("AUDIO_SERVICE_URL", "http://localhost:5003"),
		CreativeDirectorURL: getEnv("CREATIVE_DIRECTOR_URL", "http://localhost:5004"),
		Timeout:             30 * time.Second,
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
		if resp != nil {
			resp.Body.Close()
		}
		return nil, fmt.Errorf("audio service unavailable at %s", cg.config.AudioServiceURL)
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
			Duration       float64                 `json:"duration"`
			Tempo          float64                 `json:"tempo"`
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
			"duration":     result.Analysis.Duration,
			"tempo":        result.Analysis.Tempo,
			"generated_at": time.Now().Format(time.RFC3339),
			"user_prompt":  userPrompt,
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
	return vs.GenerateImageWithParams(prompt, outputPath, nil)
}

// GenerateImageWithParams generates an image with custom parameters
func (vs *AIVisualSeeder) GenerateImageWithParams(prompt, outputPath string, customParams map[string]interface{}) error {
	// Check if service is available
	healthURL := vs.config.ImageServiceURL + "/health"
	resp, err := vs.client.Get(healthURL)
	if err != nil || resp.StatusCode != 200 {
		if resp != nil {
			resp.Body.Close()
		}
		return fmt.Errorf("image service unavailable at %s", vs.config.ImageServiceURL)
	}
	resp.Body.Close()

	// Build request data with defaults
	reqData := map[string]interface{}{
		"prompt":              prompt,
		"output_path":         outputPath,
		"num_inference_steps": 20, // Fast generation for consumer hardware
		"width":               512,
		"height":              512,
	}

	// Override with custom parameters if provided
	for k, v := range customParams {
		reqData[k] = v
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
	return a.GenerateAnimationWithParams(startFrame, endFrame, outputPath, duration, nil)
}

// GenerateAnimationWithParams generates a video with custom parameters
func (a *AIAnimator) GenerateAnimationWithParams(startFrame, endFrame, outputPath string, duration float64, customParams map[string]interface{}) error {
	// Check if service is available
	healthURL := a.config.VideoServiceURL + "/health"
	resp, err := a.client.Get(healthURL)
	if err != nil || resp.StatusCode != 200 {
		if resp != nil {
			resp.Body.Close()
		}
		return fmt.Errorf("video service unavailable at %s", a.config.VideoServiceURL)
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

	// Build request data with defaults
	reqData := map[string]interface{}{
		"start_frame": startFrame,
		"end_frame":   endFrame,
		"output_path": outputPath,
		"num_frames":  numFrames,
		"fps":         int(fps),
	}

	// Override with custom parameters if provided
	for k, v := range customParams {
		reqData[k] = v
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
