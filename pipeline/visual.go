package pipeline

import (
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/TheApeMachine/lumiere/config"
	"github.com/TheApeMachine/lumiere/models"
	"github.com/google/uuid"
)

// VisualSeeder generates keyframe images from concepts
type VisualSeeder struct {
	outputDir              string
	baseSeedDensity        float64
	intensityMultiplier    float64
	enableFacePreservation bool
	enableQualityControl   bool
	minQualityScore        float64
}

// NewVisualSeeder creates a new visual seeder
func NewVisualSeeder(outputDir string) *VisualSeeder {
	return &VisualSeeder{
		outputDir:              outputDir,
		baseSeedDensity:        2.0,  // 2 seeds per minute by default
		intensityMultiplier:    2.0,  // Up to 2x more seeds at maximum intensity
		enableFacePreservation: true,
		enableQualityControl:   true,
		minQualityScore:        0.5,
	}
}

// NewVisualSeederWithConfig creates a visual seeder with custom configuration
func NewVisualSeederWithConfig(outputDir string, cfg *config.Config) *VisualSeeder {
	return &VisualSeeder{
		outputDir:              outputDir,
		baseSeedDensity:        cfg.BaseSeedDensity,
		intensityMultiplier:    cfg.IntensityMultiplier,
		enableFacePreservation: cfg.EnableFacePreservation,
		enableQualityControl:   cfg.EnableQualityControl,
		minQualityScore:        cfg.MinQualityScore,
	}
}

// GenerateSeeds creates keyframe images for each key moment with variable density
// In production, this would call an AI image generation API like:
// - Stable Diffusion
// - DALL-E
// - Midjourney API
func (vs *VisualSeeder) GenerateSeeds(projectID string, concept *models.Concept, characterImages []string) ([]models.VisualSeed, error) {
	seeds := []models.VisualSeed{}

	// Create project output directory
	projectDir := filepath.Join(vs.outputDir, projectID, "seeds")
	if err := os.MkdirAll(projectDir, config.DefaultDirPerms); err != nil {
		return nil, fmt.Errorf("failed to create seeds directory: %w", err)
	}

	// Select primary character image for face preservation
	var primaryCharacter string
	if len(characterImages) > 0 && vs.enableFacePreservation {
		primaryCharacter = characterImages[0]
	}

	// Generate seeds with variable density based on intensity
	seedsToGenerate := vs.calculateSeedPositions(concept)

	for i, seedPos := range seedsToGenerate {
		seedID := uuid.New().String()
		imagePath := filepath.Join(projectDir, fmt.Sprintf("seed_%d_%s.png", i, seedID))

		// Enhance prompt with character reference if available
		enhancedPrompt := vs.enhancePromptWithCharacter(seedPos.Prompt, primaryCharacter)

		// Simulate image generation
		if err := vs.generateImage(enhancedPrompt, imagePath, characterImages); err != nil {
			return nil, fmt.Errorf("failed to generate image for seed %d: %w", i, err)
		}

		// Validate quality if enabled
		qualityScore := 1.0
		validationStatus := "pass"
		if vs.enableQualityControl {
			qualityScore = vs.validateImageQuality(imagePath)
			if qualityScore < vs.minQualityScore {
				validationStatus = "needs_review"
			}
		}

		seed := models.VisualSeed{
			ID:                 seedID,
			Timestamp:          seedPos.Timestamp,
			Prompt:             enhancedPrompt,
			ImagePath:          imagePath,
			KeyMomentID:        seedPos.KeyMomentID,
			CharacterReference: primaryCharacter,
			QualityScore:       qualityScore,
			ValidationStatus:   validationStatus,
		}
		seeds = append(seeds, seed)
	}

	return seeds, nil
}

// SeedPosition represents a calculated position for a seed
type SeedPosition struct {
	Timestamp   float64
	Prompt      string
	Intensity   float64
	KeyMomentID int
}

// calculateSeedPositions determines seed positions with variable density based on intensity
func (vs *VisualSeeder) calculateSeedPositions(concept *models.Concept) []SeedPosition {
	positions := []SeedPosition{}

	// Always include key moments
	for i, moment := range concept.KeyMoments {
		positions = append(positions, SeedPosition{
			Timestamp:   moment.Timestamp,
			Prompt:      moment.Prompt,
			Intensity:   moment.Intensity,
			KeyMomentID: i,
		})
	}

	// Add additional seeds between key moments based on intensity
	if concept.AudioAnalysis != nil && len(concept.AudioAnalysis.Beats) > 0 {
		positions = vs.addIntensityBasedSeeds(positions, concept)
	}

	return positions
}

// addIntensityBasedSeeds adds extra seeds in high-intensity sections
func (vs *VisualSeeder) addIntensityBasedSeeds(basePositions []SeedPosition, concept *models.Concept) []SeedPosition {
	allPositions := basePositions

	// Find high intensity periods and add more seeds
	for i := 0; i < len(concept.KeyMoments)-1; i++ {
		currentMoment := concept.KeyMoments[i]
		nextMoment := concept.KeyMoments[i+1]

		// If this is a high intensity section, add intermediate seeds
		if currentMoment.Intensity > 0.6 {
			// Calculate number of extra seeds based on intensity
			duration := nextMoment.Timestamp - currentMoment.Timestamp
			extraSeeds := vs.calculateExtraSeedCount(duration, currentMoment.Intensity)

			if extraSeeds > 0 {
				// Align extra seeds with beats if available
				interval := duration / float64(extraSeeds+1)
				for j := 1; j <= extraSeeds; j++ {
					timestamp := currentMoment.Timestamp + interval*float64(j)

					// Align to nearest beat if audio analysis available
					if concept.AudioAnalysis != nil && len(concept.AudioAnalysis.Beats) > 0 {
						timestamp = findNearestBeatInRange(timestamp, concept.AudioAnalysis.Beats,
							currentMoment.Timestamp, nextMoment.Timestamp)
					}

					// Interpolate prompt between key moments
					prompt := fmt.Sprintf("%s transitioning to %s",
						currentMoment.Description, nextMoment.Description)

					allPositions = append(allPositions, SeedPosition{
						Timestamp:   timestamp,
						Prompt:      prompt,
						Intensity:   currentMoment.Intensity,
						KeyMomentID: i,
					})
				}
			}
		}
	}

	return allPositions
}

// calculateExtraSeedCount determines how many extra seeds to add based on duration and intensity
func (vs *VisualSeeder) calculateExtraSeedCount(duration float64, intensity float64) int {
	// Calculate base seeds for this duration
	totalSeeds := duration / 60.0 * vs.baseSeedDensity * intensity * vs.intensityMultiplier
	// Subtract 1 because we already have the key moment seed
	extraSeeds := int(totalSeeds) - 1
	if extraSeeds < 0 {
		extraSeeds = 0
	}
	return extraSeeds
}

// findNearestBeatInRange finds the nearest beat within a time range
func findNearestBeatInRange(target float64, beats []float64, minTime, maxTime float64) float64 {
	if len(beats) == 0 {
		return target
	}

	// Initialize with first valid beat or target
	nearest := target
	minDiff := math.MaxFloat64

	for _, beat := range beats {
		if beat >= minTime && beat <= maxTime {
			diff := math.Abs(target - beat)
			if diff < minDiff {
				minDiff = diff
				nearest = beat
			}
		}
	}

	return nearest
}

// enhancePromptWithCharacter adds character reference to prompt for face preservation
func (vs *VisualSeeder) enhancePromptWithCharacter(prompt, characterImage string) string {
	if !vs.enableFacePreservation || characterImage == "" {
		return prompt
	}

	// In production, this would pass the character image to the AI model
	// For now, we just enhance the text prompt
	return fmt.Sprintf("%s, featuring the character from reference image, maintaining facial features and likeness", prompt)
}

// validateImageQuality performs quality validation on generated image
func (vs *VisualSeeder) validateImageQuality(_ string) float64 {
	// TODO: In production, this would:
	// 1. Check for blur/sharpness using variance of Laplacian
	// 2. Check for artifacts using noise detection
	// 3. Verify composition using rule of thirds or similar
	// 4. Check color balance using histogram analysis

	// Placeholder: Return a deterministic simulated score for testing
	// This ensures consistent behavior and makes testing predictable
	return 0.75 // Fixed good score for placeholder implementation
}

// generateImage simulates AI image generation
// In production, integrate with actual image generation APIs
func (vs *VisualSeeder) generateImage(prompt string, outputPath string, characterImages []string) error {
	// Create a placeholder image file
	// In production, this would:
	// 1. Call image generation API with prompt
	// 2. Optionally use characterImages for character consistency
	// 3. Download and save the generated image

	placeholderContent := fmt.Sprintf("# AI Generated Image\nPrompt: %s\nCharacter Images: %v\n", prompt, characterImages)

	if err := os.WriteFile(outputPath, []byte(placeholderContent), config.DefaultFilePerms); err != nil {
		return fmt.Errorf("failed to write placeholder image: %w", err)
	}

	return nil
}
