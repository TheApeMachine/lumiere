package pipeline

import (
	"fmt"
	"log"
	"os"
	"os/exec"

	"github.com/TheApeMachine/lumiere/config"
	"github.com/TheApeMachine/lumiere/models"
)

// Pipeline orchestrates the complete video generation process
type Pipeline struct {
	config           *config.Config
	conceptGenerator *ConceptGenerator
	visualSeeder     *VisualSeeder
	animator         *Animator
	aiServiceConfig  *AIServiceConfig
	useAI            bool
}

// NewPipeline creates a new pipeline instance
func NewPipeline(cfg *config.Config) *Pipeline {
	// Check if AI services should be used
	useAI := os.Getenv("USE_AI_SERVICES") == "true"

	return &Pipeline{
		config:           cfg,
		conceptGenerator: NewConceptGenerator(),
		visualSeeder:     NewVisualSeederWithConfig(cfg.OutputDir, cfg),
		animator:         NewAnimatorWithConfig(cfg.OutputDir, cfg),
		aiServiceConfig:  DefaultAIServiceConfig(),
		useAI:            useAI,
	}
}

// Process runs the complete pipeline for a project
func (p *Pipeline) Process(project *models.Project) (err error) {
	log.Printf("Starting pipeline for project %s (AI services: %v)", project.ID, p.useAI)

	// Step 1: Concept Generation
	if p.config.ConceptGenerationEnabled {
		log.Printf("Step 1/3: Generating concept...")
		var concept *models.Concept
		var err error

		if p.useAI {
			// Try AI service first
			aiGenerator := NewAIConceptGenerator(p.aiServiceConfig)
			concept, err = aiGenerator.Generate(project.AudioFile, project.Prompt)
		} else {
			// Use simulation
			concept, err = p.conceptGenerator.Generate(project.AudioFile, project.Prompt)
		}

		if err != nil {
			return fmt.Errorf("concept generation failed: %w", err)
		}
		project.Concept = concept
		project.Status = "concept_generated"
		log.Printf("Concept generated with %d key moments", len(concept.KeyMoments))
	}

	// Step 2: Visual Seeding
	if p.config.VisualSeedingEnabled && project.Concept != nil {
		log.Printf("Step 2/3: Generating visual seeds...")
		var seeds []models.VisualSeed
		var err error

		if p.useAI {
			// Use AI service for image generation
			seeds, err = p.generateSeedsWithAI(project.ID, project.Concept, project.CharacterImages)
		} else {
			seeds, err = p.visualSeeder.GenerateSeeds(project.ID, project.Concept, project.CharacterImages)
		}

		if err != nil {
			return fmt.Errorf("visual seeding failed: %w", err)
		}
		project.VisualSeeds = seeds
		project.Status = "seeds_generated"
		log.Printf("Generated %d visual seeds", len(seeds))
	}

	// Step 3: Animation
	if p.config.AnimationEnabled && len(project.VisualSeeds) > 0 {
		log.Printf("Step 3/3: Generating animations...")
		var animations []models.Animation

		if p.useAI {
			// Use AI service for video generation
			animations, err = p.generateAnimationsWithAI(project.ID, project.VisualSeeds, project.AudioFile)
		} else {
			animations, err = p.animator.GenerateAnimations(project.ID, project.VisualSeeds, project.AudioFile)
		}

		if err != nil {
			return fmt.Errorf("animation generation failed: %w", err)
		}
		project.Animations = animations
		project.Status = "animations_generated"
		log.Printf("Generated %d animation segments", len(animations))

		// Compose final video
		log.Printf("Composing final video...")
		var finalVideo string
		var err error

		if p.useAI {
			// Use proper video composition with ffmpeg
			finalVideo, err = p.composeVideoWithFFmpeg(project.ID, animations, project.AudioFile)
		} else {
			// Use placeholder composition
			finalVideo, err = p.animator.ComposeVideo(project.ID, animations, project.AudioFile)
		}

		if err != nil {
			return fmt.Errorf("video composition failed: %w", err)
		}
		project.FinalVideo = finalVideo
		project.Status = "completed"
		log.Printf("Final video created: %s", finalVideo)
	}

	log.Printf("Pipeline completed for project %s", project.ID)
	return nil
}

// generateSeedsWithAI generates visual seeds using AI service with creative director
func (p *Pipeline) generateSeedsWithAI(projectID string, concept *models.Concept, characterImages []string) ([]models.VisualSeed, error) {
	// First, get enhanced concept from creative director
	enhancedConcept, err := p.getEnhancedConceptFromCreativeDirector(projectID, concept, characterImages)
	if err != nil {
		log.Printf("Warning: Could not enhance concept with creative director: %v", err)
		// Fall back to original concept
		enhancedConcept = concept
	}

	aiSeeder := NewAIVisualSeeder(p.config.OutputDir, p.aiServiceConfig)
	seeds := []models.VisualSeed{}

	projectDir := fmt.Sprintf("%s/%s/seeds", p.config.OutputDir, projectID)
	os.MkdirAll(projectDir, 0755)

	// Use enhanced prompts from creative director if available
	moments := enhancedConcept.KeyMoments
	if len(moments) == 0 {
		moments = concept.KeyMoments // Fallback to original
	}

	for i, moment := range moments {
		seedID := fmt.Sprintf("seed_%d", i)
		imagePath := fmt.Sprintf("%s/%s.png", projectDir, seedID)

		// Generate image using enhanced prompt
		err := aiSeeder.GenerateImage(moment.Prompt, imagePath)
		if err != nil {
			return nil, fmt.Errorf("failed to generate seed %d: %w", i, err)
		}

		seed := models.VisualSeed{
			ID:          seedID,
			Timestamp:   moment.Timestamp,
			Prompt:      moment.Prompt,
			ImagePath:   imagePath,
			KeyMomentID: i,
		}
		seeds = append(seeds, seed)
	}

	return seeds, nil
}

// generateAnimationsWithAI generates animations using AI service
func (p *Pipeline) generateAnimationsWithAI(projectID string, seeds []models.VisualSeed, _ string) ([]models.Animation, error) {
	aiAnimator := NewAIAnimator(p.config.OutputDir, p.aiServiceConfig)
	animations := []models.Animation{}

	projectDir := fmt.Sprintf("%s/%s/animations", p.config.OutputDir, projectID)
	os.MkdirAll(projectDir, 0755)

	for i := 0; i < len(seeds)-1; i++ {
		currentSeed := seeds[i]
		nextSeed := seeds[i+1]

		animID := fmt.Sprintf("anim_%d", i)
		videoPath := fmt.Sprintf("%s/%s.mp4", projectDir, animID)
		duration := nextSeed.Timestamp - currentSeed.Timestamp

		// Generate animation using AI service
		err := aiAnimator.GenerateAnimation(currentSeed.ImagePath, nextSeed.ImagePath, videoPath, duration)
		if err != nil {
			return nil, fmt.Errorf("failed to generate animation %d: %w", i, err)
		}

		animation := models.Animation{
			ID:           animID,
			StartTime:    currentSeed.Timestamp,
			EndTime:      nextSeed.Timestamp,
			VideoPath:    videoPath,
			FirstFrameID: currentSeed.ID,
			LastFrameID:  nextSeed.ID,
		}
		animations = append(animations, animation)
	}

	return animations, nil
}

// composeVideoWithFFmpeg combines animation segments into final video using ffmpeg
func (p *Pipeline) composeVideoWithFFmpeg(projectID string, animations []models.Animation, audioPath string) (string, error) {
	if len(animations) == 0 {
		return "", fmt.Errorf("no animations to compose")
	}

	// Detect ffmpeg; fallback to placeholder composition if missing
	if _, lookErr := exec.LookPath("ffmpeg"); lookErr != nil {
		log.Printf("FFmpeg not found, falling back to placeholder composition: %v", lookErr)
		return p.animator.ComposeVideo(projectID, animations, audioPath)
	}

	projectDir := fmt.Sprintf("%s/%s", p.config.OutputDir, projectID)
	finalVideoPath := fmt.Sprintf("%s/final_video.mp4", projectDir)

	// Create a file list for ffmpeg concat
	concatListPath := fmt.Sprintf("%s/concat_list.txt", projectDir)
	concatContent := ""

	for _, anim := range animations {
		// Check if animation file exists and is a real video
		if _, err := os.Stat(anim.VideoPath); err != nil {
			log.Printf("Warning: Animation file not found: %s", anim.VideoPath)
			continue
		}
		concatContent += fmt.Sprintf("file '%s'\n", anim.VideoPath)
	}

	if concatContent == "" {
		return "", fmt.Errorf("no valid animation files found")
	}

	// Write concat list file
	if err := os.WriteFile(concatListPath, []byte(concatContent), 0644); err != nil {
		return "", fmt.Errorf("failed to create concat list: %w", err)
	}

	// Use ffmpeg to concatenate videos and add audio
	cmd := exec.Command("ffmpeg",
		"-f", "concat",
		"-safe", "0",
		"-i", concatListPath,
		"-i", audioPath,
		"-c:v", "libx264",
		"-c:a", "aac",
		"-shortest", // Match shortest stream (video or audio)
		"-y",        // Overwrite output file
		finalVideoPath,
	)

	log.Printf("Running ffmpeg command: %s", cmd.String())

	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("FFmpeg error: %s", string(output))
		return "", fmt.Errorf("ffmpeg failed: %w", err)
	}

	log.Printf("FFmpeg output: %s", string(output))

	// Clean up concat list file
	os.Remove(concatListPath)

	return finalVideoPath, nil
}

// getEnhancedConceptFromCreativeDirector gets an enhanced concept from the creative director service
func (p *Pipeline) getEnhancedConceptFromCreativeDirector(projectID string, originalConcept *models.Concept, characterImages []string) (*models.Concept, error) {
	log.Printf("Creative director enhancement requested for project %s", projectID)

	client := newCreativeDirectorClient()
	if err := client.health(); err != nil {
		return nil, fmt.Errorf("creative director unavailable: %w", err)
	}

	resp, err := client.generateSeedPrompts(projectID, originalConcept, characterImages)
	if err != nil {
		return nil, err
	}

	// Build enhanced concept using returned concept fields and prompts
	enhanced := &models.Concept{
		Description:   originalConcept.Description,
		Theme:         resp.ConceptFinal.Theme,
		Mood:          resp.ConceptFinal.Mood,
		VisualStyle:   resp.ConceptFinal.VisualStyle,
		KeyMoments:    make([]models.KeyMoment, 0),
		AudioAnalysis: originalConcept.AudioAnalysis,
		Metadata:      originalConcept.Metadata,
		Lyrics:        originalConcept.Lyrics,
		Intensity:     originalConcept.Intensity,
	}

	// Map prompts back to key moments using beats if available
	for _, pr := range resp.Prompts {
		km := models.KeyMoment{
			Timestamp:   pr.StartBeat,
			Description: fmt.Sprintf("scene between %.2f and %.2f", pr.StartBeat, pr.EndBeat),
			Prompt:      pr.Prompt,
			Intensity:   0.5,
		}
		enhanced.KeyMoments = append(enhanced.KeyMoments, km)
	}

	return enhanced, nil
}
