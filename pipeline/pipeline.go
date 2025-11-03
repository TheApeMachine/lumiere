package pipeline

import (
	"fmt"
	"log"

	"github.com/TheApeMachine/lumiere/config"
	"github.com/TheApeMachine/lumiere/models"
)

// Pipeline orchestrates the complete video generation process
type Pipeline struct {
	config           *config.Config
	conceptGenerator *ConceptGenerator
	visualSeeder     *VisualSeeder
	animator         *Animator
}

// NewPipeline creates a new pipeline instance
func NewPipeline(cfg *config.Config) *Pipeline {
	return &Pipeline{
		config:           cfg,
		conceptGenerator: NewConceptGenerator(),
		visualSeeder:     NewVisualSeeder(cfg.OutputDir),
		animator:         NewAnimator(cfg.OutputDir),
	}
}

// Process runs the complete pipeline for a project
func (p *Pipeline) Process(project *models.Project) error {
	log.Printf("Starting pipeline for project %s", project.ID)
	
	// Step 1: Concept Generation
	if p.config.ConceptGenerationEnabled {
		log.Printf("Step 1/3: Generating concept...")
		concept, err := p.conceptGenerator.Generate(project.AudioFile, project.Prompt)
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
		seeds, err := p.visualSeeder.GenerateSeeds(project.ID, project.Concept, project.CharacterImages)
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
		animations, err := p.animator.GenerateAnimations(project.ID, project.VisualSeeds, project.AudioFile)
		if err != nil {
			return fmt.Errorf("animation generation failed: %w", err)
		}
		project.Animations = animations
		project.Status = "animations_generated"
		log.Printf("Generated %d animation segments", len(animations))
		
		// Compose final video
		log.Printf("Composing final video...")
		finalVideo, err := p.animator.ComposeVideo(project.ID, animations, project.AudioFile)
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
