package main

import (
	"fmt"
	"log"
	"os"

	"github.com/TheApeMachine/lumiere/api"
	"github.com/TheApeMachine/lumiere/config"
)

func main() {
	// Load configuration
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	// Ensure required directories exist
	if err := ensureDirectories(cfg); err != nil {
		log.Fatalf("Failed to create directories: %v", err)
	}

	// Start API server
	fmt.Printf("Starting Lumiere AI Music Video Generator on port %s\n", cfg.ServerPort)
	if err := api.Start(cfg); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

func ensureDirectories(cfg *config.Config) error {
	dirs := []string{cfg.UploadDir, cfg.OutputDir}
	for _, dir := range dirs {
		if err := os.MkdirAll(dir, config.DefaultDirPerms); err != nil {
			return fmt.Errorf("failed to create directory %s: %w", dir, err)
		}
	}
	return nil
}
