package api

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"

	"github.com/TheApeMachine/lumiere/config"
	"github.com/TheApeMachine/lumiere/models"
	"github.com/TheApeMachine/lumiere/pipeline"
)

// Server represents the API server
type Server struct {
	config   *config.Config
	pipeline *pipeline.Pipeline
	projects map[string]*models.Project
	mu       sync.RWMutex
}

// NewServer creates a new API server
func NewServer(cfg *config.Config) *Server {
	return &Server{
		config:   cfg,
		pipeline: pipeline.NewPipeline(cfg),
		projects: make(map[string]*models.Project),
	}
}

// Start initializes and starts the API server
func Start(cfg *config.Config) error {
	server := NewServer(cfg)
	router := setupRouter(server)
	return router.Run(":" + cfg.ServerPort)
}

func setupRouter(server *Server) *gin.Engine {
	router := gin.Default()
	
	// Health check
	router.GET("/health", server.healthCheck)
	
	// API endpoints
	api := router.Group("/api/v1")
	{
		api.POST("/projects", server.createProject)
		api.GET("/projects/:id", server.getProject)
		api.GET("/projects", server.listProjects)
		api.POST("/projects/:id/process", server.processProject)
	}
	
	return router
}

func (s *Server) healthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status": "healthy",
		"service": "lumiere",
	})
}

func (s *Server) createProject(c *gin.Context) {
	// Parse multipart form
	if err := c.Request.ParseMultipartForm(100 << 20); err != nil { // 100 MB max
		c.JSON(http.StatusBadRequest, gin.H{"error": "Failed to parse form"})
		return
	}
	
	// Get prompt
	prompt := c.PostForm("prompt")
	if prompt == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Prompt is required"})
		return
	}
	
	// Create project
	projectID := uuid.New().String()
	project := &models.Project{
		ID:        projectID,
		Prompt:    prompt,
		Status:    "created",
		CreatedAt: time.Now(),
		CharacterImages: []string{},
	}
	
	// Handle audio file upload
	audioFile, err := c.FormFile("audio")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Audio file is required"})
		return
	}
	
	// Save audio file
	audioPath := filepath.Join(s.config.UploadDir, projectID, "audio.mp3")
	if err := os.MkdirAll(filepath.Dir(audioPath), 0755); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create upload directory"})
		return
	}
	
	if err := c.SaveUploadedFile(audioFile, audioPath); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to save audio file"})
		return
	}
	project.AudioFile = audioPath
	
	// Handle optional character images
	form := c.Request.MultipartForm
	if form != nil && form.File["character_images"] != nil {
		for i, fileHeader := range form.File["character_images"] {
			imagePath := filepath.Join(s.config.UploadDir, projectID, fmt.Sprintf("character_%d.png", i))
			
			file, err := fileHeader.Open()
			if err != nil {
				continue
			}
			defer file.Close()
			
			outFile, err := os.Create(imagePath)
			if err != nil {
				continue
			}
			defer outFile.Close()
			
			if _, err := io.Copy(outFile, file); err != nil {
				continue
			}
			
			project.CharacterImages = append(project.CharacterImages, imagePath)
		}
	}
	
	// Store project
	s.mu.Lock()
	s.projects[projectID] = project
	s.mu.Unlock()
	
	c.JSON(http.StatusCreated, project)
}

func (s *Server) getProject(c *gin.Context) {
	projectID := c.Param("id")
	
	s.mu.RLock()
	project, exists := s.projects[projectID]
	s.mu.RUnlock()
	
	if !exists {
		c.JSON(http.StatusNotFound, gin.H{"error": "Project not found"})
		return
	}
	
	c.JSON(http.StatusOK, project)
}

func (s *Server) listProjects(c *gin.Context) {
	s.mu.RLock()
	projectList := make([]*models.Project, 0, len(s.projects))
	for _, project := range s.projects {
		projectList = append(projectList, project)
	}
	s.mu.RUnlock()
	
	c.JSON(http.StatusOK, gin.H{
		"projects": projectList,
		"count":    len(projectList),
	})
}

func (s *Server) processProject(c *gin.Context) {
	projectID := c.Param("id")
	
	s.mu.RLock()
	project, exists := s.projects[projectID]
	s.mu.RUnlock()
	
	if !exists {
		c.JSON(http.StatusNotFound, gin.H{"error": "Project not found"})
		return
	}
	
	if project.Status != "created" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Project already processed or in progress"})
		return
	}
	
	// Process asynchronously
	project.Status = "processing"
	go func() {
		if err := s.pipeline.Process(project); err != nil {
			project.Status = "failed"
			fmt.Printf("Pipeline failed for project %s: %v\n", projectID, err)
		} else {
			completedAt := time.Now()
			project.CompletedAt = &completedAt
		}
	}()
	
	c.JSON(http.StatusOK, gin.H{
		"message": "Processing started",
		"project_id": projectID,
	})
}
