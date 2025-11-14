package pipeline

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/TheApeMachine/lumiere/models"
)

type creativeDirectorClient struct {
	baseURL    string
	httpClient *http.Client
}

func newCreativeDirectorClient() *creativeDirectorClient {
	base := os.Getenv("CREATIVE_DIRECTOR_URL")
	if base == "" {
		base = "http://localhost:5004"
	}
	return &creativeDirectorClient{
		baseURL: base,
		httpClient: &http.Client{
			Timeout: 180 * time.Second, // 3 minutes for full agent pipeline (can take 60-90 seconds)
		},
	}
}

func (c *creativeDirectorClient) health() error {
	req, _ := http.NewRequest("GET", fmt.Sprintf("%s/healthz", c.baseURL), nil)
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return fmt.Errorf("creative-director unhealthy: %s", resp.Status)
	}
	return nil
}

type cdCharacter struct {
	ID              string   `json:"id"`
	Name            string   `json:"name,omitempty"`
	Role            string   `json:"role,omitempty"`
	Descriptors     []string `json:"descriptors,omitempty"`
	ReferenceImages []string `json:"reference_images,omitempty"`
	IdentityToken   string   `json:"identity_token,omitempty"`
}

type cdConcept struct {
	Title       string        `json:"title"`
	Theme       string        `json:"theme,omitempty"`
	Mood        string        `json:"mood,omitempty"`
	VisualStyle string        `json:"visual_style,omitempty"`
	Characters  []cdCharacter `json:"characters,omitempty"`
}

type cdSeedPromptsRequest struct {
	ProjectID     string                 `json:"project_id"`
	AudioSummary  map[string]interface{} `json:"audio_summary"`
	Concept       cdConcept              `json:"concept"`
	NumVariations int                    `json:"num_variations"`
	AudioFile     string                 `json:"audio_file,omitempty"`
}

type cdSeedPrompt struct {
	Prompt          string   `json:"prompt"`
	StartBeat       float64  `json:"start_beat"`
	EndBeat         float64  `json:"end_beat"`
	Characters      []string `json:"characters"`
	ReferenceImages []string `json:"reference_images"`
}

type cdSeedPromptsResponse struct {
	ConceptFinal cdConcept      `json:"concept_final"`
	Prompts      []cdSeedPrompt `json:"prompts"`
}

type cdVideoPromptsRequest struct {
	ProjectID    string                 `json:"project_id"`
	AudioSummary map[string]interface{} `json:"audio_summary"`
	Concept      cdConcept              `json:"concept"`
	BeatMap      []float64              `json:"beat_map"`
	AudioFile    string                 `json:"audio_file,omitempty"`
}

type cdVideoPromptSegment struct {
	StartBeat          float64  `json:"start_beat"`
	EndBeat            float64  `json:"end_beat"`
	Prompt             string   `json:"prompt"`
	NegativePrompt     string   `json:"negative_prompt"`
	MotionNotes        string   `json:"motion_notes"`
	Transition         string   `json:"transition"`
	CharactersOnScreen []string `json:"characters_on_screen"`
	ReferenceImages    []string `json:"reference_images"`
	SceneIndex         int      `json:"scene_index"`
	BeatIndex          int      `json:"beat_index"`
}

type cdVideoPromptsResponse struct {
	ConceptFinal cdConcept              `json:"concept_final"`
	Segments     []cdVideoPromptSegment `json:"segments"`
}

func (c *creativeDirectorClient) generateSeedPrompts(projectID string, concept *models.Concept, characterImages []string, audioFile string) (*cdSeedPromptsResponse, error) {
	// Build audio summary from concept analysis
	audioSummary := map[string]interface{}{}
	if concept != nil && concept.AudioAnalysis != nil {
		audioSummary["bpm"] = concept.AudioAnalysis.Tempo
		audioSummary["beats"] = concept.AudioAnalysis.Beats
	}

	// Map characters from uploaded images
	chars := make([]cdCharacter, 0)
	for i, path := range characterImages {
		chars = append(chars, cdCharacter{ID: fmt.Sprintf("char_%d", i), ReferenceImages: []string{path}})
	}

	reqPayload := cdSeedPromptsRequest{
		ProjectID:    projectID,
		AudioSummary: audioSummary,
		Concept: cdConcept{
			Title:       concept.Description,
			Theme:       concept.Theme,
			Mood:        concept.Mood,
			VisualStyle: concept.VisualStyle,
			Characters:  chars,
		},
		NumVariations: 7,
		AudioFile:     audioFile,
	}

	body, _ := json.Marshal(reqPayload)
	url := fmt.Sprintf("%s/v1/seed_prompts", c.baseURL)
	resp, err := c.httpClient.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("creative-director error: %s", resp.Status)
	}
	var out cdSeedPromptsResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (c *creativeDirectorClient) generateVideoPrompts(projectID string, concept *models.Concept, audioFile string) (*cdVideoPromptsResponse, error) {
	audioSummary := map[string]interface{}{}
	var beats []float64
	if concept != nil && concept.AudioAnalysis != nil {
		audioSummary["bpm"] = concept.AudioAnalysis.Tempo
		audioSummary["beats"] = concept.AudioAnalysis.Beats
		beats = concept.AudioAnalysis.Beats
	}

	reqPayload := cdVideoPromptsRequest{
		ProjectID:    projectID,
		AudioSummary: audioSummary,
		Concept: cdConcept{
			Title:       concept.Description,
			Theme:       concept.Theme,
			Mood:        concept.Mood,
			VisualStyle: concept.VisualStyle,
			Characters:  []cdCharacter{},
		},
		BeatMap:   beats,
		AudioFile: audioFile,
	}

	body, _ := json.Marshal(reqPayload)
	url := fmt.Sprintf("%s/v1/video_prompts", c.baseURL)
	resp, err := c.httpClient.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("creative-director error: %s", resp.Status)
	}
	var out cdVideoPromptsResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return &out, nil
}
