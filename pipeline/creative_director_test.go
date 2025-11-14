package pipeline

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/TheApeMachine/lumiere/models"
)

func TestCreativeDirectorClient_GenerateSeedPrompts(t *testing.T) {
	// Mock server
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		w.Write([]byte(`{"status":"ok"}`))
	})
	mux.HandleFunc("/v1/seed_prompts", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		io := map[string]interface{}{
			"concept_final": map[string]string{
				"theme":        "change",
				"mood":         "uplifting",
				"visual_style": "cyberpunk",
			},
			"prompts": []map[string]interface{}{
				{"prompt": "scene A", "start_beat": 0.0, "end_beat": 1.0, "characters": []string{}, "reference_images": []string{}},
				{"prompt": "scene B", "start_beat": 1.0, "end_beat": 2.0, "characters": []string{}, "reference_images": []string{}},
			},
		}
		_ = json.NewEncoder(w).Encode(io)
	})
	server := httptest.NewServer(mux)
	defer server.Close()

	os.Setenv("CREATIVE_DIRECTOR_URL", server.URL)

	client := newCreativeDirectorClient()
	if err := client.health(); err != nil {
		t.Fatalf("health failed: %v", err)
	}

	concept := &models.Concept{Description: "Test"}
	resp, err := client.generateSeedPrompts("proj1", concept, []string{"/tmp/char.png"}, "")
	if err != nil {
		t.Fatalf("generateSeedPrompts error: %v", err)
	}
	if resp.ConceptFinal.VisualStyle != "cyberpunk" {
		t.Fatalf("unexpected visual style: %s", resp.ConceptFinal.VisualStyle)
	}
	if len(resp.Prompts) != 2 {
		t.Fatalf("expected 2 prompts, got %d", len(resp.Prompts))
	}
}
