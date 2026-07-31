package extproc

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestDecisionPromptSelectorCallsConcreteHelperModel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-vsr-looper-request") != "true" {
			t.Fatalf("missing internal looper header")
		}
		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if body["model"] != "router-small" {
			t.Fatalf("model = %v", body["model"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-router",
			"object":"chat.completion",
			"created":1,
			"model":"router-small",
			"choices":[{
				"index":0,
				"message":{
					"role":"assistant",
					"content":"{\"selected_model\":\"reasoning-large\",\"rationale\":\"Hard task\"}"
				},
				"finish_reason":"stop"
			}]
		}`))
	}))
	defer server.Close()

	router := &OpenAIRouter{Config: &config.RouterConfig{
		Looper: config.LooperConfig{Endpoint: server.URL},
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"general-small":   {Description: "Fast model"},
			"reasoning-large": {Description: "Hard reasoning"},
		}},
	}}
	selector := router.newDecisionPromptSelector(config.PromptSelectionConfig{
		Model:        "router-small",
		Instructions: "Choose by difficulty.",
	})

	result, err := selector.Select(t.Context(), &selection.SelectionContext{
		Query: "solve this proof",
		CandidateModels: []config.ModelRef{
			{Model: "general-small"},
			{Model: "reasoning-large"},
		},
	})
	if err != nil {
		t.Fatalf("Select() error = %v", err)
	}
	if result.SelectedModel != "reasoning-large" {
		t.Fatalf("SelectedModel = %q", result.SelectedModel)
	}
}
