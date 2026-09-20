package classification

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestLLMLabelClassifierReturnsReportedDistribution(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
			ChatTemplateKwargs map[string]bool `json:"chat_template_kwargs"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if len(request.Messages) != 2 || !strings.Contains(request.Messages[0].Content, `"scores"`) {
			http.Error(w, "missing score contract", http.StatusBadRequest)
			return
		}
		if enabled, exists := request.ChatTemplateKwargs["enable_thinking"]; !exists || enabled {
			http.Error(w, "missing disabled reasoning control", http.StatusBadRequest)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{{
				"message": map[string]interface{}{
					"content": `{"scores":{"SAFE":0.23,"RISKY":0.77},"rationale":"destructive operation"}`,
				},
			}},
		})
	}))
	defer server.Close()

	disabled := false
	external := &config.ExternalModelConfig{
		ModelEndpoint: config.ClassifierVLLMEndpoint{Address: "placeholder", Port: 1},
		ModelName:     "test-model",
		Reasoning: &config.ExternalModelReasoningConfig{
			Family:       "qwen3",
			UseReasoning: &disabled,
		},
	}
	cfg := &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{
		ReasoningConfig: config.ReasoningConfig{ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
			"qwen3": {
				Type:        config.ReasoningFamilyTypeChatTemplateKwargs,
				Parameter:   "enable_thinking",
				Modes:       []string{config.ReasoningModeEnabled, config.ReasoningModeDisabled},
				DefaultMode: config.ReasoningModeEnabled,
			},
		}},
	}}
	reasoning, err := resolveExternalModelReasoningControl(cfg.ReasoningFamilies, external)
	if err != nil {
		t.Fatalf("resolveExternalModelReasoningControl() error = %v", err)
	}
	classifier, err := newLLMLabelClassifier(
		config.ClassifierSignalRule{
			Model:        "test-model",
			Labels:       []string{"SAFE", "RISKY"},
			Instructions: "Classify the input.",
		},
		external,
		reasoning,
	)
	if err != nil {
		t.Fatalf("newLLMLabelClassifier() error = %v", err)
	}
	setTestVLLMClientURL(classifier.(*llmLabelClassifier).client, server.URL)

	result, err := classifier.Classify(context.Background(), "delete production")
	if err != nil {
		t.Fatalf("Classify() error = %v", err)
	}
	if result.Scores["SAFE"] != 0.23 || result.Scores["RISKY"] != 0.77 {
		t.Errorf("scores = %v, want SAFE=0.23 and RISKY=0.77", result.Scores)
	}
	if result.Rationale != "destructive operation" {
		t.Errorf("rationale = %q, want destructive operation", result.Rationale)
	}
}

func TestLLMClassifierReasoningRequestBoundary(t *testing.T) {
	enabled := true
	disabled := false
	tests := []struct {
		name       string
		reasoning  *config.ExternalModelReasoningConfig
		families   map[string]config.ReasoningFamilyConfig
		wantEffort string
		wantChat   map[string]interface{}
	}{
		{name: "omitted"},
		{
			name:      "disabled",
			reasoning: &config.ExternalModelReasoningConfig{Family: "qwen3", UseReasoning: &disabled},
			families: map[string]config.ReasoningFamilyConfig{
				"qwen3": {
					Type: config.ReasoningFamilyTypeChatTemplateKwargs, Parameter: "enable_thinking",
					Modes: []string{config.ReasoningModeEnabled, config.ReasoningModeDisabled},
				},
			},
			wantChat: map[string]interface{}{"enable_thinking": false},
		},
		{
			name:      "enabled with effort",
			reasoning: &config.ExternalModelReasoningConfig{Family: "effort", UseReasoning: &enabled, ReasoningEffort: "low"},
			families: map[string]config.ReasoningFamilyConfig{
				"effort": {
					Type: config.ReasoningFamilyTypeReasoningEffort, Parameter: "reasoning_effort",
					Levels: []string{"low", "high"}, Default: "high",
				},
			},
			wantChat: map[string]interface{}{"reasoning_effort": "low"},
		},
		{
			name:      "top-level effort",
			reasoning: &config.ExternalModelReasoningConfig{Family: "top-level", UseReasoning: &enabled, ReasoningEffort: "high"},
			families: map[string]config.ReasoningFamilyConfig{
				"top-level": {
					Type: config.ReasoningFamilyTypeTopLevelReasoningEffort, Parameter: "reasoning_effort",
					Levels: []string{"low", "high"}, Default: "low",
				},
			},
			wantEffort: "high",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			requests := make(chan map[string]interface{}, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
					http.Error(w, "unexpected classifier request", http.StatusBadRequest)
					return
				}
				var request map[string]interface{}
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}
				requests <- request
				_ = json.NewEncoder(w).Encode(map[string]interface{}{
					"choices": []map[string]interface{}{{
						"message": map[string]interface{}{
							"content": `{"scores":{"SAFE":0.23,"RISKY":0.77},"rationale":"classified"}`,
						},
					}},
				})
			}))
			t.Cleanup(server.Close)

			cfg := &config.RouterConfig{
				ExternalModels: []config.ExternalModelConfig{{
					Name:          "judge",
					Provider:      "vllm",
					ModelRole:     config.ModelRoleClassification,
					ModelEndpoint: config.ClassifierVLLMEndpoint{Address: "placeholder", Port: 1},
					ModelName:     "test-model",
					Reasoning:     tt.reasoning,
				}},
				IntelligentRouting: config.IntelligentRouting{
					ReasoningConfig: config.ReasoningConfig{ReasoningFamilies: tt.families},
					Signals: config.Signals{ClassifierRules: []config.ClassifierSignalRule{{
						Name:         "risk",
						Type:         config.ClassifierSignalTypeLLM,
						Model:        "judge",
						Labels:       []string{"SAFE", "RISKY"},
						Instructions: "Classify the input.",
					}}},
				},
			}
			builder := &classifierOptionBuilder{cfg: cfg}
			apply, err := builder.buildGenericClassifiersOption()
			if err != nil {
				t.Fatalf("buildGenericClassifiersOption() error = %v", err)
			}
			owner := &Classifier{}
			apply(owner)
			t.Cleanup(func() { closeLabelClassifiers(owner.genericClassifiers) })
			classifier := owner.genericClassifiers["risk"].(*llmLabelClassifier)
			setTestVLLMClientURL(classifier.client, server.URL)

			if _, err := classifier.Classify(context.Background(), "delete production"); err != nil {
				t.Fatalf("Classify() error = %v", err)
			}
			request := <-requests
			if tt.wantEffort == "" {
				if _, exists := request["reasoning_effort"]; exists {
					t.Fatalf("unexpected reasoning_effort: %v", request)
				}
			} else if request["reasoning_effort"] != tt.wantEffort {
				t.Errorf("reasoning_effort = %v, want %q", request["reasoning_effort"], tt.wantEffort)
			}
			if tt.wantChat == nil {
				if _, exists := request["chat_template_kwargs"]; exists {
					t.Fatalf("unexpected chat_template_kwargs: %v", request)
				}
			} else {
				got, ok := request["chat_template_kwargs"].(map[string]interface{})
				if !ok || !mapsEqual(got, tt.wantChat) {
					t.Errorf("chat_template_kwargs = %v, want %v", request["chat_template_kwargs"], tt.wantChat)
				}
			}
		})
	}
}

func TestParseLLMLabelClassificationRejectsInvalidDistribution(t *testing.T) {
	tests := []struct {
		name    string
		content string
		wantErr string
	}{
		{
			name:    "legacy one-hot response",
			content: `{"label":"RISKY","rationale":"test"}`,
			wantErr: "exactly scores and rationale",
		},
		{
			name:    "missing label",
			content: `{"scores":{"RISKY":1},"rationale":"test"}`,
			wantErr: "exactly the declared labels",
		},
		{
			name:    "undeclared label",
			content: `{"scores":{"SAFE":0.2,"OTHER":0.8},"rationale":"test"}`,
			wantErr: `missing label "RISKY"`,
		},
		{
			name:    "score out of range",
			content: `{"scores":{"SAFE":-0.1,"RISKY":1.1},"rationale":"test"}`,
			wantErr: "within [0, 1]",
		},
		{
			name:    "scores do not sum to one",
			content: `{"scores":{"SAFE":0.2,"RISKY":0.3},"rationale":"test"}`,
			wantErr: "want approximately 1",
		},
		{
			name:    "empty rationale",
			content: `{"scores":{"SAFE":0.2,"RISKY":0.8},"rationale":" "}`,
			wantErr: "empty rationale",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := parseLLMLabelClassification(tt.content, []string{"SAFE", "RISKY"}, false)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("parseLLMLabelClassification() error = %v, want %q", err, tt.wantErr)
			}
		})
	}
}
