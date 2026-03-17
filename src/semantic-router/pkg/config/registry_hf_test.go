package config

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

type hfFixtureHits struct {
	api       int
	config    int
	tokenizer int
	readme    int
}

func TestGetModelRegistryInfoByPathOverlaysHuggingFaceMetadata(t *testing.T) {
	hits := &hfFixtureHits{}
	server := newFeedbackDetectorFixtureServer(t, hits)
	defer server.Close()

	restore := useTestRegistryResolver(server)
	defer restore()

	info := GetModelRegistryInfoByPath("models/mmbert32k-feedback-detector-merged")
	requireFeedbackDetectorOverlay(t, info)
	requireFixtureHits(t, hits, 1)

	info = GetModelRegistryInfoByPath("models/mmbert32k-feedback-detector-merged")
	if info == nil {
		t.Fatal("expected cached registry info")
	}
	requireFixtureHits(t, hits, 1)
}

func TestGetModelRegistryInfoByPathFallsBackToLocalRegistry(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "nope", http.StatusBadGateway)
	}))
	defer server.Close()

	restore := useTestRegistryResolver(server)
	defer restore()

	info := GetModelRegistryInfoByPath("models/mmbert32k-feedback-detector-merged")
	if info == nil {
		t.Fatal("expected model registry info")
	}
	if info.Description != "Merged 4-class user feedback classifier based on mmbert-32k-yarn for direct inference without PEFT." {
		t.Fatalf("expected local fallback description, got %q", info.Description)
	}
	if info.ParameterSize != "307M" {
		t.Fatalf("expected local fallback parameter size, got %q", info.ParameterSize)
	}
	if info.PipelineTag != "" {
		t.Fatalf("expected no remote pipeline tag on fallback, got %q", info.PipelineTag)
	}
}

func newFeedbackDetectorFixtureServer(t *testing.T, hits *hfFixtureHits) *httptest.Server {
	t.Helper()

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/models/llm-semantic-router/mmbert32k-feedback-detector-merged":
			hits.api++
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id":           "llm-semantic-router/mmbert32k-feedback-detector-merged",
				"pipeline_tag": "text-classification",
				"tags": []string{
					"transformers",
					"text-classification",
					"feedback-detection",
					"dataset:llm-semantic-router/feedback-detector-dataset",
				},
				"cardData": map[string]any{
					"base_model":   "llm-semantic-router/mmbert-32k-yarn",
					"license":      "apache-2.0",
					"language":     []string{"en", "zh"},
					"datasets":     []string{"llm-semantic-router/feedback-detector-dataset"},
					"pipeline_tag": "text-classification",
					"tags":         []string{"text-classification", "feedback-detection", "multilingual"},
				},
				"safetensors": map[string]any{
					"total": 307533316,
				},
			})
		case "/llm-semantic-router/mmbert32k-feedback-detector-merged/raw/main/config.json":
			hits.config++
			_ = json.NewEncoder(w).Encode(map[string]any{
				"hidden_size":             768,
				"max_position_embeddings": 32768,
				"id2label": map[string]string{
					"0": "SAT",
					"1": "NEED_CLARIFICATION",
					"2": "WRONG_ANSWER",
					"3": "WANT_DIFFERENT",
				},
			})
		case "/llm-semantic-router/mmbert32k-feedback-detector-merged/raw/main/tokenizer_config.json":
			hits.tokenizer++
			_ = json.NewEncoder(w).Encode(map[string]any{
				"model_max_length": 32768,
			})
		case "/llm-semantic-router/mmbert32k-feedback-detector-merged/raw/main/README.md":
			hits.readme++
			_, _ = w.Write([]byte(`---
license: apache-2.0
---

# mmBERT-32K Feedback Detector (Merged)

A 4-class user feedback classifier based on mmbert-32k-yarn.

## Details
More text here.
`))
		default:
			t.Fatalf("unexpected request path %q", r.URL.Path)
		}
	}))
}

func useTestRegistryResolver(server *httptest.Server) func() {
	originalResolver := defaultRegistryCardResolver
	resolver := newModelRegistryCardResolver()
	resolver.baseURL = server.URL
	resolver.client = server.Client()
	resolver.ttl = time.Hour
	defaultRegistryCardResolver = resolver

	return func() {
		defaultRegistryCardResolver = originalResolver
	}
}

func requireFeedbackDetectorOverlay(t *testing.T, info *ModelRegistryInfo) {
	t.Helper()

	if info == nil {
		t.Fatal("expected model registry info")
	}
	requireFeedbackDetectorIdentity(t, info)
	requireFeedbackDetectorCapabilities(t, info)
	requireFeedbackDetectorCardMetadata(t, info)
}

func requireFeedbackDetectorIdentity(t *testing.T, info *ModelRegistryInfo) {
	t.Helper()

	if info.RepoID != "llm-semantic-router/mmbert32k-feedback-detector-merged" {
		t.Fatalf("expected canonical repo id, got %q", info.RepoID)
	}
	if info.Description != "A 4-class user feedback classifier based on mmbert-32k-yarn." {
		t.Fatalf("expected README description overlay, got %q", info.Description)
	}
	if info.ParameterSize != "307M" {
		t.Fatalf("expected parameter size from safetensors, got %q", info.ParameterSize)
	}
}

func requireFeedbackDetectorCapabilities(t *testing.T, info *ModelRegistryInfo) {
	t.Helper()

	if info.EmbeddingDim != 768 {
		t.Fatalf("expected embedding dim 768, got %d", info.EmbeddingDim)
	}
	if info.MaxContextLength != 32768 {
		t.Fatalf("expected max context 32768, got %d", info.MaxContextLength)
	}
	if info.NumClasses != 4 {
		t.Fatalf("expected 4 classes, got %d", info.NumClasses)
	}
	if info.PipelineTag != "text-classification" {
		t.Fatalf("expected pipeline tag, got %q", info.PipelineTag)
	}
}

func requireFeedbackDetectorCardMetadata(t *testing.T, info *ModelRegistryInfo) {
	t.Helper()

	if info.BaseModel != "llm-semantic-router/mmbert-32k-yarn" {
		t.Fatalf("expected base model, got %q", info.BaseModel)
	}
	if info.License != "apache-2.0" {
		t.Fatalf("expected license, got %q", info.License)
	}
	if len(info.Languages) != 2 {
		t.Fatalf("expected languages, got %+v", info.Languages)
	}
	if len(info.Datasets) != 1 {
		t.Fatalf("expected datasets, got %+v", info.Datasets)
	}
}

func requireFixtureHits(t *testing.T, hits *hfFixtureHits, want int) {
	t.Helper()

	if hits.api != want || hits.config != want || hits.tokenizer != want || hits.readme != want {
		t.Fatalf(
			"expected %d fetch per remote resource, got api=%d config=%d tokenizer=%d readme=%d",
			want,
			hits.api,
			hits.config,
			hits.tokenizer,
			hits.readme,
		)
	}
}
