//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestHandleConfigGetReturnsFullRouterConfig(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "config.yaml")
	cfg := minimalDeployTestConfig("math_route")
	cfg.Projections.Partitions = []config.ProjectionPartition{{
		Name:        "subject_partition",
		Semantics:   "softmax_exclusive",
		Temperature: 0.1,
		Members:     []string{"math"},
		Default:     "math",
	}}
	if err := os.WriteFile(configPath, mustMarshalCanonicalConfigYAML(t, cfg), 0o644); err != nil {
		t.Fatalf("write router config: %v", err)
	}

	apiServer := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		configPath:        configPath,
	}

	req := httptest.NewRequest(http.MethodGet, "/config/router", nil)
	rr := httptest.NewRecorder()

	apiServer.handleConfigGet(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rr.Code, rr.Body.String())
	}

	var payload map[string]interface{}
	if err := json.Unmarshal(rr.Body.Bytes(), &payload); err != nil {
		t.Fatalf("json.Unmarshal error: %v", err)
	}

	routing, ok := payload["routing"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected routing block, got %#v", payload)
	}
	if _, hasSignals := routing["signals"].(map[string]interface{}); !hasSignals {
		t.Fatalf("expected routing.signals, got %#v", routing)
	}
	projections, ok := routing["projections"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected routing.projections, got %#v", routing)
	}
	if _, ok := projections["partitions"]; !ok {
		t.Fatalf("expected routing.projections.partitions in response, got %#v", projections)
	}
}

func TestHandleCombinedClassificationReturnsAllSubResponses(t *testing.T) {
	apiServer := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            &config.RouterConfig{},
	}

	body, err := json.Marshal(map[string]interface{}{"text": "Briefly explain what an API is."})
	if err != nil {
		t.Fatalf("json.Marshal error: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/v1/classify/combined", bytes.NewReader(body))
	rr := httptest.NewRecorder()

	apiServer.handleCombinedClassification(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rr.Code, rr.Body.String())
	}

	var response CombinedClassificationResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
		t.Fatalf("json.Unmarshal error: %v", err)
	}
	if response.Intent == nil || response.PII == nil || response.Security == nil {
		t.Fatalf("expected combined response to include all sub-responses, got %+v", response)
	}
}

func TestHandleClassificationMetricsReportsCounts(t *testing.T) {
	apiServer := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Signals: config.Signals{
					Categories: []config.Category{{
						CategoryMetadata: config.CategoryMetadata{Name: "math"},
					}},
					KeywordRules:   []config.KeywordRule{{Name: "urgent"}},
					EmbeddingRules: []config.EmbeddingRule{{Name: "fast_qa_en"}},
				},
				Projections: config.Projections{
					Partitions: []config.ProjectionPartition{{
						Name:        "subject_partition",
						Semantics:   "softmax_exclusive",
						Temperature: 0.1,
						Members:     []string{"fast_qa_en", "fast_qa_default"},
						Default:     "fast_qa_default",
					}},
					Scores: []config.ProjectionScore{{
						Name:   "difficulty_score",
						Method: "weighted_sum",
						Inputs: []config.ProjectionScoreInput{{
							Type:   config.SignalTypeKeyword,
							Name:   "urgent",
							Weight: 0.2,
						}},
					}},
					Mappings: []config.ProjectionMapping{{
						Name:   "difficulty_band",
						Source: "difficulty_score",
						Method: "threshold_bands",
						Outputs: []config.ProjectionMappingOutput{{
							Name: "balance_medium",
							GTE:  floatPtr(0.2),
						}},
					}},
				},
				Decisions: []config.Decision{{
					Name:      "math_route",
					ModelRefs: []config.ModelRef{{Model: "qwen-math"}},
					Rules:     config.RuleCombination{Operator: "AND"},
				}},
			},
		},
	}

	req := httptest.NewRequest(http.MethodGet, "/metrics/classification", nil)
	rr := httptest.NewRecorder()

	apiServer.handleClassificationMetrics(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rr.Code, rr.Body.String())
	}

	var response ClassificationMetricsResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
		t.Fatalf("json.Unmarshal error: %v", err)
	}
	if response.DecisionCount != 1 {
		t.Fatalf("decision_count = %d, want 1", response.DecisionCount)
	}
	if response.ProjectionPartitionCount != 1 || response.ProjectionScoreCount != 1 || response.ProjectionMappingCount != 1 {
		t.Fatalf("unexpected projection counts: %+v", response)
	}
	if response.SignalCounts["domains"] != 1 || response.SignalCounts["embeddings"] != 1 {
		t.Fatalf("unexpected signal counts: %+v", response.SignalCounts)
	}
	if response.SignalCounts["projection_partitions"] != 1 ||
		response.SignalCounts["projection_scores"] != 1 ||
		response.SignalCounts["projection_mappings"] != 1 {
		t.Fatalf("unexpected projection signal counts: %+v", response.SignalCounts)
	}
}

func floatPtr(v float64) *float64 {
	return &v
}
