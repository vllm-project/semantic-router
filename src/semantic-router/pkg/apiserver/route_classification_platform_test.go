//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestHandleGetConfigReturnsRoutingSurface(t *testing.T) {
	apiServer := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Signals: config.Signals{
					Categories: []config.Category{
						{
							CategoryMetadata: config.CategoryMetadata{Name: "math"},
						},
						{
							CategoryMetadata: config.CategoryMetadata{
								Name:           "general",
								MMLUCategories: []string{"other"},
							},
						},
					},
				},
				Projections: config.Projections{
					Partitions: []config.ProjectionPartition{{
						Name:        "subject_partition",
						Semantics:   "softmax_exclusive",
						Temperature: 0.1,
						Members:     []string{"math", "general"},
						Default:     "general",
					}},
				},
				Decisions: []config.Decision{{
					Name:     "math_route",
					Tier:     2,
					Priority: 100,
					Rules: config.RuleCombination{
						Operator: "AND",
						Conditions: []config.RuleNode{{
							Type: "domain",
							Name: "math",
						}},
					},
					ModelRefs: []config.ModelRef{{Model: "qwen-math"}},
				}},
			},
		},
	}

	req := httptest.NewRequest(http.MethodGet, "/config/classification", nil)
	rr := httptest.NewRecorder()

	apiServer.handleGetConfig(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rr.Code, rr.Body.String())
	}

	var payload map[string]interface{}
	if err := json.Unmarshal(rr.Body.Bytes(), &payload); err != nil {
		t.Fatalf("json.Unmarshal error: %v", err)
	}

	routing, ok := payload["routing"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected routing document, got %#v", payload)
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

func TestHandleUpdateConfigMergesRoutingPayload(t *testing.T) {
	apiServer := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Signals: config.Signals{
					Categories: []config.Category{{
						CategoryMetadata: config.CategoryMetadata{Name: "math"},
					}},
				},
				Decisions: []config.Decision{{
					Name:     "math_route",
					Priority: 100,
					Rules: config.RuleCombination{
						Operator: "AND",
						Conditions: []config.RuleNode{{
							Type: "domain",
							Name: "math",
						}},
					},
					ModelRefs: []config.ModelRef{{Model: "qwen-math"}},
				}},
			},
		},
	}

	body, err := json.Marshal(map[string]interface{}{
		"routing": map[string]interface{}{
			"signals": map[string]interface{}{
				"domains": []map[string]interface{}{
					{"name": "math", "description": "math"},
					{"name": "general", "description": "general", "mmlu_categories": []string{"other"}},
				},
			},
			"projections": map[string]interface{}{
				"partitions": []map[string]interface{}{{
					"name":        "subject_partition",
					"semantics":   "softmax_exclusive",
					"temperature": 0.1,
					"members":     []string{"math", "general"},
					"default":     "general",
				}},
			},
		},
	})
	if err != nil {
		t.Fatalf("json.Marshal error: %v", err)
	}

	req := httptest.NewRequest(http.MethodPut, "/config/classification", bytes.NewReader(body))
	rr := httptest.NewRecorder()

	apiServer.handleUpdateConfig(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rr.Code, rr.Body.String())
	}
	if len(apiServer.config.Projections.Partitions) != 1 {
		t.Fatalf("expected updated config to include projection partition, got %+v", apiServer.config.Projections.Partitions)
	}
	if len(apiServer.config.Categories) != 2 {
		t.Fatalf("expected updated domains to be applied, got %+v", apiServer.config.Categories)
	}
	if len(apiServer.config.Decisions) != 1 {
		t.Fatalf("expected existing decisions to survive merge, got %+v", apiServer.config.Decisions)
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
