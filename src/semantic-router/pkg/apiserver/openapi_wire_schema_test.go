//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestOpenAPIWireRepresentationMatchesNullableScoreEncoding(t *testing.T) {
	for _, available := range []bool{false, true} {
		for _, test := range []struct {
			name, field string
			value       any
		}{
			{"classification", "confidence", services.Classification{ConfidenceAvailable: &available}},
			{"decision", "confidence", services.DecisionResult{ConfidenceAvailable: &available}},
			{"fact-check", "confidence", services.FactCheckResponse{ConfidenceAvailable: available}},
			{"feedback", "confidence", services.UserFeedbackResponse{ConfidenceAvailable: available}},
			{"signal-metric", "confidence", classification.SignalMetrics{ConfidenceAvailable: &available}},
			{"replay", "confidence_score", store.Record{ConfidenceScoreAvailable: available}},
		} {
			t.Run(test.name+"/"+map[bool]string{true: "available-zero", false: "unavailable"}[available], func(t *testing.T) {
				encoded, err := json.Marshal(test.value)
				if err != nil {
					t.Fatal(err)
				}
				var response map[string]any
				if err = json.Unmarshal(encoded, &response); err != nil {
					t.Fatal(err)
				}
				value, exists := response[test.field]
				if !exists || (!available && value != nil) || (available && value != float64(0)) {
					t.Fatalf("wrong wire score: %s", encoded)
				}
				schema := openAPISchemaFromType(reflect.TypeOf(test.value), make(map[reflect.Type]bool))
				field := schema.Properties[test.field]
				if field.Type != "number" || !field.Nullable {
					t.Fatalf("wire score %s disagrees with schema %+v", encoded, field)
				}
				count := 0
				for _, name := range schema.Required {
					if name == test.field {
						count++
					}
				}
				if count != 1 {
					t.Fatalf("wire score must occur once in required: %v", schema.Required)
				}
			})
		}
	}
}

func TestOpenAPIReplayHTTPMatchesScoreWireSchema(t *testing.T) {
	body, err := json.Marshal(routerreplay.ListResponse{Data: []routerreplay.RoutingRecord{
		{ID: "unavailable"},
		{ID: "available-zero", ConfidenceScoreAvailable: true, JailbreakScoreAvailable: true, ResponseJailbreakScoreAvailable: true},
	}})
	if err != nil {
		t.Fatal(err)
	}
	registry := routerruntime.NewRegistry(nil)
	registry.SetReplayRuntime(&replayRuntimeStub{handled: true, response: routerruntime.ReplayResponse{StatusCode: http.StatusOK, Body: body}})
	api := &ClassificationAPIServer{runtimeRegistry: registry}
	response := httptest.NewRecorder()
	api.setupRoutes().ServeHTTP(response, httptest.NewRequest(http.MethodGet, apiObservabilityReplaysPath, nil))
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var actual struct {
		Data []map[string]any `json:"data"`
	}
	if err = json.Unmarshal(response.Body.Bytes(), &actual); err != nil {
		t.Fatal(err)
	}
	schema := api.generateOpenAPISpec().Paths[apiObservabilityReplaysPath].Get.Responses["200"].Content["application/json"].Schema.Properties["data"].Items
	for _, name := range []string{"confidence_score", "jailbreak_confidence", "response_jailbreak_confidence"} {
		field := schema.Properties[name]
		if field.Type != "number" || !field.Nullable || actual.Data[1][name] != float64(0) {
			t.Fatalf("available-zero %s: schema=%+v actual=%v", name, field, actual.Data[1])
		}
		value, exists := actual.Data[0][name]
		if name == "confidence_score" {
			if !exists || value != nil || !slices.Contains(schema.Required, name) {
				t.Fatalf("unavailable score must be required null: schema=%+v actual=%v", field, actual.Data[0])
			}
		} else if exists || slices.Contains(schema.Required, name) {
			t.Fatalf("unavailable guard score must be omitted: required=%v actual=%v", schema.Required, actual.Data[0])
		}
	}
	metrics := api.generateOpenAPISpec().Paths[apiRoutingPreviewPath].Post.Responses["200"].Content["application/json"].Schema.Properties["metrics"].Properties["keyword"].Properties["confidence"]
	if metrics.Type != "number" || !metrics.Nullable {
		t.Fatalf("nested routing metrics must use wire representation: %+v", metrics)
	}
}
