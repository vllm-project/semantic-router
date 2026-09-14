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

// newHTTPClassifyHallucinationDetector declares a token_spans.v1 grounding
// service through the binding plan and returns the detector built for it.
func newHTTPClassifyHallucinationDetector(t *testing.T, handler http.HandlerFunc) *EndpointHallucinationDetector {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	cfg := &config.RouterConfig{}
	cfg.ExternalModels = []config.ExternalModelConfig{{Name: "grounding", ModelName: "grounding-spans", ModelRole: config.ModelRoleClassification, ModelEndpoint: endpointForTestServer(t, server)}}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"grounding": {Provider: "http", ExternalModel: "grounding"}}
	cfg.ModelBindings = map[string]config.ModelBinding{"hallucination_detector": {Deployment: "grounding", Adapter: config.RemoteClassifierProtocolHTTPClassify, Contract: config.RemoteClassifierContractTokenSpans}}
	models, err := newClassifierModelRuntime(cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	detector, err := NewEndpointHallucinationDetector(&models.cfg.HallucinationMitigation.HallucinationModel, models)
	if err != nil {
		t.Fatalf("NewEndpointHallucinationDetector: %v", err)
	}
	if err := detector.Initialize(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = detector.Close() })
	return detector
}

func TestHTTPClassifyHallucination_AnswerIsInputsAndOffsetsIndexAnswer(t *testing.T) {
	const (
		contextText = "The café opened in 2001."
		question    = "When did it open?"
		answer      = "Café opened in 1999." // é is two bytes: code point 15 is byte 16
	)
	var got httpClassifyRequest
	detector := newHTTPClassifyHallucinationDetector(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/classify" {
			t.Errorf("path = %q", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Error(err)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "grounding-spans",
			"spans": []map[string]any{{"label": "HALLUCINATED", "start": 15, "end": 19, "text": "1999", "score": 0.91}},
		})
	})
	result, err := detector.DetectWithNLI(context.Background(), contextText, question, answer)
	if err != nil {
		t.Fatal(err)
	}
	if got.Inputs != answer || got.Parameters["context"] != contextText || got.Parameters["question"] != question {
		t.Fatalf("request = %+v", got)
	}
	if !result.HallucinationDetected || len(result.Spans) != 1 {
		t.Fatalf("result = %+v", result)
	}
	span := result.Spans[0]
	if span.Text != "1999" || span.Start != 16 || span.End != 20 || answer[span.Start:span.End] != span.Text {
		t.Fatalf("span = %+v", span)
	}
}

func TestHTTPClassifyHallucination_OutsideLabelIsAnError(t *testing.T) {
	detector := newHTTPClassifyHallucinationDetector(t, func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode([]map[string]any{{"label": "SUPPORTED", "start": 0, "end": 4, "text": "Café", "score": 0.5}})
	})
	_, err := detector.DetectWithNLI(context.Background(), "context", "q", "Café opened.")
	if err == nil || !strings.Contains(err.Error(), "outside label") {
		t.Fatalf("err = %v", err)
	}
}

func TestHTTPClassifyHallucination_TruncatedScanIsNotClean(t *testing.T) {
	detector := newHTTPClassifyHallucinationDetector(t, func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"spans": []any{}, "truncated_at": 4})
	})
	_, err := detector.DetectWithNLI(context.Background(), "context", "q", "Café opened.")
	if err == nil {
		t.Fatal("expected a partial-scan error, got a clean verdict")
	}
}
