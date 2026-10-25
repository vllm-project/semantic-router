package services

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A declared truncation is a partial success under on_error: allow and a
// refusal under block. The refusal must survive the service layer, which
// tolerates the truncation sentinel as partial coverage.
func TestPIIServiceTruncationPolicy(t *testing.T) {
	for _, policy := range []string{config.OnErrorAllow, config.OnErrorBlock} {
		for _, mode := range []string{"complete", "truncated", "hard_error"} {
			t.Run(policy+"/"+mode, func(t *testing.T) {
				service := truncationPolicyPIIService(t, policy, mode)
				response, err := service.DetectPII(context.Background(), PIIRequest{
					Text: "Alice has an unscanned suffix", Options: &PIIOptions{MaskEntities: true},
				})
				wantError := mode == "hard_error" || (mode == "truncated" && policy == config.OnErrorBlock)
				if wantError {
					if err == nil || response != nil {
						t.Fatalf("scan must be refused: response=%+v err=%v", response, err)
					}
					return
				}
				if err != nil || response == nil {
					t.Fatalf("expected detections: response=%+v err=%v", response, err)
				}
				if !response.HasPII || len(response.Entities) != 1 || response.ScanIncomplete != (mode == "truncated") {
					t.Fatalf("expected one detection with correct coverage: %+v", response)
				}
			})
		}
	}
}

func truncationPolicyPIIService(t *testing.T, policy, mode string) *ClassificationService {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if mode == "hard_error" {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		body := map[string]any{"spans": []map[string]any{
			{"label": "PERSON", "text": "Alice", "start": 0, "end": 5, "score": 0.99},
		}}
		if mode == "truncated" {
			body["truncated_at"] = 5
		}
		if err := json.NewEncoder(w).Encode(body); err != nil {
			t.Errorf("encode response: %v", err)
		}
	}))
	t.Cleanup(server.Close)
	u, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil {
		t.Fatal(err)
	}
	cfg := &config.RouterConfig{}
	cfg.ExternalModels = []config.ExternalModelConfig{{
		Name: "pii-repro", ModelRole: config.ModelRoleClassification, ModelName: "pii-spans",
		ModelEndpoint: config.ClassifierVLLMEndpoint{Address: u.Hostname(), Port: port, Protocol: "http"},
	}}
	cfg.PIIModel.Backend = &config.RemoteClassifierBackend{
		Protocol: config.RemoteClassifierProtocolHTTPClassify,
		Contract: config.RemoteClassifierContractTokenSpans, Model: "pii-repro",
	}
	cfg.PIIModel.OnError = policy
	cfg.PIIModel.Threshold = 0.5
	cfg.PIIMappingPath = "synthetic-pii-mapping.json"
	mapping := &classification.PIIMapping{
		LabelToIdx: map[string]int{"O": 0, "PERSON": 1},
		IdxToLabel: map[string]string{"0": "O", "1": "PERSON"},
	}
	classifier, err := classification.BuildClassifier(cfg, nil, mapping, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := classifier.Close(); err != nil {
			t.Errorf("close classifier: %v", err)
		}
	})
	return NewClassificationService(classifier, cfg)
}
