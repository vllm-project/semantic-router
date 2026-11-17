package services

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// piiPresetDecider answers the pii preset with one PERSON span, read whole or
// in part, or fails the call.
type piiPresetDecider struct{ mode string }

func (d piiPresetDecider) Decide(_ context.Context, _ string, request modelservice.Request) (modelservice.Response, error) {
	if d.mode == "hard_error" {
		return modelservice.Response{}, modelservice.ErrUnavailable
	}
	coverage := "complete"
	if d.mode == "truncated" {
		coverage = "partial"
	}
	answers := map[string]modelservice.Answer{}
	for _, question := range request.Questions {
		answers[question.ID] = modelservice.Answer{
			Type: "span", InputCoverage: coverage,
			Spans: []modelservice.Span{{Label: "PERSON", Text: "Alice", Start: 0, End: 5, Probability: 0.99}},
		}
	}
	return modelservice.Response{Answers: answers}, nil
}

// The PII API reports the spans the decision model found; a scan that read
// only part of the text is reported as incomplete rather than as clean, and a
// failed call is an error.
func TestPIIServiceReportsAPartialScan(t *testing.T) {
	for _, mode := range []string{"complete", "truncated", "hard_error"} {
		t.Run(mode, func(t *testing.T) {
			cfg := &config.RouterConfig{}
			classifier, err := classification.BuildClassifier(cfg)
			if err != nil {
				t.Fatal(err)
			}
			classifier.SetDecisionDecider(piiPresetDecider{mode: mode})
			t.Cleanup(func() { _ = classifier.Close() })
			response, err := NewClassificationService(classifier, cfg).DetectPII(context.Background(), PIIRequest{
				Text: "Alice has an unscanned suffix", Options: &PIIOptions{MaskEntities: true},
			})
			if mode == "hard_error" {
				if err == nil || response != nil {
					t.Fatalf("a failed call must be an error: response=%+v err=%v", response, err)
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
