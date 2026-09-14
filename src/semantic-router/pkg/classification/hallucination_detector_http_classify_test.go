package classification

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
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

// TestHTTPClassifyHallucination_GoldenFixtures runs the token_spans.v1 golden
// set from #2922 through the hallucination detector. The fixture labels are
// PII names, so they are mapped onto the hallucination label set case by case;
// the mechanics under test (code points to bytes, offset zero, end of text,
// overlap, nesting, rejections) are the same for both consumers.
func TestHTTPClassifyHallucination_GoldenFixtures(t *testing.T) {
	relabel := map[string]string{
		"PERSON": "HALLUCINATED", "EMAIL_ADDRESS": "contradiction", "PHONE_NUMBER": "unsupported_addition",
		"ADDRESS": "fabricated_reference", "URL": "unsupported", "CREDIT_CARD": "contradicted",
		"O": "SUPPORTED", "HUMAN_NAME": "HUMAN_NAME",
	}
	for _, tc := range loadTokenSpansFixtures(t) {
		t.Run(tc.Name, func(t *testing.T) {
			assertFixtureArithmetic(t, tc)
			spans := make([]tokenSpansFixtureSpan, len(tc.Spans))
			for i, sp := range tc.Spans {
				mapped, ok := relabel[sp.Label]
				if !ok {
					t.Fatalf("fixture label %q has no hallucination counterpart", sp.Label)
				}
				sp.Label = mapped
				spans[i] = sp
			}
			var got httpClassifyRequest
			detector := newHTTPClassifyHallucinationDetector(t, func(w http.ResponseWriter, r *http.Request) {
				_ = json.NewDecoder(r.Body).Decode(&got)
				_ = json.NewEncoder(w).Encode(map[string]any{"spans": wireSpans(spans)})
			})
			result, err := detector.ClassifyGrounded(context.Background(), tasks.GroundedTextRequest{Context: "The context the fixture never sees.", Question: "fixture?", Answer: tc.Text})
			if got.Inputs != tc.Text || got.Parameters["context"] == "" || got.Parameters["question"] != "fixture?" {
				t.Fatalf("request = %+v: the answer must be inputs and context/question parameters", got)
			}
			switch tc.Expect {
			case "accept", "accept_or_truncated_at":
				assertFixtureAccepted(t, tokenSpansFixtureCase{Name: tc.Name, Text: tc.Text, Spans: spans}, result.Entities, err)
			case "reject":
				assertFixtureRejected(t, tc, result.Entities, err)
			default:
				t.Fatalf("unknown expectation %q", tc.Expect)
			}
		})
	}
}
