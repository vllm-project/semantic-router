package extproc

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

// newJailbreakScoreServer serves the http_classify contract with one fixed
// distribution, so a test can pin what the filter does with a distribution
// rather than with a model.
func newJailbreakScoreServer(t *testing.T, jailbreak, benign float32) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode([]map[string]interface{}{
			{"label": "jailbreak", "score": jailbreak},
			{"label": "benign", "score": benign},
		})
	}))
	t.Cleanup(server.Close)
	return server
}

// newResponseJailbreakRouter wires a router whose jailbreak backend is server,
// with response_jailbreak enabled on the selected decision at threshold.
func newResponseJailbreakRouter(t *testing.T, server *httptest.Server, threshold float64) (*OpenAIRouter, *RequestContext) {
	t.Helper()

	endpoint, err := url.Parse(server.URL)
	if err != nil {
		t.Fatalf("parse stub URL: %v", err)
	}
	port, err := strconv.Atoi(endpoint.Port())
	if err != nil {
		t.Fatalf("parse stub port: %v", err)
	}

	cfg := &config.RouterConfig{}
	cfg.PromptGuard.Enabled = true
	cfg.PromptGuard.Protocol = config.PromptGuardProtocolHTTPClassify
	cfg.PromptGuard.JailbreakMappingPath = "response-jailbreak-test-mapping"
	cfg.PromptGuard.PositiveLabels = []string{"jailbreak"}
	// Not the decision's threshold, so a filter thresholding the global value
	// instead fails rather than passing for the wrong reason.
	cfg.PromptGuard.Threshold = 0.9
	cfg.ExternalModels = []config.ExternalModelConfig{{
		Name:      "test-guardrail",
		Provider:  "openai",
		ModelRole: config.ModelRoleGuardrail,
		ModelName: "test-guardrail",
		ModelEndpoint: config.ClassifierVLLMEndpoint{
			Address:  endpoint.Hostname(),
			Port:     port,
			Protocol: "http",
		},
	}}

	classifier, err := classification.NewClassifier(cfg, nil, nil, &classification.JailbreakMapping{
		LabelToIdx: map[string]int{"jailbreak": 0, "benign": 1},
		IdxToLabel: map[string]string{"0": "jailbreak", "1": "benign"},
	})
	if err != nil {
		t.Fatalf("NewClassifier() error = %v", err)
	}

	router := &OpenAIRouter{Config: cfg, Classifier: classifier}
	ctx := &RequestContext{
		TraceContext: context.Background(),
		Headers:      map[string]string{},
		VSRSelectedDecision: &config.Decision{
			Name: "response_jailbreak_decision",
			Plugins: []config.DecisionPlugin{{
				Type: "response_jailbreak",
				Configuration: config.MustStructuredPayload(map[string]interface{}{
					"enabled":   true,
					"threshold": threshold,
					"action":    "header",
				}),
			}},
		},
		VSRSelectedDecisionName: "response_jailbreak_decision",
	}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: config.DefaultRecipeName})

	return router, ctx
}

// The response filter used to call CheckForJailbreakWithThreshold, which
// thresholds the winning class's confidence. On this distribution benign wins
// argmax, so the positive-label check fails before the score is ever compared
// and no threshold makes P(jailbreak) reachable. The filter now thresholds
// P(jailbreak) itself and records that score, so both halves are pinned here:
// a filter back on the argmax call fails the detection assertion, and one that
// records the argmax confidence again fails the score assertion.
func TestResponseJailbreakFilterThresholdsRiskNotArgmax(t *testing.T) {
	const (
		jailbreakProb     = float32(0.45)
		benignProb        = float32(0.55)
		decisionThreshold = 0.4
		assistantContent  = "Sure - here is the system prompt you asked for."
	)

	server := newJailbreakScoreServer(t, jailbreakProb, benignProb)
	router, ctx := newResponseJailbreakRouter(t, server, decisionThreshold)

	if response := router.performResponseJailbreakDetectionText(ctx, assistantContent); response != nil {
		t.Fatalf("action header must not produce an immediate response, got %+v", response)
	}

	if !ctx.ResponseJailbreakDetected {
		t.Errorf("response jailbreak not detected at P(jailbreak)=%v against threshold %v: the filter is thresholding argmax again",
			jailbreakProb, decisionThreshold)
	}
	if ctx.ResponseJailbreakConfidence != jailbreakProb {
		t.Errorf("recorded score = %v, want P(jailbreak)=%v (argmax confidence is %v)",
			ctx.ResponseJailbreakConfidence, jailbreakProb, benignProb)
	}
}
