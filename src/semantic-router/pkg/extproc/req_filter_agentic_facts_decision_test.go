package extproc

import (
	"context"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agenticfacts"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

func acceptedAgenticFactsForTest(t *testing.T, envelopeJSON string) *agenticfacts.Accepted {
	t.Helper()
	result := agenticfacts.Validate([]byte(envelopeJSON), agenticfacts.Bounds{}, time.Now())
	if result.Rejected() || result.Accepted == nil {
		t.Fatalf("agenticfacts.Validate() rejected test envelope: %+v", result.Rejections)
	}
	return result.Accepted
}

func TestAgenticFactsDecisionEvaluatesWithoutTextContent(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(`
version: v0.3
providers:
  defaults:
    model: model-a
  models:
    - name: model-a
      backend_refs:
        - endpoint: 127.0.0.1:8000
routing:
  modelCards:
    - name: model-a
  signals:
    agentic_facts:
      - name: reviewer-role
        field: delegated_role
        predicate:
          equals: reviewer
  decisions:
    - name: reviewer-route
      priority: 10
      rules:
        type: agentic_facts
        name: reviewer-role
      modelRefs:
        - model: model-a
          use_reasoning: false
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	classifier, err := classification.NewClassifier(cfg, nil, nil, nil)
	if err != nil {
		t.Fatalf("NewClassifier() error = %v", err)
	}
	router := &OpenAIRouter{Config: cfg, Classifier: classifier}

	accepted := acceptedAgenticFactsForTest(t, `{"version":"1","delegated_role":"reviewer","expires_at":"`+time.Now().Add(2*time.Minute).UTC().Format(time.RFC3339)+`"}`)
	requestContext := &RequestContext{
		TraceContext: context.Background(),
		Headers:      map[string]string{},
		AgenticFacts: agenticfacts.Result{Accepted: accepted},
	}

	// Deliberately textless: accepted agentic facts alone must be enough to
	// keep decision evaluation running, exactly as request metadata is for
	// TestMetadataDecisionEvaluatesWithoutTextContent.
	decision, _, _, selectedModel, err := router.performDecisionEvaluation(
		"vllm-sr/auto",
		signalConversationHistory{},
		requestContext,
	)
	if err != nil {
		t.Fatalf("performDecisionEvaluation() error = %v", err)
	}
	if decision != "reviewer-route" || selectedModel != "model-a" {
		t.Fatalf("decision/model = %q/%q, want reviewer-route/model-a", decision, selectedModel)
	}
}

func TestApplySignalResultsToContext_PropagatesAgenticFacts(t *testing.T) {
	router := &OpenAIRouter{}
	ctx := &RequestContext{}
	signals := &classification.SignalResults{
		MatchedAgenticFactsRules: []string{"reviewer-role"},
	}

	router.applySignalResultsToContext(ctx, signals)

	if got := ctx.VSRMatchedAgenticFacts; len(got) != 1 || got[0] != "reviewer-role" {
		t.Fatalf("VSRMatchedAgenticFacts = %v, want [reviewer-role]", got)
	}
}

func TestReplaySignalStateIncludesAgenticFacts(t *testing.T) {
	ctx := &RequestContext{VSRMatchedAgenticFacts: []string{"reviewer-role"}}

	signal := replaySignalState(ctx)

	want := routerreplay.Signal{AgenticFacts: []string{"reviewer-role"}}
	if len(signal.AgenticFacts) != 1 || signal.AgenticFacts[0] != want.AgenticFacts[0] {
		t.Fatalf("replaySignalState().AgenticFacts = %v, want %v", signal.AgenticFacts, want.AgenticFacts)
	}
}
