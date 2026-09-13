package extproc

import (
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/historyreset"
)

type stubTriggerSource struct {
	result  historyreset.TriggerResult
	request historyreset.TriggerRequest
	called  int
	absent  bool
}

func (s *stubTriggerSource) TopicContinuity(
	_ context.Context,
	request historyreset.TriggerRequest,
) (historyreset.TriggerResult, bool) {
	s.called++
	s.request = request
	if s.absent {
		return historyreset.TriggerResult{}, false
	}
	return s.result, true
}

func resetDecisionFor(t *testing.T, name string, recovery map[string]interface{}) config.Decision {
	t.Helper()
	configuration := map[string]interface{}{
		"enabled": true,
		"trigger": map[string]interface{}{
			"signal":            "topic_boundary",
			"min_confidence":    0.9,
			"accepted_versions": []string{"v1"},
		},
	}
	if recovery != nil {
		configuration["recovery"] = recovery
	}
	decision := historyResetDecision(t, configuration)
	decision.Name = name
	return *decision
}

// A catalog entry is not a running producer. Activating an enabled policy
// without a wired topic-continuity source would leave every request either
// preserved with missing evidence or rejected, so composition refuses instead.
func TestRouterRefusesToActivateAnEnabledResetWithoutAProducer(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.Decisions = []config.Decision{resetDecisionFor(t, "reset", nil)}

	router := &OpenAIRouter{}
	err := router.verifyHistoryResetRuntime(cfg)
	if err == nil {
		t.Fatal("an enabled policy was activated without a producer")
	}
	if !strings.Contains(err.Error(), config.HistoryResetTriggerUnavailable) {
		t.Fatalf("unexpected error %v", err)
	}

	router.HistoryResetTriggers = &stubTriggerSource{}
	if err = router.verifyHistoryResetRuntime(cfg); err != nil {
		t.Fatalf("a wired producer must satisfy the gate: %v", err)
	}

	// A disabled policy needs no producer.
	disabled := &config.RouterConfig{}
	disabled.Decisions = []config.Decision{*historyResetDecision(t, map[string]interface{}{"enabled": false})}
	if err = (&OpenAIRouter{}).verifyHistoryResetRuntime(disabled); err != nil {
		t.Fatalf("a disabled policy must not require a producer: %v", err)
	}
}

// The recovery store is built once per process, so two routes asking for
// different backends or budgets cannot both be honoured. Refuse the
// configuration rather than letting the first request decide for the rest.
func TestRouterRefusesDisagreeingRecoveryContractsAcrossDecisions(t *testing.T) {
	router := &OpenAIRouter{HistoryResetTriggers: &stubTriggerSource{}}

	cfg := &config.RouterConfig{}
	cfg.Decisions = []config.Decision{
		resetDecisionFor(t, "first", map[string]interface{}{
			"enabled": true, "store": "redis", "max_total_bytes": 1024,
		}),
		resetDecisionFor(t, "second", map[string]interface{}{
			"enabled": true, "store": "valkey", "max_total_bytes": 1024,
		}),
	}
	err := router.verifyHistoryResetRuntime(cfg)
	if err == nil || !strings.Contains(err.Error(), "different context recovery stores") {
		t.Fatalf("disagreeing stores were accepted: %v", err)
	}

	cfg.Decisions[1] = resetDecisionFor(t, "second", map[string]interface{}{
		"enabled": true, "store": "redis", "max_total_bytes": 2048,
	})
	if err = router.verifyHistoryResetRuntime(cfg); err == nil ||
		!strings.Contains(err.Error(), "max_total_bytes") {
		t.Fatalf("disagreeing total budgets were accepted: %v", err)
	}

	cfg.Decisions[1] = resetDecisionFor(t, "second", map[string]interface{}{
		"enabled": true, "store": "redis", "max_total_bytes": 1024,
	})
	if err = router.verifyHistoryResetRuntime(cfg); err != nil {
		t.Fatalf("agreeing contracts were rejected: %v", err)
	}
}

// The producer is asked once the permitted history is resolved, and it
// receives that same view with this request's binding.
func TestTriggerSourceIsAskedForTheResolvedHistory(t *testing.T) {
	source := &stubTriggerSource{result: historyreset.TriggerResult{
		Class:      historyreset.TriggerChange,
		Confidence: 0.95,
		Signal:     "topic_boundary",
		Version:    "v1",
	}}
	router := &OpenAIRouter{HistoryResetTriggers: source}
	request := resetConversation()
	ctx := enabledResetContext(t, request)
	captureOriginalContextHistory(ctx)
	source.result.Binding = historyResetEvidenceBinding(ctx)

	router.prepareContextHistorySteps(ctx, request)
	if source.called != 1 {
		t.Fatalf("the producer was asked %d times, want 1", source.called)
	}
	if source.request.Signal != "topic_boundary" {
		t.Fatalf("the producer received signal %q", source.request.Signal)
	}
	if source.request.Binding != historyResetEvidenceBinding(ctx) {
		t.Fatal("the producer did not receive this request's binding")
	}
	if len(source.request.History.Messages) != 3 {
		t.Fatalf("the producer received %d messages, want the resolved history", len(source.request.History.Messages))
	}

	if err := router.applyContextTransformationPlan(ctx, request); err != nil {
		t.Fatalf("expected the plan to succeed: %v", err)
	}
	if len(request.Messages) != 1 {
		t.Fatalf("the resolved result did not drive a removal: %d messages", len(request.Messages))
	}
}

// A producer that returns no result is missing evidence, not a silent no-op.
func TestAbsentProducerResultIsMissingEvidence(t *testing.T) {
	router := &OpenAIRouter{HistoryResetTriggers: &stubTriggerSource{absent: true}}
	request := resetConversation()
	ctx := enabledResetContext(t, request)
	captureOriginalContextHistory(ctx)

	router.prepareContextHistorySteps(ctx, request)
	if err := router.applyContextTransformationPlan(ctx, request); err != nil {
		t.Fatalf("fail-open must preserve the request: %v", err)
	}
	if len(request.Messages) != 3 {
		t.Fatal("history must be preserved when the producer has no result")
	}
	if ctx.HistoryResetDiagnostics.Reason != historyreset.ReasonEvidenceMissing {
		t.Fatalf("unexpected diagnostics %+v", ctx.HistoryResetDiagnostics)
	}
}
