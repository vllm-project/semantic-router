package extproc

import (
	"context"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/historyreset"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
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
	err := router.verifyHistoryResetTriggerWiring(cfg)
	if err == nil {
		t.Fatal("an enabled policy was activated without a producer")
	}
	if !strings.Contains(err.Error(), config.HistoryResetTriggerUnavailable) {
		t.Fatalf("unexpected error %v", err)
	}

	router.HistoryResetTriggers = &stubTriggerSource{}
	if err = router.verifyHistoryResetTriggerWiring(cfg); err != nil {
		t.Fatalf("a wired producer must satisfy the gate: %v", err)
	}

	// A disabled policy needs no producer.
	disabled := &config.RouterConfig{}
	disabled.Decisions = []config.Decision{*historyResetDecision(t, map[string]interface{}{"enabled": false})}
	if err = (&OpenAIRouter{}).verifyHistoryResetTriggerWiring(disabled); err != nil {
		t.Fatalf("a disabled policy must not require a producer: %v", err)
	}
}

// The recovery store is built once per process, so two routes asking for
// different backends or budgets cannot both be honoured. Refuse the
// configuration rather than letting the first request decide for the rest.
func TestRouterRefusesDisagreeingRecoveryContractsAcrossDecisions(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.Decisions = []config.Decision{
		resetDecisionFor(t, "first", map[string]interface{}{
			"enabled": true, "store": "redis", "max_total_bytes": 1024,
		}),
		resetDecisionFor(t, "second", map[string]interface{}{
			"enabled": true, "store": "valkey", "max_total_bytes": 1024,
		}),
	}
	err := verifyContextRecoveryAgreement(cfg)
	if err == nil || !strings.Contains(err.Error(), "different context recovery stores") {
		t.Fatalf("disagreeing stores were accepted: %v", err)
	}

	cfg.Decisions[1] = resetDecisionFor(t, "second", map[string]interface{}{
		"enabled": true, "store": "redis", "max_total_bytes": 2048,
	})
	if err = verifyContextRecoveryAgreement(cfg); err == nil ||
		!strings.Contains(err.Error(), "max_total_bytes") {
		t.Fatalf("disagreeing total budgets were accepted: %v", err)
	}

	cfg.Decisions[1] = resetDecisionFor(t, "second", map[string]interface{}{
		"enabled": true, "store": "redis", "max_total_bytes": 1024,
	})
	if err = verifyContextRecoveryAgreement(cfg); err != nil {
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

// A terminally blocked request cannot be changed by any topic result, so the
// producer must not be asked: evaluation can be expensive, and its answer
// could only be discarded.
func TestBlockedRequestsNeverInvokeTheProducer(t *testing.T) {
	cases := []struct {
		name    string
		prepare func(t *testing.T, ctx *RequestContext, request *llmprotocol.Request)
		reason  string
	}{
		{
			name: "history_unresolved",
			prepare: func(_ *testing.T, _ *RequestContext, _ *llmprotocol.Request) {
				// Leave the original history unresolved.
			},
			reason: historyreset.ReasonHistoryUnresolved,
		},
		{
			name: "reserved_tool_conflict",
			prepare: func(_ *testing.T, ctx *RequestContext, request *llmprotocol.Request) {
				captureOriginalContextHistory(ctx)
				request.Tools = []llmprotocol.Tool{{Name: contextcompression.RetrieveToolName}}
			},
			reason: historyreset.ReasonReservedToolConflict,
		},
		{
			name: "streaming_with_required_recovery",
			prepare: func(_ *testing.T, ctx *RequestContext, _ *llmprotocol.Request) {
				captureOriginalContextHistory(ctx)
				ctx.ExpectStreamingResponse = true
			},
			reason: historyreset.ReasonStreamingUnsupported,
		},
		{
			name: "recovery_unavailable",
			prepare: func(_ *testing.T, ctx *RequestContext, _ *llmprotocol.Request) {
				captureOriginalContextHistory(ctx)
			},
			reason: historyreset.ReasonRecoveryUnavailable,
		},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			source := &stubTriggerSource{result: historyreset.TriggerResult{
				Class: historyreset.TriggerChange, Confidence: 1,
				Signal: "topic_boundary", Version: "v1",
			}}
			// No recovery store is wired, so a policy requiring recovery is
			// unavailable; the reserved-tool and streaming cases are blocked
			// before that check.
			router := &OpenAIRouter{Config: &config.RouterConfig{}, HistoryResetTriggers: source}
			request := resetConversation()
			ctx := &RequestContext{
				RequestID:           "blocked",
				SemanticRequest:     request,
				VSRSelectedDecision: recoverableResetDecision(t, nil),
			}
			bindHistoryResetPolicy(ctx)
			test.prepare(t, ctx, request)

			router.prepareContextHistorySteps(ctx, request)
			if source.called != 0 {
				t.Fatalf("the producer was asked %d times for a blocked request", source.called)
			}
			if err := router.applyContextTransformationPlan(ctx, request); err != nil {
				t.Fatalf("fail-open must preserve the request: %v", err)
			}
			if len(request.Messages) != 3 {
				t.Fatalf("a blocked request must keep its history, got %d", len(request.Messages))
			}
			if ctx.HistoryResetDiagnostics.Reason != test.reason {
				t.Fatalf("unexpected diagnostics %+v", ctx.HistoryResetDiagnostics)
			}
			if len(ctx.ContextCompressionRecoveryKeys) != 0 {
				t.Fatal("a blocked request must not perform recovery I/O")
			}
		})
	}
}

// A candidate router that fails activation must release everything it built,
// or a rejected reload leaks resources on every attempt while the previous
// router keeps serving.
func TestRejectedCandidateRouterReleasesItsResources(t *testing.T) {
	closed := false
	router := &OpenAIRouter{resources: &resourceScope{}}
	router.resources.add(func() error {
		closed = true
		return nil
	})

	cfg := &config.RouterConfig{}
	cfg.Decisions = []config.Decision{resetDecisionFor(t, "reset", nil)}
	if err := router.verifyHistoryResetTriggerWiring(cfg); err == nil {
		t.Fatal("expected the candidate to be rejected")
	}
	if err := router.Close(); err != nil {
		t.Fatalf("closing the rejected candidate failed: %v", err)
	}
	if !closed {
		t.Fatal("the rejected candidate did not release its resources")
	}
}

// Configuration-only disagreement is caught before any component is built.
func TestRecoveryAgreementIsCheckedBeforeComponentsAreBuilt(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.Decisions = []config.Decision{
		resetDecisionFor(t, "first", map[string]interface{}{
			"enabled": true, "store": "redis",
		}),
		resetDecisionFor(t, "second", map[string]interface{}{
			"enabled": true, "store": "valkey",
		}),
	}
	if _, err := buildOpenAIRouterFromConfig(cfg); err == nil ||
		!strings.Contains(err.Error(), "different context recovery stores") {
		t.Fatalf("construction accepted disagreeing recovery stores: %v", err)
	}
}

// The rejection path is exercised through the real constructor, and the test
// fails if the release step is ever dropped: a rejected reload that leaks its
// components would otherwise go unnoticed.
func TestConstructionRejectsAnEnabledPolicyWithoutAProducer(t *testing.T) {
	released := 0
	cfg := &config.RouterConfig{}
	cfg.Decisions = []config.Decision{resetDecisionFor(t, "reset", nil)}

	router, err := buildOpenAIRouterFromConfigWithCloser(cfg, func(candidate *OpenAIRouter) error {
		released++
		return candidate.Close()
	})
	if err == nil {
		if router != nil {
			_ = router.Close()
		}
		t.Fatal("construction accepted an enabled policy with no producer")
	}
	if router != nil {
		t.Fatal("a rejected construction must not return a usable router")
	}
	if !strings.Contains(err.Error(), config.HistoryResetTriggerUnavailable) {
		t.Fatalf("unexpected error %v", err)
	}
	if released != 1 {
		t.Fatalf("the rejected candidate was released %d times, want 1", released)
	}
}
