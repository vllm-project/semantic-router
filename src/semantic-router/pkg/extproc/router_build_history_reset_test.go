package extproc

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/historyreset"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/topiccontinuity"
)

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

// A terminally blocked request cannot be changed by any topic result, so the
// evidence is not even resolved: a change result for the trigger's rule is
// present and must neither authorize removal nor reach the diagnostics.
func TestBlockedRequestsIgnoreAChangeResult(t *testing.T) {
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
			// No recovery store is wired, so a policy requiring recovery is
			// unavailable; the reserved-tool and streaming cases are blocked
			// before that check.
			router := &OpenAIRouter{Config: &config.RouterConfig{}}
			request := resetConversation()
			ctx := &RequestContext{
				RequestID:           "blocked",
				SemanticRequest:     request,
				VSRSelectedDecision: recoverableResetDecision(t, nil),
				TopicContinuityResults: map[string]topiccontinuity.Result{
					"topic_boundary": {
						SchemaVersion: topiccontinuity.SchemaVersion, Signal: "topic_boundary",
						Class: topiccontinuity.ClassChange, Confidence: 1,
						Reason: topiccontinuity.ReasonExplicitChange, Coverage: topiccontinuity.CoverageFull,
						HistorySource: topiccontinuity.SourceOriginalSnapshot,
					},
				},
			}
			bindHistoryResetPolicy(ctx)
			test.prepare(t, ctx, request)

			router.prepareContextHistorySteps(ctx, request)
			if ctx.HistoryResetTrigger != nil {
				t.Fatalf("a blocked request resolved evidence: %+v", ctx.HistoryResetTrigger)
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
			if len(ctx.ContextRecoveryKeys) != 0 {
				t.Fatal("a blocked request must not perform recovery I/O")
			}
		})
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
