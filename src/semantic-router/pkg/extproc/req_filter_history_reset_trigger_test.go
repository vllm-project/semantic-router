package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/historyreset"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/topiccontinuity"
)

func topicResetPolicy(signal string, minConfidence float64) map[string]interface{} {
	return map[string]interface{}{
		"enabled": true,
		"trigger": map[string]interface{}{
			"signal":            signal,
			"min_confidence":    minConfidence,
			"accepted_versions": []string{topiccontinuity.SchemaVersion},
		},
	}
}

// topicResetRun evaluates the declared topic-continuity rules exactly as the
// request pipeline does, then runs the reset policy over the same request.
func topicResetRun(
	t *testing.T,
	request *llmprotocol.Request,
	policy map[string]interface{},
	rules ...config.TopicContinuityRule,
) *RequestContext {
	t.Helper()
	router := topicContinuityRouter(t, rules...)
	ctx := topicContinuityContext(request)
	ctx.VSRSelectedDecision = historyResetDecision(t, policy)
	bindHistoryResetPolicy(ctx)

	router.evaluateTopicContinuity(ctx)
	router.prepareContextHistorySteps(ctx, request)
	if err := router.applyContextTransformationPlan(ctx, request); err != nil {
		t.Fatalf("the fail-open plan must not fail: %v", err)
	}
	return ctx
}

// An explicit change phrase is the evidence the reset was built for: the
// evaluated rule authorizes removal of the complete prior turn, and only the
// live turn reaches the provider.
func TestTopicChangeResultDrivesTheReset(t *testing.T) {
	request := topicChangeRequest()
	ctx := topicResetRun(t, request, topicResetPolicy("topic_boundary", 0.9),
		config.TopicContinuityRule{Name: "topic_boundary"})

	if len(request.Messages) != 1 || request.Messages[0].Role != llmprotocol.RoleUser {
		t.Fatalf("only the live turn should remain, got %+v", request.Messages)
	}
	diagnostics := ctx.HistoryResetDiagnostics
	if diagnostics == nil || diagnostics.Outcome != historyreset.OutcomeApplied ||
		diagnostics.RemovedTurns != 1 || diagnostics.RemovedMessages != 2 {
		t.Fatalf("unexpected diagnostics %+v", diagnostics)
	}
	if diagnostics.Signal != "topic_boundary" || diagnostics.TriggerClass != historyreset.TriggerChange ||
		diagnostics.Version != topiccontinuity.SchemaVersion {
		t.Fatalf("diagnostics do not describe the evaluated rule: %+v", diagnostics)
	}
}

// Continuation and unknown results never remove anything.
func TestNonChangeTopicResultsPreserveHistory(t *testing.T) {
	cases := []struct {
		name   string
		live   string
		reason string
	}{
		{name: "continuation", live: "ok", reason: historyreset.ReasonEvidenceContinuation},
		{name: "unknown", live: "Can you also make the loader thread safe?", reason: historyreset.ReasonEvidenceUnknown},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			request := topicChangeRequest()
			request.Messages[2] = topicText(llmprotocol.RoleUser, test.live)
			ctx := topicResetRun(t, request, topicResetPolicy("topic_boundary", 0.9),
				config.TopicContinuityRule{Name: "topic_boundary"})
			if len(request.Messages) != 3 {
				t.Fatalf("history must be preserved, got %d messages", len(request.Messages))
			}
			if ctx.HistoryResetDiagnostics.Reason != test.reason {
				t.Fatalf("result %+v gave diagnostics %+v",
					ctx.TopicContinuityResults["topic_boundary"], ctx.HistoryResetDiagnostics)
			}
		})
	}
}

// The trigger names one rule. A different rule's change result is not this
// policy's evidence, so the request keeps its history as missing evidence.
func TestTriggerReadsOnlyTheNamedRule(t *testing.T) {
	request := topicChangeRequest()
	ctx := topicResetRun(t, request, topicResetPolicy("topic_boundary", 0.9),
		config.TopicContinuityRule{Name: "another_rule"})
	if _, ok := ctx.TopicContinuityResults["another_rule"]; !ok {
		t.Fatal("fixture no longer evaluates the declared rule")
	}
	if len(request.Messages) != 3 {
		t.Fatalf("history must be preserved, got %d messages", len(request.Messages))
	}
	if ctx.HistoryResetDiagnostics.Reason != historyreset.ReasonEvidenceMissing {
		t.Fatalf("unexpected diagnostics %+v", ctx.HistoryResetDiagnostics)
	}
}

// The threshold is the operator's: an explicit change scores 0.9, so a policy
// demanding more keeps the history and reports why.
func TestTopicChangeBelowTheThresholdPreservesHistory(t *testing.T) {
	request := topicChangeRequest()
	ctx := topicResetRun(t, request, topicResetPolicy("topic_boundary", 0.95),
		config.TopicContinuityRule{Name: "topic_boundary"})
	if len(request.Messages) != 3 {
		t.Fatalf("history must be preserved, got %d messages", len(request.Messages))
	}
	if ctx.HistoryResetDiagnostics.Reason != historyreset.ReasonEvidenceLowConfidence {
		t.Fatalf("unexpected diagnostics %+v", ctx.HistoryResetDiagnostics)
	}
}

func TestTopicContinuityResultMapping(t *testing.T) {
	change := topiccontinuity.Result{
		SchemaVersion: topiccontinuity.SchemaVersion,
		Signal:        "topic_boundary",
		Class:         topiccontinuity.ClassChange,
		Confidence:    0.9,
		Reason:        topiccontinuity.ReasonExplicitChange,
		Coverage:      topiccontinuity.CoverageFull,
		HistorySource: topiccontinuity.SourceOriginalSnapshot,
	}
	with := func(edit func(*topiccontinuity.Result)) topiccontinuity.Result {
		result := change
		edit(&result)
		return result
	}
	cases := []struct {
		name    string
		result  topiccontinuity.Result
		class   historyreset.TriggerClass
		binding string
	}{
		{name: "change", result: change, class: historyreset.TriggerChange, binding: "snapshot"},
		{name: "continuation", result: with(func(r *topiccontinuity.Result) {
			r.Class, r.Reason = topiccontinuity.ClassContinuation, topiccontinuity.ReasonReference
		}), class: historyreset.TriggerContinuation, binding: "snapshot"},
		{name: "conflicting", result: with(func(r *topiccontinuity.Result) {
			r.Class, r.Reason, r.Confidence = topiccontinuity.ClassUnknown, topiccontinuity.ReasonConflicting, 0
		}), class: historyreset.TriggerConflicting, binding: "snapshot"},
		{name: "unrecognized class", result: with(func(r *topiccontinuity.Result) {
			r.Class = "drift"
		}), class: historyreset.TriggerUnknown, binding: "snapshot"},
		{name: "other provenance", result: with(func(r *topiccontinuity.Result) {
			r.HistorySource = "enriched_request"
		}), class: historyreset.TriggerChange, binding: ""},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			trigger := historyResetTriggerFromTopicContinuity(test.result, "snapshot")
			if trigger.Class != test.class || trigger.Binding != test.binding {
				t.Fatalf("mapped %+v to %+v", test.result, trigger)
			}
			if trigger.Signal != test.result.Signal || trigger.Version != test.result.SchemaVersion ||
				trigger.Confidence != test.result.Confidence || trigger.Fallback != test.result.Fallback {
				t.Fatalf("mapping dropped a field: %+v", trigger)
			}
		})
	}

}
