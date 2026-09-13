package historyreset

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
)

func testPolicy() Policy {
	return Policy{
		Signal:           "topic_boundary",
		MinConfidence:    0.9,
		AcceptedVersions: []string{"v1"},
	}
}

func acceptedChange() TriggerResult {
	return TriggerResult{
		Class:      TriggerChange,
		Confidence: 0.95,
		Signal:     "topic_boundary",
		Version:    "v1",
	}
}

// historyMessage builds one eligible prior-history message.
func historyMessage(id, turn int, role string) contextcompression.MessageView {
	return contextcompression.MessageView{
		ID:          id,
		Role:        role,
		Source:      contextcompression.SourceHistory,
		Eligibility: contextcompression.EligibleHistoryRemoval,
		TurnID:      turn,
		Blocks:      []contextcompression.BlockView{{ID: 0, Text: "prior"}},
	}
}

func protect(
	message contextcompression.MessageView,
	protection contextcompression.Protection,
) contextcompression.MessageView {
	message.Protection = protection
	message.Eligibility = 0
	return message
}

func withExchange(
	message contextcompression.MessageView,
	ids ...string,
) contextcompression.MessageView {
	message.ExchangeIDs = ids
	return message
}

func planIDs(t *testing.T, policy Policy, trigger TriggerResult, messages ...contextcompression.MessageView) ([]int, Diagnostics) {
	t.Helper()
	edits, diagnostics := Plan(
		context.Background(),
		policy,
		trigger,
		contextcompression.TransformationView{Messages: messages},
	)
	return edits.RemoveMessages, diagnostics
}

func TestPlanRemovesCompleteEligibleTurns(t *testing.T) {
	ids, diagnostics := planIDs(t, testPolicy(), acceptedChange(),
		historyMessage(0, 0, "user"),
		historyMessage(1, 0, "assistant"),
		historyMessage(2, 2, "user"),
		historyMessage(3, 2, "assistant"),
		protect(historyMessage(4, 4, "user"), contextcompression.ProtectLiveTurn),
	)
	if len(ids) != 4 || ids[0] != 0 || ids[3] != 3 {
		t.Fatalf("expected both prior turns, got %v", ids)
	}
	if diagnostics.Outcome != OutcomeApplied || diagnostics.Reason != ReasonApplied {
		t.Fatalf("unexpected outcome %+v", diagnostics)
	}
	if diagnostics.RemovedTurns != 2 || diagnostics.RemovedMessages != 4 {
		t.Fatalf("unexpected removal counts %+v", diagnostics)
	}
	if diagnostics.ExaminedMessages != 5 || diagnostics.RetainedMessages != 1 {
		t.Fatalf("unexpected examined/retained counts %+v", diagnostics)
	}
	if diagnostics.ProtectedMessages != 1 {
		t.Fatalf("unexpected protected count %+v", diagnostics)
	}
}

func TestPlanKeepsTurnsThatMixProtectedContent(t *testing.T) {
	ids, diagnostics := planIDs(t, testPolicy(), acceptedChange(),
		historyMessage(0, 0, "user"),
		protect(historyMessage(1, 0, "assistant"), contextcompression.ProtectMultimodal),
		historyMessage(2, 2, "user"),
		historyMessage(3, 2, "assistant"),
	)
	if len(ids) != 2 || ids[0] != 2 || ids[1] != 3 {
		t.Fatalf("expected only the fully eligible turn, got %v", ids)
	}
	if diagnostics.RemovedTurns != 1 {
		t.Fatalf("unexpected removed turn count %+v", diagnostics)
	}
}

func TestPlanKeepsTurnsLinkedToRetainedToolExchanges(t *testing.T) {
	ids, _ := planIDs(t, testPolicy(), acceptedChange(),
		withExchange(historyMessage(0, 0, "assistant"), "call-1"),
		withExchange(historyMessage(1, 0, "tool"), "call-1"),
		withExchange(historyMessage(2, 2, "assistant"), "call-1"),
		protect(withExchange(historyMessage(3, 2, "tool"), "call-1"), contextcompression.ProtectLiveTurn),
	)
	if len(ids) != 0 {
		t.Fatalf("a retained tool result must pin its whole exchange, got %v", ids)
	}
}

func TestPlanClosesRetentionTransitively(t *testing.T) {
	// Turn 4 is pinned by the live turn through call-2, and turn 2 is then
	// pinned by turn 4 through call-1. Only turn 0 may be removed.
	ids, _ := planIDs(t, testPolicy(), acceptedChange(),
		historyMessage(0, 0, "user"),
		withExchange(historyMessage(2, 2, "assistant"), "call-1"),
		withExchange(historyMessage(4, 4, "assistant"), "call-1", "call-2"),
		protect(withExchange(historyMessage(6, 6, "tool"), "call-2"), contextcompression.ProtectLiveTurn),
	)
	if len(ids) != 1 || ids[0] != 0 {
		t.Fatalf("expected only the unlinked turn, got %v", ids)
	}
}

func TestPlanIgnoresEnrichedAndUnknownContent(t *testing.T) {
	rag := historyMessage(1, 0, "tool")
	rag.Source = contextcompression.SourceRAG
	rag.Eligibility = 0

	unknown := historyMessage(2, -1, "assistant")
	unknown.Protection = contextcompression.ProtectUnknown
	unknown.Eligibility = 0

	ids, _ := planIDs(t, testPolicy(), acceptedChange(),
		historyMessage(0, 0, "user"),
		rag,
		unknown,
	)
	if len(ids) != 0 {
		t.Fatalf("enriched or unknown content must pin its turn, got %v", ids)
	}
}

func TestPlanSkipsWithoutAuthorizingEvidence(t *testing.T) {
	cases := []struct {
		name    string
		trigger TriggerResult
		reason  string
	}{
		{"missing", TriggerResult{}, ReasonEvidenceMissing},
		{
			"unversioned",
			TriggerResult{Class: TriggerChange, Confidence: 1, Signal: "topic_boundary"},
			ReasonEvidenceUnsupportedVersion,
		},
		{"continuation", TriggerResult{Class: TriggerContinuation, Signal: "topic_boundary", Version: "v1"}, ReasonEvidenceContinuation},
		{"unknown", TriggerResult{Class: TriggerUnknown, Signal: "topic_boundary", Version: "v1"}, ReasonEvidenceUnknown},
		{
			"low_confidence",
			TriggerResult{Class: TriggerChange, Confidence: 0.5, Signal: "topic_boundary", Version: "v1"},
			ReasonEvidenceLowConfidence,
		},
		{
			"fallback",
			TriggerResult{
				Class: TriggerChange, Confidence: 1, Signal: "topic_boundary",
				Version: "v1", Fallback: true,
			},
			ReasonEvidenceFallback,
		},
		{
			"wrong_signal",
			TriggerResult{Class: TriggerChange, Confidence: 1, Signal: "other", Version: "v1"},
			ReasonEvidenceWrongSignal,
		},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			ids, diagnostics := planIDs(t, testPolicy(), test.trigger,
				historyMessage(0, 0, "user"),
				historyMessage(1, 0, "assistant"),
			)
			if len(ids) != 0 {
				t.Fatalf("expected no removal, got %v", ids)
			}
			if diagnostics.Outcome != OutcomeSkipped || diagnostics.Reason != test.reason {
				t.Fatalf("expected %q, got %+v", test.reason, diagnostics)
			}
		})
	}
}

func TestPlanRejectsUnsupportedEvidenceVersion(t *testing.T) {
	policy := testPolicy()
	policy.AcceptedVersions = []string{"v2"}
	ids, diagnostics := planIDs(t, policy, acceptedChange(), historyMessage(0, 0, "user"))
	if len(ids) != 0 || diagnostics.Reason != ReasonEvidenceUnsupportedVersion {
		t.Fatalf("expected an unsupported-version skip, got %v %+v", ids, diagnostics)
	}
}

func TestPlanRejectsTheWholeStepWhenLimitsAreExceeded(t *testing.T) {
	policy := testPolicy()
	policy.MaxHistoryTurns = 1
	ids, diagnostics := planIDs(t, policy, acceptedChange(),
		historyMessage(0, 0, "user"),
		historyMessage(1, 2, "user"),
	)
	if len(ids) != 0 || diagnostics.Reason != ReasonHistoryLimitExceeded {
		t.Fatalf("expected a whole-step skip, got %v %+v", ids, diagnostics)
	}

	policy = testPolicy()
	policy.MaxHistoryBytes = 1
	if ids, diagnostics = planIDs(t, policy, acceptedChange(), historyMessage(0, 0, "user")); len(ids) != 0 ||
		diagnostics.Reason != ReasonHistoryLimitExceeded {
		t.Fatalf("expected a byte-bound skip, got %v %+v", ids, diagnostics)
	}
}

func TestPlanHonoursCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	edits, diagnostics := Plan(ctx, testPolicy(), acceptedChange(),
		contextcompression.TransformationView{Messages: []contextcompression.MessageView{
			historyMessage(0, 0, "user"),
		}},
	)
	if len(edits.RemoveMessages) != 0 || diagnostics.Reason != ReasonCancelled {
		t.Fatalf("expected a cancelled skip, got %+v %+v", edits, diagnostics)
	}
}

func TestPlanReportsNoEligibleHistory(t *testing.T) {
	ids, diagnostics := planIDs(t, testPolicy(), acceptedChange(),
		protect(historyMessage(0, 0, "user"), contextcompression.ProtectLiveTurn),
	)
	if len(ids) != 0 || diagnostics.Reason != ReasonNoEligibleHistory {
		t.Fatalf("expected no_eligible_history, got %v %+v", ids, diagnostics)
	}
	if diagnostics.Outcome != OutcomeSkipped {
		t.Fatalf("expected a skipped outcome, got %+v", diagnostics)
	}
}

// Every evidence condition the issue names must behave differently in the two
// failure modes: fail-open preserves the request and records why, fail-closed
// stops the plan before the provider is called. A continuation and an empty
// eligible set stay ordinary no-ops in both.
func TestEvidenceOutcomesFollowTheConfiguredFailureMode(t *testing.T) {
	bound := func(class TriggerClass, confidence float64) TriggerResult {
		return TriggerResult{
			Class: class, Confidence: confidence,
			Signal: "topic_boundary", Version: "v1",
		}
	}
	cases := []struct {
		name    string
		trigger TriggerResult
		reason  string
		rejects bool
		outcome Outcome
	}{
		{"missing", TriggerResult{}, ReasonEvidenceMissing, true, OutcomeFailed},
		{"unknown", bound(TriggerUnknown, 1), ReasonEvidenceUnknown, true, OutcomeFailed},
		{"conflicting", bound(TriggerConflicting, 1), ReasonEvidenceConflicting, true, OutcomeFailed},
		{"low_confidence", bound(TriggerChange, 0.1), ReasonEvidenceLowConfidence, true, OutcomeFailed},
		{
			"unsupported_version",
			TriggerResult{Class: TriggerChange, Confidence: 1, Signal: "topic_boundary"},
			ReasonEvidenceUnsupportedVersion, true, OutcomeFailed,
		},
		{
			"wrong_signal",
			TriggerResult{Class: TriggerChange, Confidence: 1, Signal: "other", Version: "v1"},
			ReasonEvidenceWrongSignal, true, OutcomeFailed,
		},
		{
			"fallback",
			TriggerResult{
				Class: TriggerChange, Confidence: 1,
				Signal: "topic_boundary", Version: "v1", Fallback: true,
			},
			ReasonEvidenceFallback, true, OutcomeFailed,
		},
		{"continuation", bound(TriggerContinuation, 1), ReasonEvidenceContinuation, false, OutcomeSkipped},
	}
	for _, test := range cases {
		for _, failClosed := range []bool{false, true} {
			name := test.name + "_fail_open"
			if failClosed {
				name = test.name + "_fail_closed"
			}
			t.Run(name, func(t *testing.T) {
				policy := testPolicy()
				policy.FailClosed = failClosed
				request := conversation()
				action := NewAction(policy, test.trigger, "")

				ir, err := applyAction(t, request, action)
				if (err != nil) != (failClosed && test.rejects) {
					t.Fatalf("rejection = %v, want %v (err=%v)", err != nil, failClosed && test.rejects, err)
				}
				if len(request.Messages) != 5 || request.Generation != 1 {
					t.Fatalf("the request must be preserved unchanged, got %d messages", len(request.Messages))
				}
				diagnostics := action.Reconcile(ir.Transformations.Receipts())
				if diagnostics.Reason != test.reason || diagnostics.Outcome != test.outcome {
					t.Fatalf("unexpected diagnostics %+v", diagnostics)
				}
				if diagnostics.RemovedMessages != 0 {
					t.Fatalf("a non-removal outcome cannot report removals: %+v", diagnostics)
				}
			})
		}
	}
}

// An accepted change with nothing eligible left to remove is a normal no-op,
// not an evaluation failure, so fail-closed must not reject it.
func TestEmptyEligibleHistoryIsNeverAFailure(t *testing.T) {
	for _, failClosed := range []bool{false, true} {
		policy := testPolicy()
		policy.FailClosed = failClosed
		request := conversation()
		request.Messages = request.Messages[4:]
		action := NewAction(policy, acceptedChange(), "")

		ir, err := applyAction(t, request, action)
		if err != nil {
			t.Fatalf("fail_closed=%v rejected an ordinary no-op: %v", failClosed, err)
		}
		diagnostics := action.Reconcile(ir.Transformations.Receipts())
		if diagnostics.Outcome != OutcomeSkipped || diagnostics.Reason != ReasonNoEligibleHistory {
			t.Fatalf("unexpected diagnostics %+v", diagnostics)
		}
	}
}

// The configured planning budget must actually bound the evaluation.
// The action owns the budget, so the planner honours the deadline it is given.
func TestPlanningStopsAtTheConfiguredTimeout(t *testing.T) {
	policy := testPolicy()
	policy.Timeout = time.Nanosecond

	messages := make([]contextcompression.MessageView, 0, 256)
	for index := 0; index < 256; index++ {
		messages = append(messages, historyMessage(index, index/2*2, "user"))
	}
	ctx, cancel := context.WithTimeout(context.Background(), policy.Timeout)
	defer cancel()
	edits, diagnostics := Plan(
		ctx,
		policy,
		acceptedChange(),
		contextcompression.TransformationView{Messages: messages},
	)
	if len(edits.RemoveMessages) != 0 {
		t.Fatalf("an expired budget must not produce a removal, got %d ids", len(edits.RemoveMessages))
	}
	if diagnostics.Reason != ReasonCancelled {
		t.Fatalf("unexpected diagnostics %+v", diagnostics)
	}
}

// A caller's cancellation is honoured even when the policy sets no budget.
func TestPlanningHonoursCallerCancellationDuringSelection(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	messages := make([]contextcompression.MessageView, 0, 128)
	for index := 0; index < 128; index++ {
		messages = append(messages, historyMessage(index, index/2*2, "user"))
	}
	_, diagnostics := Plan(
		ctx,
		testPolicy(),
		acceptedChange(),
		contextcompression.TransformationView{Messages: messages},
	)
	if diagnostics.Reason != ReasonCancelled {
		t.Fatalf("unexpected diagnostics %+v", diagnostics)
	}
}

// A confidence that cannot be compared with the threshold must be rejected:
// NaN fails every ordered comparison and would otherwise slip through.
func TestUnusableConfidenceCannotAuthorizeRemoval(t *testing.T) {
	for name, confidence := range map[string]float64{
		"nan":          math.NaN(),
		"positive_inf": math.Inf(1),
		"negative_inf": math.Inf(-1),
		"above_one":    1.5,
		"below_zero":   -0.5,
	} {
		t.Run(name, func(t *testing.T) {
			trigger := acceptedChange()
			trigger.Confidence = confidence
			edits, diagnostics := Plan(
				context.Background(),
				testPolicy(),
				trigger,
				contextcompression.TransformationView{Messages: []contextcompression.MessageView{
					historyMessage(0, 0, "user"),
					historyMessage(1, 0, "assistant"),
				}},
			)
			if len(edits.RemoveMessages) != 0 {
				t.Fatalf("unusable confidence authorized %d removals", len(edits.RemoveMessages))
			}
			if diagnostics.Reason != ReasonEvidenceInvalidConfidence {
				t.Fatalf("unexpected diagnostics %+v", diagnostics)
			}
		})
	}
}

// A policy that has not declared which producer contracts it trusts cannot
// verify compatibility, so it must not act on any result.
func TestUndeclaredAcceptedVersionsRejectEveryResult(t *testing.T) {
	policy := testPolicy()
	policy.AcceptedVersions = nil
	edits, diagnostics := Plan(
		context.Background(),
		policy,
		acceptedChange(),
		contextcompression.TransformationView{Messages: []contextcompression.MessageView{
			historyMessage(0, 0, "user"),
		}},
	)
	if len(edits.RemoveMessages) != 0 || diagnostics.Reason != ReasonEvidenceUnsupportedVersion {
		t.Fatalf("unexpected outcome %v %+v", edits.RemoveMessages, diagnostics)
	}
}
