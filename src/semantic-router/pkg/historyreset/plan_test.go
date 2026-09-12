package historyreset

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
)

func testPolicy() Policy {
	return Policy{Signal: "topic_boundary", MinConfidence: 0.9}
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
		{"continuation", TriggerResult{Class: TriggerContinuation, Signal: "topic_boundary"}, ReasonEvidenceContinuation},
		{"unknown", TriggerResult{Class: TriggerUnknown, Signal: "topic_boundary"}, ReasonEvidenceUnknown},
		{
			"low_confidence",
			TriggerResult{Class: TriggerChange, Confidence: 0.5, Signal: "topic_boundary"},
			ReasonEvidenceLowConfidence,
		},
		{
			"fallback",
			TriggerResult{Class: TriggerChange, Confidence: 1, Signal: "topic_boundary", Fallback: true},
			ReasonEvidenceFallback,
		},
		{
			"wrong_signal",
			TriggerResult{Class: TriggerChange, Confidence: 1, Signal: "other"},
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
	policy.SignalVersions = []string{"v2"}
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
