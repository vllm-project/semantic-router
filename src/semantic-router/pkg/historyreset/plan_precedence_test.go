package historyreset

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
)

func view(messages ...contextcompression.MessageView) contextcompression.TransformationView {
	return contextcompression.TransformationView{Messages: messages}
}

// removableHistory has one complete removable prior turn and the live turn.
func removableHistory() contextcompression.TransformationView {
	return view(
		historyMessage(0, 0, "user"),
		historyMessage(1, 0, "assistant"),
		protect(historyMessage(2, 2, "user"), contextcompression.ProtectLiveTurn),
	)
}

// overLimitPolicy admits a single history turn, so removableHistory, with a
// prior turn plus the live turn, exceeds it.
func overLimitPolicy() Policy {
	policy := testPolicy()
	policy.MaxHistoryTurns = 1
	return policy
}

func failClosed(policy Policy) Policy {
	policy.FailClosed = true
	return policy
}

func assertUnchangedCounts(t *testing.T, diagnostics Diagnostics, examined, protected int) {
	t.Helper()
	if diagnostics.ExaminedMessages != examined || diagnostics.RetainedMessages != examined ||
		diagnostics.ProtectedMessages != protected ||
		diagnostics.RemovedMessages != 0 || diagnostics.RemovedTurns != 0 {
		t.Fatalf("counts must describe an unchanged request of %d messages (%d protected): %+v",
			examined, protected, diagnostics)
	}
}

// Individually eligible messages are not a removable set: a turn that also
// holds protected content, or that a retained tool exchange pins, is kept
// whole. With nothing actually removable, missing or unknown evidence must be
// an ordinary no-op even under fail_closed.
func TestEvidenceIsJudgedOnlyAgainstARemovableSet(t *testing.T) {
	histories := map[string]struct {
		view      contextcompression.TransformationView
		protected int
	}{
		"turn shares a protected message": {view(
			historyMessage(0, 0, "user"),
			protect(historyMessage(1, 0, "assistant"), contextcompression.ProtectMultimodal),
			protect(historyMessage(2, 2, "user"), contextcompression.ProtectLiveTurn),
		), 2},
		"turn pinned by a retained tool exchange": {view(
			withExchange(historyMessage(0, 0, "assistant"), "call-1"),
			withExchange(historyMessage(1, 0, "tool"), "call-1"),
			protect(withExchange(historyMessage(2, 2, "tool"), "call-1"), contextcompression.ProtectLiveTurn),
		), 1},
	}
	triggers := map[string]TriggerResult{
		"missing": {},
		"unknown": {Class: TriggerUnknown, Signal: "topic_boundary", Version: "v1"},
	}
	for historyName, history := range histories {
		for triggerName, trigger := range triggers {
			t.Run(historyName+"/"+triggerName, func(t *testing.T) {
				action := NewAction(failClosed(testPolicy()), trigger, "")
				edits, err := action.propose(context.Background(), history.view)
				if err != nil || len(edits.RemoveMessages) != 0 {
					t.Fatalf("expected an ordinary no-op, got edits=%v err=%v", edits.RemoveMessages, err)
				}
				diagnostics := action.Reconcile(nil)
				if diagnostics.Outcome != OutcomeSkipped || diagnostics.Reason != ReasonNoEligibleHistory {
					t.Fatalf("unexpected diagnostics %+v", diagnostics)
				}
				assertUnchangedCounts(t, diagnostics, 3, history.protected)
			})
		}
	}
}

// A validated continuation can never authorize removal, so it stays a no-op
// however long the history is, rather than failing on the history limit.
func TestValidatedContinuationIsANoOpOverTheHistoryLimit(t *testing.T) {
	continuation := TriggerResult{Class: TriggerContinuation, Signal: "topic_boundary", Version: "v1"}
	action := NewAction(failClosed(overLimitPolicy()), continuation, "")
	if _, err := action.propose(context.Background(), removableHistory()); err != nil {
		t.Fatalf("a continuation was rejected over the history limit: %v", err)
	}
	diagnostics := action.Reconcile(nil)
	if diagnostics.Reason != ReasonEvidenceContinuation || diagnostics.Outcome != OutcomeSkipped {
		t.Fatalf("unexpected diagnostics %+v", diagnostics)
	}
	assertUnchangedCounts(t, diagnostics, 3, 1)
}

// Only a fully validated continuation takes the early no-op. A result merely
// labelled continuation keeps its own evidence reason within the limits, and
// is a history-limit failure over them; both reject under fail_closed.
func TestContinuationLabelledEvidenceMustValidateBeforeTheShortcut(t *testing.T) {
	continuation := func(edit func(*TriggerResult)) TriggerResult {
		result := TriggerResult{Class: TriggerContinuation, Signal: "topic_boundary", Version: "v1", Binding: "this"}
		edit(&result)
		return result
	}
	cases := []struct {
		name    string
		trigger TriggerResult
		reason  string
	}{
		{"wrong signal", continuation(func(r *TriggerResult) { r.Signal = "other" }), ReasonEvidenceWrongSignal},
		{"missing version", continuation(func(r *TriggerResult) { r.Version = "" }), ReasonEvidenceUnsupportedVersion},
		{"unsupported version", continuation(func(r *TriggerResult) { r.Version = "v9" }), ReasonEvidenceUnsupportedVersion},
		{"stale binding", continuation(func(r *TriggerResult) { r.Binding = "another request" }), ReasonEvidenceStale},
		{"fallback", continuation(func(r *TriggerResult) { r.Fallback = true }), ReasonEvidenceFallback},
	}
	for _, test := range cases {
		for _, overLimit := range []bool{false, true} {
			name, policy, want := test.name+"/within limits", testPolicy(), test.reason
			if overLimit {
				name, policy, want = test.name+"/over limits", overLimitPolicy(), ReasonHistoryLimitExceeded
			}
			t.Run(name, func(t *testing.T) {
				policy.Binding = "this"
				action := NewAction(failClosed(policy), test.trigger, "")
				if _, err := action.propose(context.Background(), removableHistory()); err == nil {
					t.Fatal("unvalidated evidence was accepted as a no-op under fail_closed")
				}
				diagnostics := action.Reconcile(nil)
				if diagnostics.Reason != want || diagnostics.Outcome != OutcomeFailed {
					t.Fatalf("want %s, got %+v", want, diagnostics)
				}
				assertUnchangedCounts(t, diagnostics, 3, 1)
			})
		}
	}
}

// The history limit is enforced before tool-exchange closure, the expensive
// part of planning. Without any individual candidate the request is a no-op
// whatever its size; with one, an over-limit history fails on the limit even
// for an accepted change.
func TestHistoryLimitPrecedence(t *testing.T) {
	nothingEligible := view(
		protect(historyMessage(0, 0, "user"), contextcompression.ProtectMultimodal),
		protect(historyMessage(1, 0, "assistant"), contextcompression.ProtectMultimodal),
		protect(historyMessage(2, 2, "user"), contextcompression.ProtectLiveTurn),
	)
	_, diagnostics := plan(context.Background(), overLimitPolicy(), acceptedChange(), nothingEligible)
	if diagnostics.Reason != ReasonNoEligibleHistory {
		t.Fatalf("no candidate over the limit: unexpected diagnostics %+v", diagnostics)
	}
	_, diagnostics = plan(context.Background(), overLimitPolicy(), acceptedChange(), removableHistory())
	if diagnostics.Reason != ReasonHistoryLimitExceeded {
		t.Fatalf("a candidate over the limit: unexpected diagnostics %+v", diagnostics)
	}
	assertUnchangedCounts(t, diagnostics, 3, 1)
}

// Every non-applied evaluation describes its input: all examined messages are
// retained and none removed.
func TestNonAppliedOutcomesRetainEveryExaminedMessage(t *testing.T) {
	cases := map[string]struct {
		policy  Policy
		trigger TriggerResult
		reason  string
	}{
		"unknown":        {testPolicy(), TriggerResult{Class: TriggerUnknown, Signal: "topic_boundary", Version: "v1"}, ReasonEvidenceUnknown},
		"continuation":   {testPolicy(), TriggerResult{Class: TriggerContinuation, Signal: "topic_boundary", Version: "v1"}, ReasonEvidenceContinuation},
		"low confidence": {testPolicy(), TriggerResult{Class: TriggerChange, Confidence: 0.5, Signal: "topic_boundary", Version: "v1"}, ReasonEvidenceLowConfidence},
		"over the limit": {overLimitPolicy(), acceptedChange(), ReasonHistoryLimitExceeded},
	}
	for name, test := range cases {
		t.Run(name, func(t *testing.T) {
			_, diagnostics := plan(context.Background(), test.policy, test.trigger, removableHistory())
			if diagnostics.Reason != test.reason {
				t.Fatalf("unexpected diagnostics %+v", diagnostics)
			}
			assertUnchangedCounts(t, diagnostics, 3, 1)
		})
	}
	blocked := NewAction(testPolicy(), acceptedChange(), ReasonRecoveryUnavailable)
	if _, err := blocked.propose(context.Background(), removableHistory()); err == nil {
		t.Fatal("a blocked action must fail its step")
	}
	assertUnchangedCounts(t, blocked.Reconcile(nil), 3, 1)
}

// A proposal that was never committed must not keep its removal counts: a
// failed or no-change receipt, or no receipt at all, leaves every examined
// message retained.
func TestUncommittedProposalsRestoreRetainedCounts(t *testing.T) {
	proposed := Diagnostics{
		Outcome: OutcomeApplied, Reason: ReasonApplied,
		ExaminedMessages: 5, RetainedMessages: 1, RemovedMessages: 4, RemovedTurns: 2,
	}
	receipts := map[string][]contextcompression.TransformationReceipt{
		"failed receipt":  {{Kind: contextcompression.TransformReset, Status: contextcompression.TransformationFailed, Reason: "invariant_violation"}},
		"skipped receipt": {{Kind: contextcompression.TransformReset, Status: contextcompression.TransformationSkipped, Reason: "no_changes"}},
		"no receipt":      nil,
	}
	for name, receipt := range receipts {
		t.Run(name, func(t *testing.T) {
			action := &Action{result: proposed, evaluated: true}
			diagnostics := action.Reconcile(receipt)
			if diagnostics.RemovedMessages != 0 || diagnostics.RemovedTurns != 0 ||
				diagnostics.RetainedMessages != diagnostics.ExaminedMessages {
				t.Fatalf("an uncommitted proposal kept its counts: %+v", diagnostics)
			}
		})
	}
}

// A failed recovery write stops the removal, so the request is reported as
// unchanged rather than with the counts of the abandoned proposal.
func TestRecoveryWriteFailureReportsAnUnchangedRequest(t *testing.T) {
	writer := &recoveryWriterStub{err: errors.New("store unavailable")}
	action, request := recoverableAction(t, writer, testPolicy())
	ir, _ := applyAction(t, request, action)
	diagnostics := action.Reconcile(ir.Transformations.Receipts())
	if diagnostics.Reason != ReasonRecoveryWriteFailed {
		t.Fatalf("unexpected diagnostics %+v", diagnostics)
	}
	if diagnostics.RemovedMessages != 0 || diagnostics.RemovedTurns != 0 ||
		diagnostics.RetainedMessages != diagnostics.ExaminedMessages || diagnostics.ExaminedMessages == 0 {
		t.Fatalf("a failed write must report an unchanged request: %+v", diagnostics)
	}
}

// countdownContext reports cancellation after a fixed number of checks, so a
// test can stop an evaluation part-way through a scan.
type countdownContext struct {
	context.Context
	remaining int
}

func (c *countdownContext) Err() error {
	if c.remaining <= 0 {
		return context.Canceled
	}
	c.remaining--
	return nil
}

// A long chain of linked tool exchanges needs one closure pass per link.
// Cancellation is observed between passes, and a cancelled evaluation reports
// unavailable (zero) counts rather than recomputing them after the deadline.
func TestLongExchangeChainStaysCancellationResponsive(t *testing.T) {
	const links = 200
	messages := make([]contextcompression.MessageView, 0, links+1)
	for turn := 0; turn < links; turn++ {
		messages = append(messages, withExchange(historyMessage(turn, turn, "assistant"),
			exchangeID(turn), exchangeID(turn+1)))
	}
	messages = append(messages, protect(withExchange(historyMessage(links, links, "tool"),
		exchangeID(links)), contextcompression.ProtectLiveTurn))

	// The survey consumes a few checks; the closure then needs ~links passes.
	ctx := &countdownContext{Context: context.Background(), remaining: 20}
	_, diagnostics := plan(ctx, testPolicy(), acceptedChange(), view(messages...))
	if diagnostics.Reason != ReasonCancelled {
		t.Fatalf("closure ignored cancellation: %+v", diagnostics)
	}
	if diagnostics.ExaminedMessages != 0 || diagnostics.RetainedMessages != 0 {
		t.Fatalf("a cancelled evaluation must report unavailable counts: %+v", diagnostics)
	}

	// Without cancellation the whole chain is pinned by the live turn.
	ids, diagnostics := planIDs(t, testPolicy(), acceptedChange(), messages...)
	if len(ids) != 0 || diagnostics.Reason != ReasonNoEligibleHistory {
		t.Fatalf("the linked chain must stay pinned: %v %+v", ids, diagnostics)
	}
}

func exchangeID(index int) string {
	return fmt.Sprintf("call-%d", index)
}
