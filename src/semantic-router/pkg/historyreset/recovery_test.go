package historyreset

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

type recoveryWriterStub struct {
	payloads []string
	err      error
}

func (w *recoveryWriterStub) Store(_ context.Context, payload string) (string, error) {
	if w.err != nil {
		return "", w.err
	}
	w.payloads = append(w.payloads, payload)
	return fmt.Sprintf("key-%d", len(w.payloads)), nil
}

func recoverableAction(t *testing.T, writer RecoveryWriter, policy Policy) (*Action, *llmprotocol.Request) {
	t.Helper()
	request := conversation()
	detached := make(map[int]llmprotocol.Message, len(request.Messages))
	for index, message := range request.Messages {
		detached[index] = message
	}
	return NewAction(policy, acceptedChange(), "").WithRecovery(writer, detached), request
}

func TestRecoverableRemovalStoresTheRemovedTurnsBeforeCommitting(t *testing.T) {
	writer := &recoveryWriterStub{}
	action, request := recoverableAction(t, writer, testPolicy())

	ir, err := applyAction(t, request, action)
	if err != nil {
		t.Fatalf("expected the plan to succeed: %v", err)
	}
	if len(request.Messages) != 1 {
		t.Fatalf("expected the prior turns to be removed, got %d messages", len(request.Messages))
	}
	if len(writer.payloads) != 1 {
		t.Fatalf("expected exactly one stored payload, got %d", len(writer.payloads))
	}
	if action.RecoveryKey() != "key-1" {
		t.Fatalf("unexpected issued key %q", action.RecoveryKey())
	}

	envelope, err := DecodeEnvelope(writer.payloads[0])
	if err != nil {
		t.Fatalf("stored payload is not a valid envelope: %v", err)
	}
	if envelope.Messages != 4 || envelope.Turns != 2 || len(envelope.Removed) != 4 {
		t.Fatalf("unexpected envelope counts %+v", envelope)
	}
	if envelope.Removed[0].ID != 0 || envelope.Removed[0].Message.Content[0].Text != "old question" {
		t.Fatalf("envelope lost the removed content: %+v", envelope.Removed[0])
	}
	if envelope.Removed[0].TurnID == envelope.Removed[2].TurnID {
		t.Fatal("envelope must preserve distinct turn identities")
	}

	diagnostics := action.Reconcile(ir.Transformations.Receipts())
	if diagnostics.RecoveryStatus != RecoveryStored || diagnostics.RecoveryEntries != 1 {
		t.Fatalf("unexpected recovery diagnostics %+v", diagnostics)
	}
}

func TestRecoveryEnvelopePreservesToolExchanges(t *testing.T) {
	writer := &recoveryWriterStub{}
	action, request := recoverableAction(t, writer, testPolicy())
	request.Messages = []llmprotocol.Message{
		text(llmprotocol.RoleUser, "old question"),
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
			Kind:     llmprotocol.ContentToolCall,
			ToolCall: &llmprotocol.ToolCall{ID: "a", Name: "lookup", Arguments: `{"q":1}`},
		}}},
		{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{
			Kind: llmprotocol.ContentToolResult,
			ToolResult: &llmprotocol.ToolResult{
				CallID:  "a",
				Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "a result"}},
			},
		}}},
		text(llmprotocol.RoleUser, "live question"),
	}
	detached := make(map[int]llmprotocol.Message, len(request.Messages))
	for index, message := range request.Messages {
		detached[index] = message
	}
	action.WithRecovery(writer, detached)

	if _, err := applyAction(t, request, action); err != nil {
		t.Fatalf("expected the plan to succeed: %v", err)
	}
	envelope, err := DecodeEnvelope(writer.payloads[0])
	if err != nil {
		t.Fatalf("stored payload is not a valid envelope: %v", err)
	}
	call := envelope.Removed[1].Message.Content[0].ToolCall
	result := envelope.Removed[2].Message.Content[0].ToolResult
	if call == nil || call.ID != "a" || call.Arguments != `{"q":1}` {
		t.Fatalf("envelope lost the tool call: %+v", envelope.Removed[1])
	}
	if result == nil || result.CallID != "a" || result.Content[0].Text != "a result" {
		t.Fatalf("envelope lost the tool result: %+v", envelope.Removed[2])
	}
}

// A failed write must leave the conversation intact: the shared executor
// cannot restore messages once it has removed them.
func TestRecoveryWriteFailurePreservesHistory(t *testing.T) {
	writer := &recoveryWriterStub{err: fmt.Errorf("store outage")}
	action, request := recoverableAction(t, writer, testPolicy())

	ir, err := applyAction(t, request, action)
	if err != nil {
		t.Fatalf("fail-open must not stop the plan: %v", err)
	}
	if len(request.Messages) != 5 || request.Generation != 1 {
		t.Fatalf("history must survive a failed recovery write, got %d messages", len(request.Messages))
	}
	diagnostics := action.Reconcile(ir.Transformations.Receipts())
	if diagnostics.Outcome != OutcomeFailed || diagnostics.Reason != ReasonRecoveryWriteFailed {
		t.Fatalf("unexpected diagnostics %+v", diagnostics)
	}
	if diagnostics.RecoveryStatus != RecoveryFailed || diagnostics.RemovedMessages != 0 {
		t.Fatalf("a failed write cannot report stored removals: %+v", diagnostics)
	}
}

func TestRecoveryWriteFailureRejectsUnderFailClosed(t *testing.T) {
	policy := testPolicy()
	policy.FailClosed = true
	writer := &recoveryWriterStub{err: fmt.Errorf("store outage")}
	action, request := recoverableAction(t, writer, policy)

	if _, err := applyAction(t, request, action); err == nil {
		t.Fatal("fail-closed must reject before dispatch")
	}
	if len(request.Messages) != 5 {
		t.Fatal("a rejected request must keep its pre-reset state")
	}
}

func TestRecoveryPayloadBudgetPreservesHistory(t *testing.T) {
	policy := testPolicy()
	policy.MaxRecoveryBytes = 16
	writer := &recoveryWriterStub{}
	action, request := recoverableAction(t, writer, policy)

	ir, err := applyAction(t, request, action)
	if err != nil {
		t.Fatalf("fail-open must not stop the plan: %v", err)
	}
	if len(request.Messages) != 5 {
		t.Fatal("an oversized payload must not remove history")
	}
	if len(writer.payloads) != 0 {
		t.Fatal("an oversized payload must not reach the store")
	}
	diagnostics := action.Reconcile(ir.Transformations.Receipts())
	if diagnostics.Reason != ReasonRecoveryLimitExceeded {
		t.Fatalf("unexpected diagnostics %+v", diagnostics)
	}
}

// The policy's view carries text only, so removal must stop when the trusted
// caller did not supply the complete content for a selected message.
func TestRecoveryRefusesRemovalWithoutDetachedContent(t *testing.T) {
	writer := &recoveryWriterStub{}
	request := conversation()
	action := NewAction(testPolicy(), acceptedChange(), "").
		WithRecovery(writer, map[int]llmprotocol.Message{0: request.Messages[0]})

	ir, err := applyAction(t, request, action)
	if err != nil {
		t.Fatalf("fail-open must not stop the plan: %v", err)
	}
	if len(request.Messages) != 5 {
		t.Fatal("incomplete recovery content must not remove history")
	}
	diagnostics := action.Reconcile(ir.Transformations.Receipts())
	if diagnostics.Reason != ReasonRecoveryWriteFailed {
		t.Fatalf("unexpected diagnostics %+v", diagnostics)
	}
}

func TestRecoveryIsNotRequestedWhenNothingIsRemoved(t *testing.T) {
	writer := &recoveryWriterStub{}
	action, request := recoverableAction(t, writer, testPolicy())
	request.Messages = request.Messages[4:]

	if _, err := applyAction(t, request, action); err != nil {
		t.Fatalf("expected a no-op plan to succeed: %v", err)
	}
	if len(writer.payloads) != 0 || action.RecoveryKey() != "" {
		t.Fatal("a no-op reset must not write recovery content")
	}
}

func TestDecodeEnvelopeRejectsAnUnknownVersion(t *testing.T) {
	if _, err := DecodeEnvelope(`{"version":"other","removed":[]}`); err == nil {
		t.Fatal("an unknown envelope version was accepted")
	}
	if _, err := DecodeEnvelope("not json"); err == nil {
		t.Fatal("a malformed envelope was accepted")
	}
}

// Nothing in the stored payload should be reachable through the diagnostics.
func TestRecoveryDiagnosticsOmitKeysAndContent(t *testing.T) {
	writer := &recoveryWriterStub{}
	action, request := recoverableAction(t, writer, testPolicy())

	ir, err := applyAction(t, request, action)
	if err != nil {
		t.Fatalf("expected the plan to succeed: %v", err)
	}
	diagnostics := action.Reconcile(ir.Transformations.Receipts())
	rendered := fmt.Sprintf("%+v", diagnostics)
	for _, forbidden := range []string{action.RecoveryKey(), "old question", "old answer"} {
		if forbidden != "" && strings.Contains(rendered, forbidden) {
			t.Fatalf("diagnostics leaked %q: %s", forbidden, rendered)
		}
	}
}

func TestStaleAndConflictingEvidenceCannotAuthorizeRemoval(t *testing.T) {
	policy := testPolicy()
	policy.Binding = "request-binding"
	cases := []struct {
		name    string
		trigger TriggerResult
		reason  string
	}{
		{
			"unbound",
			TriggerResult{Class: TriggerChange, Confidence: 1, Signal: "topic_boundary"},
			ReasonEvidenceStale,
		},
		{
			"other_request",
			TriggerResult{Class: TriggerChange, Confidence: 1, Signal: "topic_boundary", Binding: "other"},
			ReasonEvidenceStale,
		},
		{
			"conflicting",
			TriggerResult{
				Class: TriggerConflicting, Confidence: 1,
				Signal: "topic_boundary", Binding: "request-binding",
			},
			ReasonEvidenceConflicting,
		},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			edits, diagnostics := Plan(
				context.Background(),
				policy,
				test.trigger,
				contextcompression.TransformationView{Messages: []contextcompression.MessageView{
					historyMessage(0, 0, "user"),
					historyMessage(1, 0, "assistant"),
				}},
			)
			if len(edits.RemoveMessages) != 0 {
				t.Fatalf("expected no removal, got %v", edits.RemoveMessages)
			}
			if diagnostics.Reason != test.reason {
				t.Fatalf("expected %q, got %+v", test.reason, diagnostics)
			}
		})
	}
}

func TestMatchingBindingAuthorizesRemoval(t *testing.T) {
	policy := testPolicy()
	policy.Binding = "request-binding"
	trigger := acceptedChange()
	trigger.Binding = "request-binding"

	edits, diagnostics := Plan(
		context.Background(),
		policy,
		trigger,
		contextcompression.TransformationView{Messages: []contextcompression.MessageView{
			historyMessage(0, 0, "user"),
			historyMessage(1, 0, "assistant"),
			protect(historyMessage(2, 2, "user"), contextcompression.ProtectLiveTurn),
		}},
	)
	if len(edits.RemoveMessages) != 2 || diagnostics.Outcome != OutcomeApplied {
		t.Fatalf("bound evidence must authorize removal, got %v %+v", edits.RemoveMessages, diagnostics)
	}
}
