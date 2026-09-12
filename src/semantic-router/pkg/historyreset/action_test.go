package historyreset

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// These tests run the action through the real shared executor so the proposal,
// its independent revalidation, the committed removal, and the receipt are
// exercised together rather than mocked.

func conversation() *llmprotocol.Request {
	return &llmprotocol.Request{
		Generation: 1,
		Model:      "model",
		Metadata:   map[string]string{"keep": "unchanged"},
		Instructions: []llmprotocol.InstructionBlock{{
			Role:    llmprotocol.RoleDeveloper,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "follow permissions"}},
		}},
		Messages: []llmprotocol.Message{
			text(llmprotocol.RoleUser, "old question"),
			text(llmprotocol.RoleAssistant, "old answer"),
			text(llmprotocol.RoleUser, "another question"),
			text(llmprotocol.RoleAssistant, "another answer"),
			text(llmprotocol.RoleUser, "live question"),
		},
	}
}

func text(role llmprotocol.Role, body string) llmprotocol.Message {
	return llmprotocol.Message{
		Role:    role,
		Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: body}},
	}
}

func applyAction(
	t *testing.T,
	request *llmprotocol.Request,
	action *Action,
) (*contextcompression.RequestIR, error) {
	t.Helper()
	ir := contextcompression.ParseSemanticRequest(request, contextcompression.Provenance{})
	err := ir.ApplySteps(context.Background(), []contextcompression.TransformationStep{action.Step()})
	return ir, err
}

func TestActionRemovesPriorTurnsThroughTheSharedExecutor(t *testing.T) {
	request := conversation()
	action := NewAction(testPolicy(), acceptedChange(), "")

	ir, err := applyAction(t, request, action)
	if err != nil {
		t.Fatalf("expected the plan to succeed: %v", err)
	}
	if len(request.Messages) != 1 || request.Messages[0].Content[0].Text != "live question" {
		t.Fatalf("expected only the live turn to survive, got %+v", request.Messages)
	}
	if len(request.Instructions) != 1 {
		t.Fatal("developer instructions must survive a reset")
	}
	if request.Metadata["keep"] != "unchanged" {
		t.Fatal("reset must not disturb request metadata")
	}
	if request.Generation != 2 {
		t.Fatalf("expected the executor to advance the generation, got %d", request.Generation)
	}

	diagnostics := action.Reconcile(ir.Transformations.Receipts())
	if diagnostics.Outcome != OutcomeApplied || diagnostics.Reason != ReasonApplied {
		t.Fatalf("unexpected diagnostics %+v", diagnostics)
	}
	if diagnostics.RemovedMessages != 4 || diagnostics.RetainedMessages != 1 {
		t.Fatalf("committed counts must match the receipt, got %+v", diagnostics)
	}
	if diagnostics.Signal != "topic_boundary" || diagnostics.TriggerClass != TriggerChange {
		t.Fatalf("diagnostics must carry the trigger identity, got %+v", diagnostics)
	}
}

func TestActionPreservesHistoryWithoutAuthorizingEvidence(t *testing.T) {
	request := conversation()
	action := NewAction(testPolicy(), TriggerResult{
		Class:      TriggerContinuation,
		Confidence: 1,
		Signal:     "topic_boundary",
	}, "")

	ir, err := applyAction(t, request, action)
	if err != nil {
		t.Fatalf("a continuation is a normal no-op: %v", err)
	}
	if len(request.Messages) != 5 || request.Generation != 1 {
		t.Fatalf("continuation must leave the request untouched, got %d messages", len(request.Messages))
	}

	diagnostics := action.Reconcile(ir.Transformations.Receipts())
	if diagnostics.Outcome != OutcomeSkipped || diagnostics.Reason != ReasonEvidenceContinuation {
		t.Fatalf("unexpected diagnostics %+v", diagnostics)
	}
	if diagnostics.RemovedMessages != 0 {
		t.Fatalf("a skipped action cannot report removals, got %+v", diagnostics)
	}
}

func TestActionKeepsToolExchangesIntact(t *testing.T) {
	request := conversation()
	request.Messages = []llmprotocol.Message{
		text(llmprotocol.RoleUser, "old question"),
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
			Kind:     llmprotocol.ContentToolCall,
			ToolCall: &llmprotocol.ToolCall{ID: "a", Name: "lookup", Arguments: "{}"},
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
	action := NewAction(testPolicy(), acceptedChange(), "")

	if _, err := applyAction(t, request, action); err != nil {
		t.Fatalf("expected the plan to succeed: %v", err)
	}
	if len(request.Messages) != 1 || request.Messages[0].Content[0].Text != "live question" {
		t.Fatalf("expected the complete exchange to be removed together, got %+v", request.Messages)
	}
}

func TestActionFailsOpenWhenBlockedBeforePlanning(t *testing.T) {
	request := conversation()
	action := NewAction(testPolicy(), acceptedChange(), ReasonRecoveryUnavailable)

	ir, err := applyAction(t, request, action)
	if err != nil {
		t.Fatalf("fail-open must not stop the plan: %v", err)
	}
	if len(request.Messages) != 5 {
		t.Fatal("a blocked action must preserve the pre-reset request")
	}

	diagnostics := action.Reconcile(ir.Transformations.Receipts())
	if diagnostics.Outcome != OutcomeFailed || diagnostics.Reason != ReasonRecoveryUnavailable {
		t.Fatalf("unexpected diagnostics %+v", diagnostics)
	}
}

func TestActionFailsClosedWhenBlockedBeforePlanning(t *testing.T) {
	request := conversation()
	policy := testPolicy()
	policy.FailClosed = true
	action := NewAction(policy, acceptedChange(), ReasonUnsupportedRepresentation)

	ir, err := applyAction(t, request, action)
	if err == nil {
		t.Fatal("fail-closed must stop the plan before dispatch")
	}
	if len(request.Messages) != 5 || request.Generation != 1 {
		t.Fatal("a rejected request must keep its pre-reset state")
	}

	diagnostics := action.Reconcile(ir.Transformations.Receipts())
	if diagnostics.Outcome != OutcomeFailed {
		t.Fatalf("expected a failed outcome, got %+v", diagnostics)
	}
	if diagnostics.RemovedMessages != 0 {
		t.Fatalf("a failed action cannot report removals, got %+v", diagnostics)
	}
}

func TestActionReportsNotEvaluatedWhenTheStepNeverRuns(t *testing.T) {
	action := NewAction(testPolicy(), acceptedChange(), "")
	diagnostics := action.Reconcile(nil)
	if diagnostics.Outcome != OutcomeSkipped || diagnostics.Reason != ReasonNotEvaluated {
		t.Fatalf("unexpected diagnostics %+v", diagnostics)
	}
}

// Re-entering an already completed kind within one request returns the original
// receipt, so the action must not publish a second applied event.
func TestActionKeepsTheFirstFinalizedResultOnReentry(t *testing.T) {
	request := conversation()
	action := NewAction(testPolicy(), acceptedChange(), "")

	ir, err := applyAction(t, request, action)
	if err != nil {
		t.Fatalf("expected the plan to succeed: %v", err)
	}
	first := action.Reconcile(ir.Transformations.Receipts())

	if err = ir.ApplySteps(
		context.Background(),
		[]contextcompression.TransformationStep{action.Step()},
	); err != nil {
		t.Fatalf("re-entry must be accepted: %v", err)
	}
	if len(request.Messages) != 1 {
		t.Fatal("re-entry must not remove anything further")
	}
	if second := action.Reconcile(ir.Transformations.Receipts()); second != first {
		t.Fatalf("expected the finalized result to be stable, got %+v then %+v", first, second)
	}
}

// An enabled history step opts the request into shared protection of live-turn
// history blocks even when the reset itself removes nothing.
func TestEnabledActionProtectsLiveHistoryEvenWhenItRemovesNothing(t *testing.T) {
	request := conversation()
	request.Messages = request.Messages[4:]
	action := NewAction(testPolicy(), acceptedChange(), "")

	ir, err := applyAction(t, request, action)
	if err != nil {
		t.Fatalf("expected a no-op plan to succeed: %v", err)
	}
	diagnostics := action.Reconcile(ir.Transformations.Receipts())
	if diagnostics.Reason != ReasonNoEligibleHistory {
		t.Fatalf("expected no_eligible_history, got %+v", diagnostics)
	}
	if len(request.Messages) != 1 {
		t.Fatal("the live turn must survive")
	}
}
