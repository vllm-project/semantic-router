package contextdedup

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func textMessage(role llmprotocol.Role, text string) llmprotocol.Message {
	return llmprotocol.Message{Role: role, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}}}
}

// duplicatedRequest is a history sent twice ahead of the live turn.
func duplicatedRequest() *llmprotocol.Request {
	return &llmprotocol.Request{
		Generation: 1, Model: "model", Metadata: map[string]string{"keep": "unchanged"},
		Instructions: []llmprotocol.InstructionBlock{{Role: llmprotocol.RoleDeveloper, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "follow permissions"}}}},
		Messages: []llmprotocol.Message{
			textMessage(llmprotocol.RoleUser, "old question"), textMessage(llmprotocol.RoleAssistant, "old answer"),
			textMessage(llmprotocol.RoleUser, "old question"), textMessage(llmprotocol.RoleAssistant, "old answer"),
			textMessage(llmprotocol.RoleUser, "live question"),
		},
	}
}

func encoded(t *testing.T, value interface{}) string {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func apply(t *testing.T, request *llmprotocol.Request, action *Action) (*contextcompression.RequestIR, error) {
	t.Helper()
	ir := contextcompression.ParseSemanticRequest(request, contextcompression.Provenance{})
	action.WithResolver(RequestResolver(ir))
	return ir, ir.ApplySteps(context.Background(), []contextcompression.TransformationStep{action.Step()})
}

func TestActionRemovesDuplicateThroughSharedExecutor(t *testing.T) {
	request := duplicatedRequest()
	action := NewAction(testPolicy(), "")
	ir, err := apply(t, request, action)
	if err != nil {
		t.Fatal(err)
	}
	if len(request.Messages) != 3 || request.Messages[2].Content[0].Text != "live question" || request.Generation != 2 {
		t.Fatalf("wrong survivors: %+v", request.Messages)
	}
	if request.Messages[0].Content[0].Text != "old question" || request.Messages[1].Content[0].Text != "old answer" {
		t.Fatal("the earlier copy must be the retained one")
	}
	if request.Instructions[0].Content[0].Text != "follow permissions" || request.Metadata["keep"] != "unchanged" {
		t.Fatal("non-history content changed")
	}
	result := action.Reconcile(ir.Transformations.Receipts())
	if result.Outcome != OutcomeApplied || result.RemovedMessages != 2 || result.RemovedTurns != 1 || result.RetainedMessages != 3 {
		t.Fatalf("unexpected diagnostics %+v", result)
	}
	receipts := ir.Transformations.Receipts()
	if len(receipts) != 1 || receipts[0].Kind != contextcompression.TransformDeduplicate || receipts[0].MessagesRemoved != 2 {
		t.Fatalf("unexpected receipts %+v", receipts)
	}
	for _, leak := range []string{"old question", "old answer", "live question", "follow permissions"} {
		if strings.Contains(encoded(t, receipts), leak) || strings.Contains(encoded(t, result), leak) {
			t.Fatalf("receipt leaked content %q", leak)
		}
	}
	before := encoded(t, request)
	if err = ir.ApplySteps(context.Background(), []contextcompression.TransformationStep{action.Step()}); err != nil {
		t.Fatal(err)
	}
	if encoded(t, request) != before || len(ir.Transformations.Receipts()) != 1 {
		t.Fatal("repeated plan changed state")
	}
	if again := action.Reconcile(ir.Transformations.Receipts()); again.RemovedMessages != 2 {
		t.Fatalf("repeated reconcile changed the result: %+v", again)
	}
}

func TestActionFollowsResetRemovalsBeforeDeduplicating(t *testing.T) {
	request := &llmprotocol.Request{Model: "model", Messages: []llmprotocol.Message{
		textMessage(llmprotocol.RoleUser, "stale"), textMessage(llmprotocol.RoleAssistant, "stale answer"),
		textMessage(llmprotocol.RoleUser, "old question"), textMessage(llmprotocol.RoleAssistant, "old answer"),
		textMessage(llmprotocol.RoleUser, "old question"), textMessage(llmprotocol.RoleAssistant, "old answer"),
		textMessage(llmprotocol.RoleUser, "live question"),
	}}
	ir := contextcompression.ParseSemanticRequest(request, contextcompression.Provenance{})
	action := NewAction(testPolicy(), "").WithResolver(RequestResolver(ir))
	reset := contextcompression.TransformationStep{
		Kind: contextcompression.TransformReset, Enabled: true, FailureMode: contextcompression.FailureClosed,
		Propose: func(context.Context, contextcompression.TransformationView) (contextcompression.TransformationEdits, error) {
			return contextcompression.TransformationEdits{RemoveMessages: []int{0, 1}}, nil
		},
	}
	if err := ir.ApplySteps(context.Background(), []contextcompression.TransformationStep{reset, action.Step()}); err != nil {
		t.Fatal(err)
	}
	if len(request.Messages) != 3 || request.Messages[0].Content[0].Text != "old question" || request.Messages[2].Content[0].Text != "live question" {
		t.Fatalf("wrong survivors after reset then dedup: %+v", request.Messages)
	}
	result := action.Reconcile(ir.Transformations.Receipts())
	if result.Outcome != OutcomeApplied || result.RemovedMessages != 2 || result.ExaminedMessages != 5 {
		t.Fatalf("unexpected diagnostics %+v", result)
	}
}

func TestActionFailureModes(t *testing.T) {
	for _, tc := range []struct {
		name       string
		policy     Policy
		blocked    string
		wantErr    bool
		wantReason string
	}{
		{"blocked_fail_open", testPolicy(), ReasonUnsupportedRepresentation, false, ReasonUnsupportedRepresentation},
		{"blocked_fail_closed", Policy{FailClosed: true, MaxSegmentTurns: 64}, ReasonUnsupportedRepresentation, true, ReasonUnsupportedRepresentation},
		{"limit_fail_open", Policy{MaxHistoryTurns: 1, MaxSegmentTurns: 64}, "", false, ReasonHistoryLimitExceeded},
		{"limit_fail_closed", Policy{MaxHistoryTurns: 1, MaxSegmentTurns: 64, FailClosed: true}, "", true, ReasonHistoryLimitExceeded},
		{"timeout_fail_open", Policy{Timeout: time.Nanosecond, MaxSegmentTurns: 64}, "", false, ReasonCancelled},
	} {
		t.Run(tc.name, func(t *testing.T) {
			request := duplicatedRequest()
			before := encoded(t, request)
			action := NewAction(tc.policy, tc.blocked)
			if tc.name == "timeout_fail_open" {
				time.Sleep(time.Millisecond)
			}
			ir, err := apply(t, request, action)
			if (err != nil) != tc.wantErr {
				t.Fatalf("error mismatch: %v", err)
			}
			if encoded(t, request) != before {
				t.Fatal("a failed evaluation must leave the request unchanged")
			}
			result := action.Reconcile(ir.Transformations.Receipts())
			if result.Outcome != OutcomeFailed || result.Reason != tc.wantReason || result.RemovedMessages != 0 {
				t.Fatalf("unexpected diagnostics %+v", result)
			}
			receipt := ir.Transformations.Receipts()[0]
			if receipt.Status != contextcompression.TransformationFailed || receipt.Reason != "policy_failed" {
				t.Fatalf("unexpected receipt %+v", receipt)
			}
		})
	}
}

func TestActionCancelledBeforeProposalIsNotEvaluated(t *testing.T) {
	// The shared executor refuses to run a policy on a cancelled context, so
	// the action is never evaluated and the request is untouched.
	request := duplicatedRequest()
	ir := contextcompression.ParseSemanticRequest(request, contextcompression.Provenance{})
	action := NewAction(Policy{Timeout: time.Hour, MaxSegmentTurns: 64}, "").WithResolver(RequestResolver(ir))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := ir.ApplySteps(ctx, []contextcompression.TransformationStep{action.Step()}); err != nil {
		t.Fatal(err)
	}
	if len(request.Messages) != 5 {
		t.Fatal("cancelled plan changed the request")
	}
	result := action.Reconcile(ir.Transformations.Receipts())
	if result.Outcome != OutcomeFailed || result.Reason != ReasonNotEvaluated || result.RemovedMessages != 0 {
		t.Fatalf("unexpected diagnostics %+v", result)
	}
}

func TestActionReconcileWithoutEvaluation(t *testing.T) {
	action := NewAction(testPolicy(), "")
	result := action.Reconcile(nil)
	if result.Outcome != OutcomeSkipped || result.Reason != ReasonNotEvaluated || result.Normalization != NormalizationExact {
		t.Fatalf("unexpected diagnostics %+v", result)
	}
	if result.RecoveryStatus != RecoveryNotRequired || result.Scope != ScopeEligibleHistory {
		t.Fatalf("unexpected diagnostics %+v", result)
	}
	disabled := contextcompression.TransformationReceipt{Kind: contextcompression.TransformDeduplicate, Status: contextcompression.TransformationSkipped, Reason: "disabled"}
	if result = action.Reconcile([]contextcompression.TransformationReceipt{disabled}); result.Reason != ReasonNotEvaluated || result.Outcome != OutcomeSkipped {
		t.Fatalf("unexpected diagnostics %+v", result)
	}
}

func TestActionReconcileTrustsExecutorOverProposal(t *testing.T) {
	request := duplicatedRequest()
	ir := contextcompression.ParseSemanticRequest(request, contextcompression.Provenance{})
	action := NewAction(testPolicy(), "").WithResolver(RequestResolver(ir))
	if _, err := action.propose(context.Background(), ir.TransformationView()); err != nil {
		t.Fatal(err)
	}
	if action.result.RemovedMessages != 2 {
		t.Fatalf("proposal should have found the duplicate: %+v", action.result)
	}
	rejected := contextcompression.TransformationReceipt{Kind: contextcompression.TransformDeduplicate, Status: contextcompression.TransformationFailed, Reason: "invariant_violation"}
	result := action.Reconcile([]contextcompression.TransformationReceipt{rejected})
	if result.Outcome != OutcomeFailed || result.Reason != "invariant_violation" || result.RemovedMessages != 0 || len(result.Segments) != 0 {
		t.Fatalf("a rejected proposal removed nothing: %+v", result)
	}
	if result = action.Reconcile(nil); result.Outcome != OutcomeSkipped || result.Reason != ReasonNotEvaluated || result.RemovedMessages != 0 {
		t.Fatalf("a missing receipt removed nothing: %+v", result)
	}
	if result = action.Reconcile([]contextcompression.TransformationReceipt{{Kind: contextcompression.TransformReset, Status: contextcompression.TransformationApplied}}); result.RemovedMessages != 0 {
		t.Fatalf("another step's receipt must not count: %+v", result)
	}
}

func TestActionCleanHistorySkips(t *testing.T) {
	request := &llmprotocol.Request{Model: "model", Messages: []llmprotocol.Message{
		textMessage(llmprotocol.RoleUser, "q1"), textMessage(llmprotocol.RoleAssistant, "a1"),
		textMessage(llmprotocol.RoleUser, "q2"), textMessage(llmprotocol.RoleAssistant, "a2"),
		textMessage(llmprotocol.RoleUser, "live"),
	}}
	before := encoded(t, request)
	action := NewAction(testPolicy(), "")
	ir, err := apply(t, request, action)
	if err != nil {
		t.Fatal(err)
	}
	if encoded(t, request) != before {
		t.Fatal("clean history changed")
	}
	result := action.Reconcile(ir.Transformations.Receipts())
	if result.Outcome != OutcomeSkipped || result.Reason != ReasonNoDuplicates {
		t.Fatalf("unexpected diagnostics %+v", result)
	}
	if receipt := ir.Transformations.Receipts()[0]; receipt.Status != contextcompression.TransformationSkipped || receipt.Reason != "no_changes" {
		t.Fatalf("unexpected receipt %+v", receipt)
	}
}

func TestActionRawRequestCannotBeProven(t *testing.T) {
	body := map[string]interface{}{"model": "model", "messages": []interface{}{
		map[string]interface{}{"role": "user", "content": "old"},
		map[string]interface{}{"role": "assistant", "content": "answer"},
		map[string]interface{}{"role": "user", "content": "old"},
		map[string]interface{}{"role": "assistant", "content": "answer"},
		map[string]interface{}{"role": "user", "content": "live"},
	}}
	ir := contextcompression.ParseRequestIR(body, contextcompression.Provenance{})
	before := encoded(t, body)
	action := NewAction(testPolicy(), "").WithResolver(RequestResolver(ir))
	if err := ir.ApplySteps(context.Background(), []contextcompression.TransformationStep{action.Step()}); err != nil {
		t.Fatal(err)
	}
	if encoded(t, body) != before {
		t.Fatal("raw request changed without proof")
	}
	if result := action.Reconcile(ir.Transformations.Receipts()); result.Reason != ReasonEquivalenceUnverifiable || result.Outcome != OutcomeFailed {
		t.Fatalf("unexpected diagnostics %+v", result)
	}
}
