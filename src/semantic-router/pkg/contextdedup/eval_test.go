package contextdedup

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// evalWorkload is one representative conversation shape with the exact
// sequence the provider must receive after deduplication. Token reduction is
// only meaningful together with that behavioral check.
type evalWorkload struct {
	name    string
	request *llmprotocol.Request
	want    []llmprotocol.Message
	// minSaved is the smallest acceptable share of estimated input tokens
	// removed. Zero marks a negative workload that must not change at all.
	minSaved float64
}

const evalQuestion = "Could you walk me through how the irrigation controller schedules zones when the forecast changes overnight?"

const evalAnswer = "The controller re-reads the forecast at midnight, shifts every zone whose soil sensor is above the threshold by one cycle, and logs the change."

func evalTurn(index int) []llmprotocol.Message {
	return []llmprotocol.Message{
		textMessage(llmprotocol.RoleUser, fmt.Sprintf("%s (%d)", evalQuestion, index)),
		textMessage(llmprotocol.RoleAssistant, fmt.Sprintf("%s (%d)", evalAnswer, index)),
	}
}

func evalTurns(from, to int) []llmprotocol.Message {
	var messages []llmprotocol.Message
	for index := from; index < to; index++ {
		messages = append(messages, evalTurn(index)...)
	}
	return messages
}

func evalRequest(parts ...[]llmprotocol.Message) *llmprotocol.Request {
	request := &llmprotocol.Request{Model: "model"}
	for _, part := range parts {
		request.Messages = append(request.Messages, part...)
	}
	request.Messages = append(request.Messages, textMessage(llmprotocol.RoleUser, "What should I change for the next season?"))
	return request
}

func evalExchange(index int, callID string) []llmprotocol.Message {
	return []llmprotocol.Message{
		textMessage(llmprotocol.RoleUser, fmt.Sprintf("Run the schedule check (%d)", index)),
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: callID, Name: "check_schedule", Arguments: `{"zone":"all"}`}}}},
		{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: callID, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "12 zones scheduled, 2 deferred by the forecast"}}}}}},
		textMessage(llmprotocol.RoleAssistant, "Twelve zones are scheduled and two are deferred until the forecast clears."),
	}
}

func withIDs(messages []llmprotocol.Message, prefix string) []llmprotocol.Message {
	result := append([]llmprotocol.Message(nil), messages...)
	for index := range result {
		result[index].ID = fmt.Sprintf("%s_%d", prefix, index)
	}
	return result
}

func refusalTurn(index int) []llmprotocol.Message {
	return []llmprotocol.Message{
		textMessage(llmprotocol.RoleUser, fmt.Sprintf("Override the safety interlock (%d)", index)),
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentRefusal, Text: "I can't help with disabling a safety interlock."}}},
	}
}

func evalWorkloads() []evalWorkload {
	clean := evalRequest(evalTurns(0, 8))
	singleDouble := evalRequest(evalTurn(0), evalTurn(0), evalTurns(1, 3))
	wholeDouble := evalRequest(evalTurns(0, 6), evalTurns(0, 6))
	wholeTriple := evalRequest(evalTurns(0, 6), evalTurns(0, 6), evalTurns(0, 6))
	responses := evalRequest(withIDs(evalTurns(0, 4), "msg"), evalTurns(0, 4))
	toolDistinct := evalRequest(evalExchange(0, "call_1"), evalExchange(0, "call_2"))
	toolSame := evalRequest(evalExchange(0, "call_1"), evalExchange(0, "call_1"))
	confirmation := evalRequest(
		[]llmprotocol.Message{textMessage(llmprotocol.RoleUser, "Delete the old schedule?"), textMessage(llmprotocol.RoleAssistant, "Are you sure?")},
		[]llmprotocol.Message{textMessage(llmprotocol.RoleUser, "yes"), textMessage(llmprotocol.RoleUser, "yes")},
	)
	correction := evalRequest(evalTurn(0), []llmprotocol.Message{evalTurn(0)[0], textMessage(llmprotocol.RoleAssistant, evalAnswer+" Actually, it shifts by two cycles.")})
	refusal := evalRequest(refusalTurn(0), refusalTurn(0))
	temporal := evalRequest(evalTurn(0), evalTurn(1), evalTurn(0))
	return []evalWorkload{
		{"clean_conversation", clean, clone(clean.Messages), 0},
		// One of four turns is a repeat, so a quarter of the history goes.
		{"single_turn_double_send", singleDouble, clone(evalRequest(evalTurn(0), evalTurns(1, 3)).Messages), 0.15},
		{"whole_history_double_send", wholeDouble, clone(evalRequest(evalTurns(0, 6)).Messages), 0.40},
		{"whole_history_triple_send", wholeTriple, clone(evalRequest(evalTurns(0, 6)).Messages), 0.55},
		{"responses_prepended_history", responses, clone(evalRequest(withIDs(evalTurns(0, 4), "msg")).Messages), 0.35},
		{"tool_heavy_repeated_execution", toolDistinct, clone(toolDistinct.Messages), 0},
		{"tool_heavy_same_call_ids", toolSame, clone(toolSame.Messages), 0},
		{"confirmations", confirmation, clone(confirmation.Messages), 0},
		{"corrections", correction, clone(correction.Messages), 0},
		{"refusal_retry", refusal, clone(refusal.Messages), 0},
		{"temporal_repetition", temporal, clone(temporal.Messages), 0},
	}
}

func clone(messages []llmprotocol.Message) []llmprotocol.Message {
	return append([]llmprotocol.Message(nil), messages...)
}

// TestEvaluationTokenReductionWithBehavioralEquivalence measures token
// reduction on representative workloads and, for every one of them, asserts
// that the provider-bound sequence is exactly the expected one. Duplicate
// workloads must save at least their stated share of the estimated input
// tokens; every negative workload must be byte-identical before and after.
func TestEvaluationTokenReductionWithBehavioralEquivalence(t *testing.T) {
	counter := contextcompression.HeuristicTokenCounter{}
	var summary []string
	for _, workload := range evalWorkloads() {
		t.Run(workload.name, func(t *testing.T) {
			ir := contextcompression.ParseSemanticRequest(workload.request, contextcompression.Provenance{})
			before, _ := counter.CountRequest("model", ir)
			action := NewAction(testPolicy(), "").WithResolver(RequestResolver(ir))
			if err := ir.ApplySteps(context.Background(), []contextcompression.TransformationStep{action.Step()}); err != nil {
				t.Fatal(err)
			}
			after, _ := counter.CountRequest("model", ir)
			if encoded(t, workload.request.Messages) != encoded(t, workload.want) {
				t.Fatalf("provider-bound sequence differs from the expected one:\n got %s\nwant %s",
					encoded(t, workload.request.Messages), encoded(t, workload.want))
			}
			if after > before {
				t.Fatalf("tokens grew from %d to %d", before, after)
			}
			reduction := 0.0
			if before > 0 {
				reduction = float64(before-after) / float64(before)
			}
			diagnostics := action.Reconcile(ir.Transformations.Receipts())
			if workload.minSaved > 0 && reduction < workload.minSaved {
				t.Fatalf("duplicate workload saved only %.0f%% (%d -> %d): %+v", reduction*100, before, after, diagnostics)
			}
			if workload.minSaved == 0 && (after != before || diagnostics.RemovedMessages != 0) {
				t.Fatalf("negative workload changed: %d -> %d, %+v", before, after, diagnostics)
			}
			summary = append(summary, fmt.Sprintf("%-32s tokens %5d -> %5d  saved %3.0f%%  removed_turns=%d reason=%s",
				workload.name, before, after, reduction*100, diagnostics.RemovedTurns, diagnostics.Reason))
		})
	}
	t.Logf("evaluation summary:\n%s", strings.Join(summary, "\n"))
}

// BenchmarkPlan measures the scan at the default bounds on the most expensive
// shape they admit: a 63-turn history sent twice ahead of the live turn, so
// the whole first copy is compared as one block just under the segment bound.
func BenchmarkPlan(b *testing.B) {
	request := evalRequest(evalTurns(0, 63), evalTurns(0, 63))
	ir := contextcompression.ParseSemanticRequest(request, contextcompression.Provenance{})
	view := ir.TransformationView()
	resolver := RequestResolver(ir)
	policy := testPolicy()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		edits, diagnostics := plan(context.Background(), policy, resolver, view)
		if len(edits.RemoveMessages) != 126 || diagnostics.RemovedTurns != 63 {
			b.Fatalf("unexpected plan %+v", diagnostics)
		}
	}
}
