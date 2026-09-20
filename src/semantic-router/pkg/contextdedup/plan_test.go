package contextdedup

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func testPolicy() Policy {
	return Policy{Normalization: NormalizationExact, MaxHistoryTurns: 128, MaxHistoryBytes: 1 << 20, MaxSegmentTurns: 64}
}

// historyMessage builds one eligible prior-history message.
func historyMessage(id, turn int, role, text string) contextcompression.MessageView {
	return contextcompression.MessageView{
		ID:          id,
		Role:        role,
		Source:      contextcompression.SourceHistory,
		Eligibility: contextcompression.EligibleHistoryRemoval,
		TurnID:      turn,
		Blocks:      []contextcompression.BlockView{{ID: 0, Source: contextcompression.TargetHistory, Text: text}},
	}
}

func withBlocks(message contextcompression.MessageView, texts ...string) contextcompression.MessageView {
	message.Blocks = nil
	for index, text := range texts {
		message.Blocks = append(message.Blocks, contextcompression.BlockView{ID: index, Source: contextcompression.TargetHistory, Text: text})
	}
	return message
}

func withBlockSource(message contextcompression.MessageView, source contextcompression.TargetKind) contextcompression.MessageView {
	for index := range message.Blocks {
		message.Blocks[index].Source = source
	}
	return message
}

func protect(message contextcompression.MessageView, protection contextcompression.Protection) contextcompression.MessageView {
	message.Protection = protection
	message.Eligibility = 0
	return message
}

func withExchange(message contextcompression.MessageView, ids ...string) contextcompression.MessageView {
	message.ExchangeIDs = ids
	return message
}

// turnPair builds one complete user/assistant turn opening at message id.
func turnPair(id int, question, answer string) []contextcompression.MessageView {
	return []contextcompression.MessageView{
		historyMessage(id, id, "user", question),
		historyMessage(id+1, id, "assistant", answer),
	}
}

func conversation(parts ...[]contextcompression.MessageView) []contextcompression.MessageView {
	var messages []contextcompression.MessageView
	for _, part := range parts {
		messages = append(messages, part...)
	}
	return messages
}

func live(id int) []contextcompression.MessageView {
	return []contextcompression.MessageView{protect(historyMessage(id, id, "user", "live"), contextcompression.ProtectLiveTurn)}
}

// viewResolver rebuilds neutral text messages from the view so stage two
// sees exactly what stage one compared.
func viewResolver(messages []contextcompression.MessageView) Resolver {
	return func(id int) (llmprotocol.Message, bool) {
		for _, message := range messages {
			if message.ID != id {
				continue
			}
			result := llmprotocol.Message{Role: llmprotocol.Role(message.Role)}
			for _, block := range message.Blocks {
				result.Content = append(result.Content, llmprotocol.Content{Kind: llmprotocol.ContentText, Text: block.Text})
			}
			return result, true
		}
		return llmprotocol.Message{}, false
	}
}

func planIDs(t *testing.T, policy Policy, messages ...contextcompression.MessageView) ([]int, Diagnostics) {
	t.Helper()
	edits, diagnostics := plan(context.Background(), policy, viewResolver(messages), contextcompression.TransformationView{Messages: messages})
	return edits.RemoveMessages, diagnostics
}

func TestPlanRemovesAdjacentRepeatedTurn(t *testing.T) {
	ids, diagnostics := planIDs(t, testPolicy(), conversation(
		turnPair(0, "q1", "a1"), turnPair(2, "q1", "a1"), live(4),
	)...)
	if !reflect.DeepEqual(ids, []int{2, 3}) {
		t.Fatalf("expected the later copy, got %v", ids)
	}
	if diagnostics.Outcome != OutcomeApplied || diagnostics.Reason != ReasonApplied {
		t.Fatalf("unexpected outcome %+v", diagnostics)
	}
	if diagnostics.RemovedTurns != 1 || diagnostics.RemovedMessages != 2 || diagnostics.DuplicateSegments != 1 {
		t.Fatalf("unexpected removal counts %+v", diagnostics)
	}
	if diagnostics.ExaminedMessages != 5 || diagnostics.ExaminedTurns != 3 || diagnostics.CandidateTurns != 2 {
		t.Fatalf("unexpected examined counts %+v", diagnostics)
	}
	if diagnostics.RetainedMessages != 3 || diagnostics.ProtectedMessages != 1 || diagnostics.RemovedTextBytes != 4 {
		t.Fatalf("unexpected retained counts %+v", diagnostics)
	}
	want := []Segment{{RetainedFirstMessageID: 0, RemovedFirstMessageID: 2, Turns: 1, Messages: 2}}
	if !reflect.DeepEqual(diagnostics.Segments, want) || diagnostics.SegmentsTruncated {
		t.Fatalf("unexpected segments %+v", diagnostics.Segments)
	}
	if diagnostics.Retained[RetainedIneligible] != 1 || len(diagnostics.Retained) != 1 {
		t.Fatalf("unexpected retention reasons %+v", diagnostics.Retained)
	}
	if diagnostics.RecoveryStatus != RecoveryNotRequired || diagnostics.Recovery != RecoveryRetainedTwin {
		t.Fatalf("unexpected recovery detail %+v", diagnostics)
	}
}

func TestPlanCollapsesTripleSendToOneCopy(t *testing.T) {
	ids, diagnostics := planIDs(t, testPolicy(), conversation(
		turnPair(0, "q1", "a1"), turnPair(2, "q1", "a1"), turnPair(4, "q1", "a1"), live(6),
	)...)
	if !reflect.DeepEqual(ids, []int{2, 3, 4, 5}) {
		t.Fatalf("expected both later copies, got %v", ids)
	}
	if diagnostics.DuplicateSegments != 2 || diagnostics.RemovedTurns != 2 {
		t.Fatalf("unexpected counts %+v", diagnostics)
	}
}

func TestPlanRemovesRepeatedMultiTurnSegmentLargestFirst(t *testing.T) {
	// A whole three-turn history sent twice: [A B C A B C live].
	ids, diagnostics := planIDs(t, testPolicy(), conversation(
		turnPair(0, "a", "1"), turnPair(2, "b", "2"), turnPair(4, "c", "3"),
		turnPair(6, "a", "1"), turnPair(8, "b", "2"), turnPair(10, "c", "3"), live(12),
	)...)
	if !reflect.DeepEqual(ids, []int{6, 7, 8, 9, 10, 11}) {
		t.Fatalf("expected the second history copy, got %v", ids)
	}
	if diagnostics.DuplicateSegments != 1 || diagnostics.RemovedTurns != 3 || diagnostics.Segments[0].Turns != 3 {
		t.Fatalf("expected one three-turn segment, got %+v", diagnostics)
	}
}

func TestPlanRetainsNonAdjacentRepetition(t *testing.T) {
	ids, diagnostics := planIDs(t, testPolicy(), conversation(
		turnPair(0, "q1", "a1"), turnPair(2, "q2", "a2"), turnPair(4, "q1", "a1"), live(6),
	)...)
	if len(ids) != 0 || diagnostics.Reason != ReasonNoDuplicates || diagnostics.Outcome != OutcomeSkipped {
		t.Fatalf("temporal repetition must be retained: %v %+v", ids, diagnostics)
	}
	if diagnostics.Retained[RetainedNonAdjacent] != 1 {
		t.Fatalf("expected one non-adjacent turn, got %+v", diagnostics.Retained)
	}
}

func TestPlanIncompleteTurnBreaksAdjacency(t *testing.T) {
	// [U1 A1 U2 U1 A1 live]: the lone U2 is an incomplete turn between copies.
	ids, diagnostics := planIDs(t, testPolicy(), conversation(
		turnPair(0, "q1", "a1"),
		[]contextcompression.MessageView{historyMessage(2, 2, "user", "q2")},
		turnPair(3, "q1", "a1"), live(5),
	)...)
	if len(ids) != 0 || diagnostics.Reason != ReasonNoDuplicates {
		t.Fatalf("expected nothing removed, got %v %+v", ids, diagnostics)
	}
	if diagnostics.Retained[RetainedIncompleteTurn] != 1 || diagnostics.Retained[RetainedNonAdjacent] != 1 {
		t.Fatalf("unexpected retention reasons %+v", diagnostics.Retained)
	}
}

func TestPlanRetainsCorrectionsAndConfirmations(t *testing.T) {
	for name, messages := range map[string][]contextcompression.MessageView{
		"correction": conversation(turnPair(0, "q1", "a1"), turnPair(2, "q1", "a1 corrected"), live(4)),
		"retry_without_reply": conversation(
			[]contextcompression.MessageView{historyMessage(0, 0, "user", "q1")},
			[]contextcompression.MessageView{historyMessage(1, 1, "user", "q1")},
			live(2),
		),
		"confirmation_pair": conversation(
			turnPair(0, "delete it?", "are you sure?"),
			[]contextcompression.MessageView{historyMessage(2, 2, "user", "yes")},
			[]contextcompression.MessageView{historyMessage(3, 3, "user", "yes")},
			live(4),
		),
		"assistant_repeat_inside_turn": conversation(
			[]contextcompression.MessageView{
				historyMessage(0, 0, "user", "q1"),
				historyMessage(1, 0, "assistant", "a1"),
				historyMessage(2, 0, "assistant", "a1"),
			},
			live(3),
		),
	} {
		t.Run(name, func(t *testing.T) {
			ids, diagnostics := planIDs(t, testPolicy(), messages...)
			if len(ids) != 0 || diagnostics.Outcome != OutcomeSkipped {
				t.Fatalf("expected retention, got %v %+v", ids, diagnostics)
			}
			if evaluationFailure(diagnostics.Reason) {
				t.Fatalf("retention must not be an evaluation failure: %+v", diagnostics)
			}
		})
	}
}

func TestPlanRetainsToolExchanges(t *testing.T) {
	exchange := func(id int, call string) []contextcompression.MessageView {
		return []contextcompression.MessageView{
			historyMessage(id, id, "user", "run it"),
			withExchange(withBlocks(historyMessage(id+1, id, "assistant", "")), call),
			withExchange(withBlockSource(historyMessage(id+2, id, "tool", "result"), contextcompression.TargetToolOutput), call),
			historyMessage(id+3, id, "assistant", "done"),
		}
	}
	for name, messages := range map[string][]contextcompression.MessageView{
		"distinct_call_ids": conversation(exchange(0, "call-1"), exchange(4, "call-2"), live(8)),
		"repeated_call_ids": conversation(exchange(0, "call-1"), exchange(4, "call-1"), live(8)),
	} {
		t.Run(name, func(t *testing.T) {
			ids, diagnostics := planIDs(t, testPolicy(), messages...)
			if len(ids) != 0 || diagnostics.Retained[RetainedToolExchange] != 2 {
				t.Fatalf("tool exchanges must be retained: %v %+v", ids, diagnostics.Retained)
			}
		})
	}
}

func TestPlanAnthropicToolResultContinuesTurn(t *testing.T) {
	// Anthropic carries tool results in user messages that do not open a
	// turn, so the exchange keeps the whole turn out of scope.
	ids, diagnostics := planIDs(t, testPolicy(), conversation(
		[]contextcompression.MessageView{
			historyMessage(0, 0, "user", "run it"),
			withExchange(withBlocks(historyMessage(1, 0, "assistant", "")), "toolu_1"),
			withExchange(withBlockSource(historyMessage(2, 0, "user", "result"), contextcompression.TargetToolOutput), "toolu_1"),
			historyMessage(3, 0, "assistant", "done"),
		},
		[]contextcompression.MessageView{
			historyMessage(4, 4, "user", "run it"),
			withExchange(withBlocks(historyMessage(5, 4, "assistant", "")), "toolu_2"),
			withExchange(withBlockSource(historyMessage(6, 4, "user", "result"), contextcompression.TargetToolOutput), "toolu_2"),
			historyMessage(7, 4, "assistant", "done"),
		},
		live(8),
	)...)
	if len(ids) != 0 || diagnostics.Retained[RetainedToolExchange] != 2 {
		t.Fatalf("expected both exchanges retained, got %v %+v", ids, diagnostics.Retained)
	}
}

func TestPlanIgnoresEnrichedProtectedAndOpaqueContent(t *testing.T) {
	memory := historyMessage(0, -1, "system", "remembered")
	memory.Source, memory.Eligibility = contextcompression.SourceMemory, 0
	memory.Protection = contextcompression.ProtectUnknown
	rag := withExchange(historyMessage(7, 5, "tool", "evidence"), "rag_1")
	rag.Source, rag.Eligibility = contextcompression.SourceRAG, 0
	rag.Protection = contextcompression.ProtectLiveTurn
	ids, diagnostics := planIDs(t, testPolicy(), conversation(
		[]contextcompression.MessageView{memory},
		turnPair(1, "q1", "a1"), turnPair(3, "q1", "a1"),
		[]contextcompression.MessageView{protect(historyMessage(5, 5, "user", "live"), contextcompression.ProtectLiveTurn), protect(withExchange(historyMessage(6, 5, "assistant", ""), "rag_1"), contextcompression.ProtectLiveTurn), rag},
	)...)
	if !reflect.DeepEqual(ids, []int{3, 4}) {
		t.Fatalf("memory and RAG content must not block history dedup: %v %+v", ids, diagnostics)
	}
	if diagnostics.Retained[RetainedIneligible] != 2 {
		t.Fatalf("unexpected retention reasons %+v", diagnostics.Retained)
	}

	for name, messages := range map[string][]contextcompression.MessageView{
		"multimodal_copy": conversation(turnPair(0, "q1", "a1"), []contextcompression.MessageView{
			protect(historyMessage(2, 2, "user", "q1"), contextcompression.ProtectMultimodal), historyMessage(3, 2, "assistant", "a1"),
		}, live(4)),
		"authorization": conversation(turnPair(0, "q1", "a1"), []contextcompression.MessageView{
			protect(historyMessage(2, 2, "user", "q1"), contextcompression.ProtectAuthorization), historyMessage(3, 2, "assistant", "a1"),
		}, live(4)),
		"safety": conversation([]contextcompression.MessageView{
			protect(historyMessage(0, 0, "user", "q1"), contextcompression.ProtectSafety), historyMessage(1, 0, "assistant", "a1"),
		}, turnPair(2, "q1", "a1"), live(4)),
		"instructions": conversation(
			[]contextcompression.MessageView{protect(historyMessage(0, -1, "system", "rules"), contextcompression.ProtectInstructions|contextcompression.ProtectUnknown)},
			[]contextcompression.MessageView{protect(historyMessage(1, -1, "system", "rules"), contextcompression.ProtectInstructions|contextcompression.ProtectUnknown)},
			live(2),
		),
		"live_turn_copy": conversation(turnPair(0, "q1", "a1"), []contextcompression.MessageView{
			protect(historyMessage(2, 2, "user", "q1"), contextcompression.ProtectLiveTurn), protect(historyMessage(3, 2, "assistant", "a1"), contextcompression.ProtectLiveTurn),
		}),
		"blockless_message": conversation(turnPair(0, "q1", "a1"), []contextcompression.MessageView{
			historyMessage(2, 2, "user", "q1"), withBlocks(historyMessage(3, 2, "assistant", "a1")),
		}, live(4)),
		"memory_block_source": conversation(turnPair(0, "q1", "a1"), []contextcompression.MessageView{
			historyMessage(2, 2, "user", "q1"), withBlockSource(historyMessage(3, 2, "assistant", "a1"), contextcompression.TargetMemory),
		}, live(4)),
	} {
		t.Run(name, func(t *testing.T) {
			ids, diagnostics := planIDs(t, testPolicy(), messages...)
			if len(ids) != 0 || diagnostics.Outcome != OutcomeSkipped || evaluationFailure(diagnostics.Reason) {
				t.Fatalf("expected retention, got %v %+v", ids, diagnostics)
			}
		})
	}
}

func TestPlanReportsNoEligibleHistory(t *testing.T) {
	ids, diagnostics := planIDs(t, testPolicy(), live(0)...)
	if len(ids) != 0 || diagnostics.Reason != ReasonNoEligibleHistory || diagnostics.CandidateTurns != 0 {
		t.Fatalf("unexpected result %v %+v", ids, diagnostics)
	}
}

func TestPlanRejectsWholeStepBeyondLimits(t *testing.T) {
	messages := conversation(turnPair(0, "q1", "a1"), turnPair(2, "q1", "a1"), turnPair(4, "q2", "a2"), live(6))
	for name, policy := range map[string]Policy{
		"turns": {MaxHistoryTurns: 3, MaxSegmentTurns: 64},
		"bytes": {MaxHistoryBytes: 10, MaxSegmentTurns: 64},
	} {
		t.Run(name, func(t *testing.T) {
			ids, diagnostics := planIDs(t, policy, messages...)
			if len(ids) != 0 || diagnostics.Reason != ReasonHistoryLimitExceeded || !evaluationFailure(diagnostics.Reason) {
				t.Fatalf("a prefix must never be deduplicated: %v %+v", ids, diagnostics)
			}
		})
	}
	ids, _ := planIDs(t, Policy{MaxHistoryTurns: 4, MaxHistoryBytes: 1 << 20, MaxSegmentTurns: 64}, messages...)
	if !reflect.DeepEqual(ids, []int{2, 3}) {
		t.Fatalf("history at the limit must still be deduplicated, got %v", ids)
	}
}

func TestPlanSegmentBoundLimitsPeriod(t *testing.T) {
	messages := conversation(turnPair(0, "a", "1"), turnPair(2, "b", "2"), turnPair(4, "a", "1"), turnPair(6, "b", "2"), live(8))
	policy := testPolicy()
	policy.MaxSegmentTurns = 1
	ids, diagnostics := planIDs(t, policy, messages...)
	if len(ids) != 0 || diagnostics.Reason != ReasonNoDuplicates || diagnostics.Retained[RetainedNonAdjacent] != 2 {
		t.Fatalf("a period-two repeat needs a bound of two: %v %+v", ids, diagnostics)
	}
	policy.MaxSegmentTurns = 2
	if ids, _ = planIDs(t, policy, messages...); !reflect.DeepEqual(ids, []int{4, 5, 6, 7}) {
		t.Fatalf("expected the repeated pair removed, got %v", ids)
	}
}

func TestPlanRequiresStageTwoProof(t *testing.T) {
	messages := conversation(turnPair(0, "q1", "a1"), turnPair(2, "q1", "a1"), live(4))
	view := contextcompression.TransformationView{Messages: messages}
	edits, diagnostics := plan(context.Background(), testPolicy(), nil, view)
	if len(edits.RemoveMessages) != 0 || diagnostics.Reason != ReasonEquivalenceUnverifiable {
		t.Fatalf("no resolver must remove nothing: %+v", diagnostics)
	}
	missing := func(id int) (llmprotocol.Message, bool) { return llmprotocol.Message{}, false }
	if edits, diagnostics = plan(context.Background(), testPolicy(), missing, view); len(edits.RemoveMessages) != 0 || diagnostics.Reason != ReasonEquivalenceUnverifiable {
		t.Fatalf("an unresolved message must remove nothing: %+v", diagnostics)
	}
	base := viewResolver(messages)
	differentID := func(id int) (llmprotocol.Message, bool) {
		message, ok := base(id)
		message.ID = fmt.Sprintf("msg_%d", id)
		return message, ok
	}
	if edits, diagnostics = plan(context.Background(), testPolicy(), differentID, view); len(edits.RemoveMessages) != 0 || diagnostics.Retained[RetainedIdentityMismatch] != 1 {
		t.Fatalf("distinct item ids must retain the later copy: %v %+v", edits.RemoveMessages, diagnostics)
	}
	refusing := func(id int) (llmprotocol.Message, bool) {
		message, ok := base(id)
		if message.Role == llmprotocol.RoleAssistant {
			message.Content[0].Kind = llmprotocol.ContentRefusal
		}
		return message, ok
	}
	if edits, diagnostics = plan(context.Background(), testPolicy(), refusing, view); len(edits.RemoveMessages) != 0 || diagnostics.Retained[RetainedRefusal] != 1 {
		t.Fatalf("a refusal must retain the later copy: %v %+v", edits.RemoveMessages, diagnostics)
	}
	if !evaluationFailure(ReasonEquivalenceUnverifiable) || evaluationFailure(diagnostics.Reason) {
		t.Fatalf("only unverifiable equivalence is a failure, got %q", diagnostics.Reason)
	}
}

func TestPlanNoResolverIsHarmlessWithoutMatches(t *testing.T) {
	messages := conversation(turnPair(0, "q1", "a1"), turnPair(2, "q2", "a2"), live(4))
	edits, diagnostics := plan(context.Background(), testPolicy(), nil, contextcompression.TransformationView{Messages: messages})
	if len(edits.RemoveMessages) != 0 || diagnostics.Reason != ReasonNoDuplicates {
		t.Fatalf("a clean history needs no proof: %+v", diagnostics)
	}
}

func TestPlanHonoursCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	messages := conversation(turnPair(0, "q1", "a1"), turnPair(2, "q1", "a1"), live(4))
	edits, diagnostics := plan(ctx, testPolicy(), viewResolver(messages), contextcompression.TransformationView{Messages: messages})
	if len(edits.RemoveMessages) != 0 || diagnostics.Reason != ReasonCancelled {
		t.Fatalf("cancelled context must remove nothing: %+v", diagnostics)
	}
}

func TestPlanBoundsReceiptSegments(t *testing.T) {
	var parts [][]contextcompression.MessageView
	id := 0
	for turn := 0; turn < MaxReceiptSegments+2; turn++ {
		question := fmt.Sprintf("q%d", turn)
		parts = append(parts, turnPair(id, question, "a"), turnPair(id+2, question, "a"))
		id += 4
	}
	parts = append(parts, live(id))
	ids, diagnostics := planIDs(t, testPolicy(), conversation(parts...)...)
	if len(ids) != 2*(MaxReceiptSegments+2) || diagnostics.DuplicateSegments != MaxReceiptSegments+2 {
		t.Fatalf("unexpected removal %d %+v", len(ids), diagnostics)
	}
	if len(diagnostics.Segments) != MaxReceiptSegments || !diagnostics.SegmentsTruncated {
		t.Fatalf("segments must be bounded: %d truncated=%v", len(diagnostics.Segments), diagnostics.SegmentsTruncated)
	}
}

func TestPlanIsDeterministicAcrossRepeats(t *testing.T) {
	messages := conversation(
		turnPair(0, "a", "1"), turnPair(2, "a", "1"), turnPair(4, "b", "2"), turnPair(6, "b", "2"),
		turnPair(8, "a", "1"), turnPair(10, "b", "2"), turnPair(12, "a", "1"), turnPair(14, "b", "2"), live(16),
	)
	first, firstDiagnostics := planIDs(t, testPolicy(), messages...)
	for i := 0; i < 5; i++ {
		ids, diagnostics := planIDs(t, testPolicy(), messages...)
		if !reflect.DeepEqual(ids, first) || !reflect.DeepEqual(diagnostics, firstDiagnostics) {
			t.Fatalf("plan is not deterministic: %v vs %v", ids, first)
		}
	}
	if !reflect.DeepEqual(first, []int{2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}) {
		t.Fatalf("unexpected removals %v", first)
	}
}
