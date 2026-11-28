package turnselection

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func textMsg(role llmprotocol.Role, text string) llmprotocol.Message {
	return llmprotocol.Message{
		Role:    role,
		Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}},
	}
}

func toolCallMsg(calls ...llmprotocol.ToolCall) llmprotocol.Message {
	content := make([]llmprotocol.Content, 0, len(calls))
	for i := range calls {
		call := calls[i]
		content = append(content, llmprotocol.Content{
			Kind:     llmprotocol.ContentToolCall,
			ToolCall: &call,
		})
	}
	return llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: content}
}

func toolResultMsg(role llmprotocol.Role, callID string, text string) llmprotocol.Message {
	return llmprotocol.Message{
		Role: role,
		Content: []llmprotocol.Content{{
			Kind: llmprotocol.ContentToolResult,
			ToolResult: &llmprotocol.ToolResult{
				CallID:  callID,
				Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}},
			},
		}},
	}
}

func mustMarshalJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal json: %v", err)
	}
	return string(b)
}

func TestGroupTurnsCompleteOrderAndEligibility(t *testing.T) {
	req := &llmprotocol.Request{
		Model: "test-model",
		Messages: []llmprotocol.Message{
			textMsg(llmprotocol.RoleUser, "first question"),
			textMsg(llmprotocol.RoleAssistant, "first answer"),
			textMsg(llmprotocol.RoleUser, "second question"),
			textMsg(llmprotocol.RoleAssistant, "second answer"),
			textMsg(llmprotocol.RoleUser, "live question"),
		},
	}
	before := mustMarshalJSON(t, req)
	ir := contextcompression.ParseSemanticRequest(req, contextcompression.Provenance{})
	turns := GroupTurns(ir.TransformationView(), DefaultMaxTurnTextBytes)

	if len(turns) != 3 {
		t.Fatalf("len(turns) = %d, want 3", len(turns))
	}
	if !reflect.DeepEqual(turns[0].MessageIDs, []int{0, 1}) || !turns[0].Eligible || turns[0].TurnID != 0 {
		t.Fatalf("unexpected turn 0: %+v", turns[0])
	}
	if !reflect.DeepEqual(turns[1].MessageIDs, []int{2, 3}) || !turns[1].Eligible || turns[1].TurnID != 2 {
		t.Fatalf("unexpected turn 1: %+v", turns[1])
	}
	if !reflect.DeepEqual(turns[2].MessageIDs, []int{4}) || turns[2].Eligible || !turns[2].IsLive || turns[2].PinReason != PinLiveTurn {
		t.Fatalf("unexpected live turn: %+v", turns[2])
	}
	if mustMarshalJSON(t, req) != before {
		t.Fatal("GroupTurns mutated request")
	}
}

func TestGroupTurnsIntraTurnParallelToolExchangeEligible(t *testing.T) {
	req := &llmprotocol.Request{
		Model: "test-model",
		Messages: []llmprotocol.Message{
			textMsg(llmprotocol.RoleUser, "check db config and memory"),
			toolCallMsg(
				llmprotocol.ToolCall{ID: "c1", Name: "get_config", Arguments: `{"db":"prod"}`},
				llmprotocol.ToolCall{ID: "c2", Name: "get_memory", Arguments: `{"host":"db1"}`},
			),
			toolResultMsg(llmprotocol.RoleTool, "c1", "max_connections=200"),
			toolResultMsg(llmprotocol.RoleTool, "c2", "ram=2GB"),
			textMsg(llmprotocol.RoleAssistant, "both checks complete"),
			textMsg(llmprotocol.RoleUser, "live question"),
		},
	}
	ir := contextcompression.ParseSemanticRequest(req, contextcompression.Provenance{})
	turns := GroupTurns(ir.TransformationView(), DefaultMaxTurnTextBytes)
	if len(turns) != 2 {
		t.Fatalf("len(turns) = %d, want 2", len(turns))
	}
	if !turns[0].Eligible {
		t.Fatalf("parallel tool turn should be eligible as a complete unit, pin=%q", turns[0].PinReason)
	}
	if !reflect.DeepEqual(turns[0].MessageIDs, []int{0, 1, 2, 3, 4}) {
		t.Fatalf("turn 0 message IDs = %v", turns[0].MessageIDs)
	}
	if !reflect.DeepEqual(turns[0].ExchangeIDs, []string{"c1", "c2"}) {
		t.Fatalf("turn 0 exchange IDs = %v", turns[0].ExchangeIDs)
	}
}

func TestGroupTurnsAnthropicUserToolResultContinuation(t *testing.T) {
	req := &llmprotocol.Request{
		Model: "claude-test",
		Messages: []llmprotocol.Message{
			textMsg(llmprotocol.RoleUser, "inspect schema"),
			toolCallMsg(llmprotocol.ToolCall{ID: "tu_1", Name: "describe_table", Arguments: `{"table":"orders"}`}),
			toolResultMsg(llmprotocol.RoleUser, "tu_1", "columns: id, user_id, total"),
			textMsg(llmprotocol.RoleAssistant, "schema inspected"),
			textMsg(llmprotocol.RoleUser, "write sql query"),
		},
	}
	ir := contextcompression.ParseSemanticRequest(req, contextcompression.Provenance{})
	turns := GroupTurns(ir.TransformationView(), DefaultMaxTurnTextBytes)
	if len(turns) != 2 {
		t.Fatalf("len(turns) = %d, want 2 (user tool_result must not start a new turn)", len(turns))
	}
	if !reflect.DeepEqual(turns[0].MessageIDs, []int{0, 1, 2, 3}) || !turns[0].Eligible {
		t.Fatalf("unexpected Anthropic tool turn: %+v", turns[0])
	}
}

func TestGroupTurnsCrossTurnToolExchangePinnedConservatively(t *testing.T) {
	// A user message carrying both a tool_result for an earlier call AND new
	// user text opens a new turn while sharing ExchangeID "c1" with the earlier turn.
	req := &llmprotocol.Request{
		Model: "claude-test",
		Messages: []llmprotocol.Message{
			textMsg(llmprotocol.RoleUser, "run lookup"),
			toolCallMsg(llmprotocol.ToolCall{ID: "c1", Name: "lookup", Arguments: `{}`}),
			{
				Role: llmprotocol.RoleUser,
				Content: []llmprotocol.Content{
					{
						Kind: llmprotocol.ContentToolResult,
						ToolResult: &llmprotocol.ToolResult{
							CallID:  "c1",
							Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "lookup result"}},
						},
					},
					{Kind: llmprotocol.ContentText, Text: "and follow up on that"},
				},
			},
			textMsg(llmprotocol.RoleAssistant, "follow up answer"),
			textMsg(llmprotocol.RoleUser, "live question"),
		},
	}
	ir := contextcompression.ParseSemanticRequest(req, contextcompression.Provenance{})
	turns := GroupTurns(ir.TransformationView(), DefaultMaxTurnTextBytes)
	if len(turns) != 3 {
		t.Fatalf("len(turns) = %d, want 3", len(turns))
	}
	for i := 0; i < 2; i++ {
		if turns[i].Eligible || turns[i].PinReason != PinCrossTurnExchange {
			t.Fatalf("turn %d should be pinned for cross-turn exchange, got eligible=%v reason=%q",
				i, turns[i].Eligible, turns[i].PinReason)
		}
	}
}

func TestGroupTurnsUnresolvedToolExchangePinned(t *testing.T) {
	req := &llmprotocol.Request{
		Model: "test-model",
		Messages: []llmprotocol.Message{
			textMsg(llmprotocol.RoleUser, "run lookup"),
			toolCallMsg(llmprotocol.ToolCall{ID: "c_missing_result", Name: "lookup", Arguments: `{}`}),
			textMsg(llmprotocol.RoleUser, "live question"),
		},
	}
	ir := contextcompression.ParseSemanticRequest(req, contextcompression.Provenance{})
	turns := GroupTurns(ir.TransformationView(), DefaultMaxTurnTextBytes)
	if len(turns) != 2 {
		t.Fatalf("len(turns) = %d, want 2", len(turns))
	}
	if turns[0].Eligible {
		t.Fatalf("turn with unresolved tool call must not be eligible: %+v", turns[0])
	}
}

func TestGroupTurnsProtectedSourcesAndBlocksPinned(t *testing.T) {
	cases := []struct {
		name       string
		build      func() (*llmprotocol.Request, contextcompression.Provenance)
		wantReason string
	}{
		{
			name: "authorization",
			build: func() (*llmprotocol.Request, contextcompression.Provenance) {
				return &llmprotocol.Request{
					Messages: []llmprotocol.Message{
						textMsg(llmprotocol.RoleUser, "auth policy"),
						textMsg(llmprotocol.RoleAssistant, "acknowledged"),
						textMsg(llmprotocol.RoleUser, "live"),
					},
				}, contextcompression.Provenance{ProtectedMessages: map[int]contextcompression.Protection{0: contextcompression.ProtectAuthorization}}
			},
			wantReason: PinAuthorization,
		},
		{
			name: "safety",
			build: func() (*llmprotocol.Request, contextcompression.Provenance) {
				return &llmprotocol.Request{
					Messages: []llmprotocol.Message{
						textMsg(llmprotocol.RoleUser, "safety constraint"),
						textMsg(llmprotocol.RoleAssistant, "acknowledged"),
						textMsg(llmprotocol.RoleUser, "live"),
					},
				}, contextcompression.Provenance{ProtectedMessages: map[int]contextcompression.Protection{1: contextcompression.ProtectSafety}}
			},
			wantReason: PinSafety,
		},
		{
			name: "multimodal_image",
			build: func() (*llmprotocol.Request, contextcompression.Provenance) {
				return &llmprotocol.Request{
					Messages: []llmprotocol.Message{
						{
							Role: llmprotocol.RoleUser,
							Content: []llmprotocol.Content{
								{Kind: llmprotocol.ContentText, Text: "see screenshot"},
								{Kind: llmprotocol.ContentImage, URL: "https://example.invalid/arch.png"},
							},
						},
						textMsg(llmprotocol.RoleAssistant, "looks good"),
						textMsg(llmprotocol.RoleUser, "live"),
					},
				}, contextcompression.Provenance{}
			},
			wantReason: PinMultimodal,
		},
		{
			name: "reasoning_block_conservatively_protected",
			build: func() (*llmprotocol.Request, contextcompression.Provenance) {
				return &llmprotocol.Request{
					Messages: []llmprotocol.Message{
						textMsg(llmprotocol.RoleUser, "solve equation"),
						{
							Role: llmprotocol.RoleAssistant,
							Content: []llmprotocol.Content{
								{Kind: llmprotocol.ContentReasoning, Text: "step-by-step derivation"},
								{Kind: llmprotocol.ContentText, Text: "x = 42"},
							},
						},
						textMsg(llmprotocol.RoleUser, "live"),
					},
				}, contextcompression.Provenance{}
			},
			wantReason: PinMultimodal,
		},
		{
			name: "rag_injected_turn",
			build: func() (*llmprotocol.Request, contextcompression.Provenance) {
				return &llmprotocol.Request{
					Messages: []llmprotocol.Message{
						textMsg(llmprotocol.RoleUser, "prior question"),
						toolCallMsg(llmprotocol.ToolCall{ID: "rag_1", Name: "vsr_rag_context", Arguments: `{}`}),
						toolResultMsg(llmprotocol.RoleTool, "rag_1", "retrieved doc"),
						textMsg(llmprotocol.RoleAssistant, "prior answer"),
						textMsg(llmprotocol.RoleUser, "live"),
					},
				}, contextcompression.Provenance{RAGToolCallIDs: map[string]struct{}{"rag_1": {}}}
			},
			wantReason: PinNonHistorySource,
		},
		{
			name: "memory_injected_turn",
			build: func() (*llmprotocol.Request, contextcompression.Provenance) {
				return &llmprotocol.Request{
					Messages: []llmprotocol.Message{
						textMsg(llmprotocol.RoleUser, "prior question"),
						toolCallMsg(llmprotocol.ToolCall{ID: "memory_1", Name: "vsr_memory_context", Arguments: `{}`}),
						toolResultMsg(llmprotocol.RoleTool, "memory_1", "user prefers Go"),
						textMsg(llmprotocol.RoleAssistant, "prior answer"),
						textMsg(llmprotocol.RoleUser, "live"),
					},
				}, contextcompression.Provenance{MemoryToolCallIDs: map[string]struct{}{"memory_1": {}}}
			},
			wantReason: PinNonHistorySource,
		},
		{
			name: "leading_message_without_turn",
			build: func() (*llmprotocol.Request, contextcompression.Provenance) {
				return &llmprotocol.Request{
					Messages: []llmprotocol.Message{
						textMsg(llmprotocol.RoleAssistant, "leading assistant greeting"),
						textMsg(llmprotocol.RoleUser, "live"),
					},
				}, contextcompression.Provenance{}
			},
			wantReason: PinUnknown,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req, prov := tc.build()
			ir := contextcompression.ParseSemanticRequest(req, prov)
			turns := GroupTurns(ir.TransformationView(), DefaultMaxTurnTextBytes)
			if len(turns) < 2 {
				t.Fatalf("expected at least 2 turns, got %d", len(turns))
			}
			if turns[0].Eligible {
				t.Fatalf("turn 0 should be pinned, got eligible=true: %+v", turns[0])
			}
			if turns[0].PinReason != tc.wantReason {
				t.Fatalf("turn 0 pin reason = %q, want %q", turns[0].PinReason, tc.wantReason)
			}
		})
	}
}

func TestGroupTurnsPreservesStableIDsAfterUpstreamRemoval(t *testing.T) {
	req := &llmprotocol.Request{
		Model: "test-model",
		Messages: []llmprotocol.Message{
			textMsg(llmprotocol.RoleUser, "reset topic question"),
			textMsg(llmprotocol.RoleAssistant, "reset topic answer"),
			textMsg(llmprotocol.RoleUser, "surviving old question"),
			textMsg(llmprotocol.RoleAssistant, "surviving old answer"),
			textMsg(llmprotocol.RoleUser, "live question"),
		},
	}
	ir := contextcompression.ParseSemanticRequest(req, contextcompression.Provenance{})
	resetStep := contextcompression.TransformationStep{
		Kind:        contextcompression.TransformReset,
		Enabled:     true,
		FailureMode: contextcompression.FailureClosed,
		Propose: func(context.Context, contextcompression.TransformationView) (contextcompression.TransformationEdits, error) {
			return contextcompression.TransformationEdits{RemoveMessages: []int{0, 1}}, nil
		},
	}
	if err := ir.ApplySteps(context.Background(), []contextcompression.TransformationStep{resetStep}); err != nil {
		t.Fatalf("apply reset: %v", err)
	}
	turns := GroupTurns(ir.TransformationView(), DefaultMaxTurnTextBytes)
	if len(turns) != 2 {
		t.Fatalf("len(turns) = %d, want 2", len(turns))
	}
	if turns[0].TurnID != 2 || !reflect.DeepEqual(turns[0].MessageIDs, []int{2, 3}) {
		t.Fatalf("surviving turn lost stable IDs: %+v", turns[0])
	}
	if turns[1].TurnID != 4 || !reflect.DeepEqual(turns[1].MessageIDs, []int{4}) {
		t.Fatalf("live turn lost stable IDs: %+v", turns[1])
	}
}

func TestExtractLiveQueryIgnoresInjectedRAGAndMemoryAndTruncatesUTF8(t *testing.T) {
	req := &llmprotocol.Request{
		Messages: []llmprotocol.Message{
			textMsg(llmprotocol.RoleUser, "old turn"),
			textMsg(llmprotocol.RoleAssistant, "old reply"),
			textMsg(llmprotocol.RoleUser, "请继续第二个方案并检查数据库连接池配置"),
			toolCallMsg(llmprotocol.ToolCall{ID: "rag_1", Name: "vsr_rag_context", Arguments: `{}`}),
			toolResultMsg(llmprotocol.RoleTool, "rag_1", "RAG_INJECTED_SHOULD_NOT_ENTER_QUERY"),
			toolCallMsg(llmprotocol.ToolCall{ID: "memory_1", Name: "vsr_memory_context", Arguments: `{}`}),
			toolResultMsg(llmprotocol.RoleTool, "memory_1", "MEMORY_INJECTED_SHOULD_NOT_ENTER_QUERY"),
		},
	}
	ir := contextcompression.ParseSemanticRequest(req, contextcompression.Provenance{
		RAGToolCallIDs:    map[string]struct{}{"rag_1": {}},
		MemoryToolCallIDs: map[string]struct{}{"memory_1": {}},
	})
	turns := GroupTurns(ir.TransformationView(), DefaultMaxTurnTextBytes)
	query := extractLiveQuery(turns, 19)
	if !utf8.ValidString(query) {
		t.Fatalf("truncated query is invalid UTF-8: %q", query)
	}
	if len(query) > 19 {
		t.Fatalf("query byte len = %d, want <= 19", len(query))
	}
	if strings.Contains(query, "RAG_INJECTED") || strings.Contains(query, "MEMORY_INJECTED") {
		t.Fatalf("query leaked injected RAG/Memory text: %q", query)
	}
}
