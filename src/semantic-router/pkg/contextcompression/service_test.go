package contextcompression

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"
)

type testRecoveryStore struct {
	entries map[string]RecoveryEntry
	err     error
}

func (store *testRecoveryStore) Put(
	_ context.Context,
	entry RecoveryEntry,
	_ time.Duration,
) error {
	if store.err != nil {
		return store.err
	}
	if store.entries == nil {
		store.entries = make(map[string]RecoveryEntry)
	}
	store.entries[entry.Scope+":"+entry.Key] = entry
	return nil
}

func (store *testRecoveryStore) Get(
	_ context.Context,
	scope string,
	key string,
) (RecoveryEntry, error) {
	entry, ok := store.entries[scope+":"+key]
	if !ok {
		return RecoveryEntry{}, ErrRecoveryNotFound
	}
	return entry, nil
}

func (store *testRecoveryStore) Health(context.Context) error { return store.err }
func (store *testRecoveryStore) Close() error                 { return nil }
func (store *testRecoveryStore) InvalidateScope(
	_ context.Context,
	scope string,
) (int64, error) {
	var count int64
	for key, entry := range store.entries {
		if entry.Scope == scope {
			delete(store.entries, key)
			count++
		}
	}
	return count, store.err
}

func TestServiceUsesToolIntentAndRequestBudget(t *testing.T) {
	body := map[string]interface{}{
		"model": "model",
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "find authentication failures"},
			map[string]interface{}{
				"role": "assistant",
				"tool_calls": []interface{}{map[string]interface{}{
					"id": "call_1",
					"function": map[string]interface{}{
						"name":      "search_logs",
						"arguments": `{"service":"auth"}`,
					},
				}},
			},
			map[string]interface{}{
				"role":         "tool",
				"tool_call_id": "call_1",
				"content": strings.Repeat("irrelevant billing row\n", 300) +
					"auth validator failed\n" +
					strings.Repeat("irrelevant inventory row\n", 300),
			},
			map[string]interface{}{
				"role":         "tool",
				"tool_call_id": "call_2",
				"content":      strings.Repeat("unrelated trace\n", 500),
			},
		},
	}
	ir := ParseRequestIR(body, Provenance{})
	service := NewService()
	result := service.Apply(context.Background(), Request{
		Model:   "model",
		Request: ir,
		Policy: Policy{
			Mode: ModeAuto,
			Budget: Budget{
				TargetTokens: 500,
			},
			Targets: Targets{
				ToolOutputs: TargetPolicy{
					Mode:         TargetExtractive,
					MinTokens:    100,
					TargetTokens: 120,
				},
			},
			Scoring: Scoring{Method: "bm25"},
		},
		TokenCounter: HeuristicTokenCounter{},
	})
	if !result.Applied || result.BlocksCompressed != 2 {
		t.Fatalf("Apply() = %#v, want two compressed blocks", result)
	}
	if !strings.Contains(result.Plan.Targets[0].Query+result.Plan.Targets[1].Query, "search_logs") {
		t.Fatalf("tool intent was not used: %#v", result.Plan.Targets)
	}
	if result.TokensAfter >= result.TokensBefore {
		t.Fatalf("tokens did not decrease: before=%d after=%d", result.TokensBefore, result.TokensAfter)
	}
	encoded, _ := json.Marshal(body)
	if !strings.Contains(string(encoded), "auth validator failed") {
		t.Fatalf("relevant tool evidence was not retained: %s", encoded)
	}
	if result.Plan.Quality == "" {
		t.Fatal("quality guard outcome was not recorded")
	}
}

func TestServiceAccumulatesEstimatedCostSavings(t *testing.T) {
	service := NewService()
	service.RecordEstimatedCostSavings(0.125)
	service.RecordEstimatedCostSavings(0.375)
	if got := service.Stats().EstimatedCostSavedUSD; got != 0.5 {
		t.Fatalf("EstimatedCostSavedUSD = %v, want 0.5", got)
	}
}

func TestServiceProtectsLiveHistory(t *testing.T) {
	oldUser := strings.Repeat("old project notes ", 300)
	body := map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "system", "content": "system policy"},
			map[string]interface{}{"role": "user", "content": oldUser},
			map[string]interface{}{"role": "assistant", "content": strings.Repeat("old answer ", 300)},
			map[string]interface{}{"role": "user", "content": "live question"},
			map[string]interface{}{"role": "assistant", "content": "latest assistant"},
		},
	}
	ir := ParseRequestIR(body, Provenance{})
	result := NewService().Apply(context.Background(), Request{
		Request: ir,
		Policy: Policy{
			Mode: ModeAlways,
			Targets: Targets{
				History: TargetPolicy{
					Mode:         TargetExtractive,
					MinTokens:    10,
					TargetTokens: 30,
				},
			},
		},
		TokenCounter: HeuristicTokenCounter{},
	})
	if !result.Applied {
		t.Fatal("history compression was not applied")
	}
	messages := body["messages"].([]interface{})
	if messages[0].(map[string]interface{})["content"] != "system policy" ||
		messages[3].(map[string]interface{})["content"] != "live question" ||
		messages[4].(map[string]interface{})["content"] != "latest assistant" {
		t.Fatal("protected history was modified")
	}
	if messages[1].(map[string]interface{})["content"] == oldUser {
		t.Fatal("old history was not compressed")
	}
}

func TestServiceProtectsTypedRAGByDefault(t *testing.T) {
	content := strings.Repeat("retrieved evidence ", 300)
	body := map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "question"},
			map[string]interface{}{
				"role":         "tool",
				"tool_call_id": "custom-rag-id",
				"content":      content,
			},
		},
	}
	ir := ParseRequestIR(body, Provenance{
		RAGToolCallIDs: map[string]struct{}{"custom-rag-id": {}},
	})
	result := NewService().Apply(context.Background(), Request{
		Request: ir,
		Policy: Policy{
			Mode: ModeAlways,
			Targets: Targets{
				ToolOutputs: TargetPolicy{Mode: TargetExtractive, MinTokens: 10, TargetTokens: 30},
				RAG:         TargetPolicy{Mode: TargetPreserve},
			},
		},
		TokenCounter: HeuristicTokenCounter{},
	})
	if result.Applied {
		t.Fatal("protected RAG content was compressed")
	}
}

func TestServiceProtectsTypedMemoryByDefault(t *testing.T) {
	content := strings.Repeat("personal memory evidence ", 300)
	body := map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "system", "content": "system"},
			map[string]interface{}{"role": "user", "content": content},
			map[string]interface{}{"role": "user", "content": "live question"},
		},
	}
	ir := ParseRequestIR(body, Provenance{
		MemoryMessageIndexes: map[int]struct{}{1: {}},
	})
	result := NewService().Apply(context.Background(), Request{
		Request: ir,
		Policy: Policy{
			Mode: ModeAlways,
			Targets: Targets{
				History: TargetPolicy{Mode: TargetExtractive, MinTokens: 10, TargetTokens: 30},
				Memory:  TargetPolicy{Mode: TargetPreserve},
			},
		},
		TokenCounter: HeuristicTokenCounter{},
	})
	if result.Applied {
		t.Fatal("protected memory content was compressed")
	}
}

func TestParseRequestIRProtectsAtomicLatestToolExchange(t *testing.T) {
	body := map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "question"},
			map[string]interface{}{
				"role": "assistant",
				"tool_calls": []interface{}{map[string]interface{}{
					"id":       "call-1",
					"function": map[string]interface{}{"name": "lookup", "arguments": "{}"},
				}},
			},
			map[string]interface{}{
				"role":         "tool",
				"tool_call_id": "call-1",
				"content":      "result",
			},
		},
	}
	ir := ParseRequestIR(body, Provenance{})
	if !ir.Messages[1].Protected || !ir.Messages[2].Protected {
		t.Fatalf("tool exchange was not protected atomically: %#v", ir.Messages)
	}
}

func TestServiceRecoverableCompressionStoresOriginal(t *testing.T) {
	original := strings.Repeat("large recoverable context ", 300)
	body := map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "question"},
			map[string]interface{}{"role": "tool", "tool_call_id": "call", "content": original},
		},
	}
	store := &testRecoveryStore{}
	result := NewService().Apply(context.Background(), Request{
		Scope:    "trusted-scope",
		Request:  ParseRequestIR(body, Provenance{}),
		Recovery: store,
		Policy: Policy{
			Mode: ModeAlways,
			Targets: Targets{
				ToolOutputs: TargetPolicy{
					Mode:         TargetRecoverable,
					MinTokens:    10,
					TargetTokens: 30,
				},
			},
			Recovery: RecoveryPolicy{
				Enabled:            true,
				TTL:                time.Minute,
				MaxBytesPerRequest: len(original) + 1,
			},
		},
		TokenCounter: HeuristicTokenCounter{},
	})
	if !result.Applied || len(result.RecoveryKeys) != 1 {
		t.Fatalf("recoverable Apply() = %#v", result)
	}
	key := result.RecoveryKeys[0]
	entry, err := store.Get(context.Background(), "trusted-scope", key)
	if err != nil || entry.Content != original {
		t.Fatalf("recovery entry = %#v, %v", entry, err)
	}
	encoded, _ := json.Marshal(body)
	if !strings.Contains(string(encoded), RetrieveToolName) {
		t.Fatal("recovery marker was not injected")
	}
}

func TestServiceRecoverableCompressionPreservesJSON(t *testing.T) {
	original := `{"status":"failed","logs":"` +
		strings.Repeat("irrelevant logs ", 400) +
		`","code":500}`
	body := map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "find failed status"},
			map[string]interface{}{"role": "tool", "tool_call_id": "call", "content": original},
		},
	}
	result := NewService().Apply(context.Background(), Request{
		Scope:    "trusted-scope",
		Request:  ParseRequestIR(body, Provenance{}),
		Recovery: &testRecoveryStore{},
		Policy: Policy{
			Mode: ModeAlways,
			Targets: Targets{
				ToolOutputs: TargetPolicy{
					Mode:         TargetRecoverable,
					MinTokens:    10,
					TargetTokens: 40,
				},
			},
			Recovery: RecoveryPolicy{
				Enabled:            true,
				MaxBytesPerRequest: len(original) + 1,
			},
		},
		TokenCounter: HeuristicTokenCounter{},
	})
	if !result.Applied || len(result.RecoveryKeys) != 1 {
		t.Fatalf("recoverable JSON result = %#v", result)
	}
	content := body["messages"].([]interface{})[1].(map[string]interface{})["content"].(string)
	if !json.Valid([]byte(content)) {
		t.Fatalf("recoverable compression corrupted JSON: %s", content)
	}
}

func TestServiceDerivesBudgetFromSelectedModelCapabilities(t *testing.T) {
	body := map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "question"},
			map[string]interface{}{
				"role":    "tool",
				"content": strings.Repeat("large tool output ", 1000),
			},
		},
	}
	result := NewService().Apply(context.Background(), Request{
		Model:   "selected-model",
		Request: ParseRequestIR(body, Provenance{}),
		Policy: Policy{
			Mode: ModeAuto,
			Budget: Budget{
				TriggerAuto: true,
				TargetAuto:  true,
				ReserveAuto: true,
			},
			Targets: Targets{
				ToolOutputs: TargetPolicy{
					Mode:         TargetExtractive,
					MinTokens:    100,
					TargetTokens: 200,
				},
			},
		},
		Capabilities: ModelContextCapabilities{
			ContextWindow:   2048,
			RequestedOutput: 512,
		},
		TokenCounter: HeuristicTokenCounter{},
	})
	if result.Plan.AvailableInputTokens != 1536 ||
		result.Plan.ReservedOutputTokens != 512 ||
		result.Plan.TargetRequestTokens > 1536 {
		t.Fatalf("model-aware plan = %#v", result.Plan)
	}
	if !result.Applied {
		t.Fatal("model-aware budget did not trigger compression")
	}
}

func TestServiceReportsRecoveryBackendFailure(t *testing.T) {
	body := map[string]interface{}{
		"messages": []interface{}{
			map[string]interface{}{"role": "user", "content": "question"},
			map[string]interface{}{
				"role":    "tool",
				"content": strings.Repeat("recoverable output ", 200),
			},
		},
	}
	result := NewService().Apply(context.Background(), Request{
		Scope:    "scope",
		Request:  ParseRequestIR(body, Provenance{}),
		Recovery: &testRecoveryStore{err: errors.New("backend unavailable")},
		Policy: Policy{
			Mode: ModeAlways,
			Targets: Targets{
				ToolOutputs: TargetPolicy{
					Mode:         TargetRecoverable,
					MinTokens:    10,
					TargetTokens: 20,
				},
			},
			Recovery: RecoveryPolicy{
				Enabled:            true,
				MaxBytesPerRequest: 1 << 20,
			},
			FailureMode: FailureClosed,
		},
		TokenCounter: HeuristicTokenCounter{},
	})
	if result.Failure == nil || result.Applied {
		t.Fatalf("backend failure result = %#v", result)
	}
}
