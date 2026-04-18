package routerreplay

import (
	"reflect"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

func TestRecorderUpdateUsageCostClonesStoredValues(t *testing.T) {
	recorder := NewRecorder(store.NewMemoryStore(10, 0))
	recordID, err := recorder.AddRecord(RoutingRecord{
		ID:        "replay-usage-1",
		Decision:  "decision-a",
		RequestID: "req-1",
	})
	if err != nil {
		t.Fatalf("failed to add record: %v", err)
	}

	promptTokens := 120
	completionTokens := 45
	totalTokens := 165
	actualCost := 0.0012
	baselineCost := 0.0033
	costSavings := 0.0021
	currency := "USD"
	baselineModel := "premium-model"
	usage := UsageCost{
		PromptTokens:     &promptTokens,
		CompletionTokens: &completionTokens,
		TotalTokens:      &totalTokens,
		ActualCost:       &actualCost,
		BaselineCost:     &baselineCost,
		CostSavings:      &costSavings,
		Currency:         &currency,
		BaselineModel:    &baselineModel,
	}

	if err := recorder.UpdateUsageCost(recordID, usage); err != nil {
		t.Fatalf("failed to update usage cost: %v", err)
	}

	promptTokens = 999
	completionTokens = 999
	totalTokens = 1998
	actualCost = 9.9
	baselineCost = 19.9
	costSavings = 10.0
	currency = "CNY"
	baselineModel = "mutated-model"

	record, found := recorder.GetRecord(recordID)
	if !found {
		t.Fatal("expected to retrieve updated replay record")
	}

	assertIntPtr(t, record.PromptTokens, 120, "prompt tokens")
	assertIntPtr(t, record.CompletionTokens, 45, "completion tokens")
	assertIntPtr(t, record.TotalTokens, 165, "total tokens")
	assertFloatPtr(t, record.ActualCost, 0.0012, "actual cost")
	assertFloatPtr(t, record.BaselineCost, 0.0033, "baseline cost")
	assertFloatPtr(t, record.CostSavings, 0.0021, "cost savings")
	assertStringPtr(t, record.Currency, "USD", "currency")
	assertStringPtr(t, record.BaselineModel, "premium-model", "baseline model")
}

func TestRecorderUpdateToolTraceClonesStoredValues(t *testing.T) {
	recorder := NewRecorder(store.NewMemoryStore(10, 0))
	recordID, err := recorder.AddRecord(RoutingRecord{
		ID:        "replay-tool-trace-1",
		Decision:  "decision-a",
		RequestID: "req-1",
	})
	if err != nil {
		t.Fatalf("failed to add record: %v", err)
	}

	trace := ToolTrace{
		Flow:      "User Query -> LLM Tool Call",
		Stage:     "LLM Tool Call",
		ToolNames: []string{"get_weather"},
		Steps: []ToolTraceStep{
			{Type: "user_input", Text: "Find the weather."},
			{Type: "assistant_tool_call", ToolName: "get_weather", Arguments: "{\"location\":\"San Francisco\"}"},
		},
	}

	if err := recorder.UpdateToolTrace(recordID, trace); err != nil {
		t.Fatalf("failed to update tool trace: %v", err)
	}

	trace.Flow = "mutated"
	trace.ToolNames[0] = "mutated_tool"
	trace.Steps[0].Text = "mutated text"

	record, found := recorder.GetRecord(recordID)
	if !found {
		t.Fatal("expected to retrieve replay record")
	}
	if record.ToolTrace == nil {
		t.Fatal("expected tool trace to be stored")
	}
	if record.ToolTrace.Flow != "User Query -> LLM Tool Call" {
		t.Fatalf("unexpected tool trace flow: %q", record.ToolTrace.Flow)
	}
	if !reflect.DeepEqual(record.ToolTrace.ToolNames, []string{"get_weather"}) {
		t.Fatalf("unexpected tool names: %#v", record.ToolTrace.ToolNames)
	}
	if got := record.ToolTrace.Steps[0].Text; got != "Find the weather." {
		t.Fatalf("unexpected cloned step text: %q", got)
	}
}

func TestLogFieldsIncludesOptionalReplayMetadata(t *testing.T) {
	promptTokens := 120
	completionTokens := 45
	totalTokens := 165
	actualCost := 0.0012
	baselineCost := 0.0033
	costSavings := 0.0021
	currency := "USD"
	baselineModel := "premium-model"
	timestamp := time.Date(2026, time.March, 31, 4, 18, 0, 0, time.UTC)

	record := richReplayRoutingRecord(
		timestamp,
		&promptTokens,
		&completionTokens,
		&totalTokens,
		&actualCost,
		&baselineCost,
		&costSavings,
		&currency,
		&baselineModel,
	)

	fields := LogFields(record, "router_replay_complete")
	assertFieldValue(t, fields, "event", "router_replay_complete")
	assertFieldValue(t, fields, "session_id", "sess-log-test")
	assertFieldValue(t, fields, "turn_index", 5)
	assertFieldValue(t, fields, "replay_id", record.ID)
	assertFieldValue(t, fields, "decision_tier", 2)
	assertFieldValue(t, fields, "decision_priority", 100)
	assertFieldValue(t, fields, "selection_method", "router_dc")
	assertFieldValue(t, fields, "guardrails_enabled", true)
	assertFieldValue(t, fields, "jailbreak_type", "prompt_injection")
	assertFieldValue(t, fields, "pii_entities", []string{"email"})
	assertFieldValue(t, fields, "rag_backend", "milvus")
	assertFieldValue(t, fields, "hallucination_spans", []string{"span-a"})
	assertFieldValue(t, fields, "prompt_tokens", promptTokens)
	assertFieldValue(t, fields, "currency", currency)
	assertFieldValue(t, fields, "tool_trace_flow", "User Query -> LLM Tool Call -> Client Tool Result -> LLM Final Response")
	assertFieldValue(t, fields, "tool_trace_stage", "LLM Final Response")
	assertFieldValue(t, fields, "tool_names", []string{"get_weather"})
	assertFieldValue(t, fields, "tool_trace_step_count", 4)
	assertSignalLogFields(t, fields)
}

func richReplayRoutingRecord(
	timestamp time.Time,
	promptTokens *int,
	completionTokens *int,
	totalTokens *int,
	actualCost *float64,
	baselineCost *float64,
	costSavings *float64,
	currency *string,
	baselineModel *string,
) RoutingRecord {
	return RoutingRecord{
		ID:                "replay-1",
		Decision:          "decision-a",
		DecisionTier:      2,
		DecisionPriority:  100,
		Category:          "math",
		OriginalModel:     "model-a",
		SelectedModel:     "model-b",
		ReasoningMode:     "cot",
		ConfidenceScore:   0.91,
		SelectionMethod:   "router_dc",
		RequestID:         "req-1",
		SessionID:         "sess-log-test",
		TurnIndex:         5,
		Timestamp:         timestamp,
		FromCache:         true,
		Streaming:         true,
		ResponseStatus:    200,
		Projections:       []string{"balance_reasoning"},
		ProjectionScores:  map[string]float64{"reasoning_pressure": 0.73},
		SignalConfidences: map[string]float64{"projection:balance_reasoning": 0.73},
		SignalValues:      map[string]float64{"reask:likely_dissatisfied": 2},
		ToolTrace: &ToolTrace{
			Flow:      "User Query -> LLM Tool Call -> Client Tool Result -> LLM Final Response",
			Stage:     "LLM Final Response",
			ToolNames: []string{"get_weather"},
			Steps: []ToolTraceStep{
				{Type: "user_input", Text: "Find the weather."},
				{Type: "assistant_tool_call", ToolName: "get_weather"},
				{Type: "client_tool_result", ToolName: "get_weather"},
				{Type: "assistant_final_response", Text: "It is sunny."},
			},
		},
		Signals: Signal{
			Keyword:    []string{"math_keywords"},
			Reask:      []string{"likely_dissatisfied"},
			Complexity: []string{"complex"},
			Modality:   []string{"AR"},
			Authz:      []string{"premium_tier"},
			Jailbreak:  []string{"prompt_attack"},
			PII:        []string{"email"},
			KB:         []string{"policy_kb"},
		},
		GuardrailsEnabled:           true,
		JailbreakEnabled:            true,
		PIIEnabled:                  true,
		JailbreakDetected:           true,
		JailbreakType:               "prompt_injection",
		JailbreakConfidence:         0.9,
		ResponseJailbreakDetected:   true,
		ResponseJailbreakType:       "response_attack",
		ResponseJailbreakConfidence: 0.8,
		PIIDetected:                 true,
		PIIEntities:                 []string{"email"},
		PIIBlocked:                  true,
		RAGEnabled:                  true,
		RAGBackend:                  "milvus",
		RAGContextLength:            2048,
		RAGSimilarityScore:          0.76,
		HallucinationEnabled:        true,
		HallucinationDetected:       true,
		HallucinationConfidence:     0.66,
		HallucinationSpans:          []string{"span-a"},
		PromptTokens:                promptTokens,
		CompletionTokens:            completionTokens,
		TotalTokens:                 totalTokens,
		ActualCost:                  actualCost,
		BaselineCost:                baselineCost,
		CostSavings:                 costSavings,
		Currency:                    currency,
		BaselineModel:               baselineModel,
	}
}

func assertSignalLogFields(t *testing.T, fields map[string]interface{}) {
	t.Helper()
	signals, ok := fields["signals"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected signals to be a map, got %T", fields["signals"])
	}
	assertFieldValue(t, signals, "keyword", []string{"math_keywords"})
	assertFieldValue(t, signals, "reask", []string{"likely_dissatisfied"})
	assertFieldValue(t, signals, "complexity", []string{"complex"})
	assertFieldValue(t, signals, "modality", []string{"AR"})
	assertFieldValue(t, signals, "authz", []string{"premium_tier"})
	assertFieldValue(t, signals, "jailbreak", []string{"prompt_attack"})
	assertFieldValue(t, signals, "pii", []string{"email"})
	assertFieldValue(t, signals, "kb", []string{"policy_kb"})
}

func assertIntPtr(t *testing.T, value *int, expected int, label string) {
	t.Helper()
	if value == nil || *value != expected {
		t.Fatalf("expected %s=%d, got %#v", label, expected, value)
	}
}

func assertFloatPtr(t *testing.T, value *float64, expected float64, label string) {
	t.Helper()
	if value == nil || *value != expected {
		t.Fatalf("expected %s=%.4f, got %#v", label, expected, value)
	}
}

func assertStringPtr(t *testing.T, value *string, expected string, label string) {
	t.Helper()
	if value == nil || *value != expected {
		t.Fatalf("expected %s=%q, got %#v", label, expected, value)
	}
}

func assertFieldValue(
	t *testing.T,
	fields map[string]interface{},
	key string,
	expected interface{},
) {
	t.Helper()
	value, ok := fields[key]
	if !ok {
		t.Fatalf("expected field %q to be present", key)
	}
	if !reflect.DeepEqual(value, expected) {
		t.Fatalf("expected field %q=%#v, got %#v", key, expected, value)
	}
}
