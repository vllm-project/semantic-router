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

func TestRecorderMaxToolTraceBytesPromptTruncation(t *testing.T) {
	recorder := NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetCapturePolicy(false, false, 4096)
	recorder.SetMaxToolTraceBytes(15)

	id, err := recorder.AddRecord(RoutingRecord{
		RequestID:       "req-trunc-prompt",
		Prompt:          "This prompt exceeds fifteen bytes",
		ToolDefinitions: `[{"type":"function"}]`,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	rec, found := recorder.GetRecord(id)
	if !found {
		t.Fatal("record not found")
	}

	if len(rec.Prompt) != 15 {
		t.Errorf("expected Prompt len=15, got %d", len(rec.Prompt))
	}
	if !rec.PromptTruncated {
		t.Error("expected PromptTruncated=true")
	}
	// ToolDefinitions is shorter than 15 bytes only if json is short enough;
	// here `[{"type":"function"}]` is 22 bytes so it should be truncated too.
	if len(rec.ToolDefinitions) != 15 {
		t.Errorf("expected ToolDefinitions len=15, got %d", len(rec.ToolDefinitions))
	}
}

func TestRecorderMaxToolTraceBytesStepTruncation(t *testing.T) {
	recorder := NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetCapturePolicy(false, false, 4096)
	recorder.SetMaxToolTraceBytes(8)

	trace := &ToolTrace{
		Steps: []ToolTraceStep{
			{Type: "assistant_tool_call", Arguments: `{"city":"New York"}`, Output: ""},
			{Type: "client_tool_result", Arguments: "", Output: `{"temp":"22C"}`},
		},
	}
	id, err := recorder.AddRecord(RoutingRecord{
		RequestID: "req-trunc-steps",
		ToolTrace: trace,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	rec, found := recorder.GetRecord(id)
	if !found {
		t.Fatal("record not found")
	}

	if len(rec.ToolTrace.Steps[0].Arguments) != 8 {
		t.Errorf("expected Arguments len=8, got %d", len(rec.ToolTrace.Steps[0].Arguments))
	}
	if !rec.ToolTrace.Steps[0].Truncated {
		t.Error("expected step[0].Truncated=true")
	}
	if len(rec.ToolTrace.Steps[1].Output) != 8 {
		t.Errorf("expected Output len=8, got %d", len(rec.ToolTrace.Steps[1].Output))
	}
	if !rec.ToolTrace.Steps[1].Truncated {
		t.Error("expected step[1].Truncated=true")
	}
}

func TestRecorderMaxToolTraceBytesPreservesRawFields(t *testing.T) {
	recorder := NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetCapturePolicy(false, false, 4096)
	recorder.SetMaxToolTraceBytes(5)

	fullArgs := `{"city":"San Francisco"}`
	fullOutput := `{"temp":"22C","condition":"sunny"}`
	trace := &ToolTrace{
		Steps: []ToolTraceStep{
			{
				Type:         "assistant_tool_call",
				Arguments:    fullArgs,
				RawArguments: fullArgs,
			},
			{
				Type:      "client_tool_result",
				Output:    fullOutput,
				RawOutput: fullOutput,
			},
		},
	}
	id, err := recorder.AddRecord(RoutingRecord{
		RequestID: "req-raw-fidelity",
		ToolTrace: trace,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	rec, found := recorder.GetRecord(id)
	if !found {
		t.Fatal("record not found")
	}

	if got := rec.ToolTrace.Steps[0].Arguments; len(got) != 5 {
		t.Errorf("expected Arguments truncated to len=5, got len=%d (%q)", len(got), got)
	}
	if got := rec.ToolTrace.Steps[0].RawArguments; got != fullArgs {
		t.Errorf("expected RawArguments preserved at full fidelity; got %q, want %q", got, fullArgs)
	}
	if got := rec.ToolTrace.Steps[1].Output; len(got) != 5 {
		t.Errorf("expected Output truncated to len=5, got len=%d (%q)", len(got), got)
	}
	if got := rec.ToolTrace.Steps[1].RawOutput; got != fullOutput {
		t.Errorf("expected RawOutput preserved at full fidelity; got %q, want %q", got, fullOutput)
	}
}

func TestRecorderMaxToolTraceBytesFlagsToolDefinitionsTruncation(t *testing.T) {
	recorder := NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetCapturePolicy(false, false, 4096)
	recorder.SetMaxToolTraceBytes(10)

	longToolDefs := `[{"type":"function","function":{"name":"really_long_tool_name","description":"truncate me"}}]`
	id, err := recorder.AddRecord(RoutingRecord{
		RequestID:       "req-tooldefs-truncated",
		ToolDefinitions: longToolDefs,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	rec, found := recorder.GetRecord(id)
	if !found {
		t.Fatal("record not found")
	}

	if len(rec.ToolDefinitions) != 10 {
		t.Errorf("expected ToolDefinitions len=10, got %d", len(rec.ToolDefinitions))
	}
	if !rec.ToolDefinitionsTruncated {
		t.Error("expected ToolDefinitionsTruncated=true after truncation")
	}
}

func TestRecorderUpdateToolTraceAppliesMaxToolTraceBytes(t *testing.T) {
	recorder := NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetCapturePolicy(false, false, 4096)
	recorder.SetMaxToolTraceBytes(6)

	id, err := recorder.AddRecord(RoutingRecord{
		ID:        "replay-update-trunc",
		RequestID: "req-update-trunc",
	})
	if err != nil {
		t.Fatalf("failed to add record: %v", err)
	}

	// UpdateToolTrace is the path response-side traces take
	// (see extproc.attachRouterReplayResponse). This exchange must also be
	// capped by MaxToolTraceBytes or responses could silently blow past it.
	fullOutput := `{"temperature":"22C","condition":"sunny"}`
	trace := ToolTrace{
		Steps: []ToolTraceStep{
			{
				Type:      "client_tool_result",
				Output:    fullOutput,
				RawOutput: fullOutput,
			},
		},
	}
	if err := recorder.UpdateToolTrace(id, trace); err != nil {
		t.Fatalf("UpdateToolTrace: %v", err)
	}

	rec, found := recorder.GetRecord(id)
	if !found || rec.ToolTrace == nil || len(rec.ToolTrace.Steps) != 1 {
		t.Fatalf("unexpected stored record: found=%v, trace=%#v", found, rec.ToolTrace)
	}
	step := rec.ToolTrace.Steps[0]
	if len(step.Output) != 6 {
		t.Errorf("expected Output truncated to 6 bytes through UpdateToolTrace, got len=%d (%q)", len(step.Output), step.Output)
	}
	if !step.Truncated {
		t.Error("expected Truncated=true after UpdateToolTrace truncation")
	}
	if step.RawOutput != fullOutput {
		t.Errorf("expected RawOutput preserved on UpdateToolTrace; got %q, want %q", step.RawOutput, fullOutput)
	}
}

func makeStepsWithToolNames(names []string) []ToolTraceStep {
	steps := make([]ToolTraceStep, len(names))
	for i, n := range names {
		steps[i] = ToolTraceStep{Type: "assistant_tool_call", ToolName: n}
	}
	return steps
}

func TestRecorderMaxToolTraceStepsDropsOldestOnAddRecord(t *testing.T) {
	// Regression test for #1835: an unbounded agent session that produced
	// hundreds of tool-call steps per record was OOMing the router. With
	// MaxToolTraceSteps set, AddRecord must drop the oldest steps so memory
	// stays bounded while the most recent timeline is preserved.
	recorder := NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetCapturePolicy(false, false, 4096)
	recorder.SetMaxToolTraceSteps(3)

	steps := makeStepsWithToolNames([]string{"a", "b", "c", "d", "e"})
	id, err := recorder.AddRecord(RoutingRecord{
		ID:        "replay-step-cap",
		RequestID: "req-step-cap",
		ToolTrace: &ToolTrace{Steps: steps},
	})
	if err != nil {
		t.Fatalf("AddRecord: %v", err)
	}

	rec, found := recorder.GetRecord(id)
	if !found || rec.ToolTrace == nil {
		t.Fatalf("unexpected stored record: found=%v, trace=%#v", found, rec.ToolTrace)
	}
	if got := len(rec.ToolTrace.Steps); got != 3 {
		t.Fatalf("expected 3 retained steps, got %d", got)
	}
	wantNames := []string{"c", "d", "e"}
	for i, name := range wantNames {
		if rec.ToolTrace.Steps[i].ToolName != name {
			t.Errorf("step[%d]: expected ToolName=%q, got %q", i, name, rec.ToolTrace.Steps[i].ToolName)
		}
	}
	if !rec.ToolTrace.StepsTruncated {
		t.Error("expected StepsTruncated=true after step-count cap")
	}
	if rec.ToolTrace.DroppedStepCount != 2 {
		t.Errorf("expected DroppedStepCount=2, got %d", rec.ToolTrace.DroppedStepCount)
	}
}

func TestRecorderMaxToolTraceStepsZeroDisablesCap(t *testing.T) {
	recorder := NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetCapturePolicy(false, false, 4096)
	// 0 = no cap. The default for NewRecorder is also 0 to preserve
	// backwards-compatible behaviour for external callers; the
	// plugin-level config supplies a sane non-zero default.
	recorder.SetMaxToolTraceSteps(0)

	steps := makeStepsWithToolNames([]string{"a", "b", "c", "d", "e"})
	id, err := recorder.AddRecord(RoutingRecord{
		ID:        "replay-step-no-cap",
		RequestID: "req-step-no-cap",
		ToolTrace: &ToolTrace{Steps: steps},
	})
	if err != nil {
		t.Fatalf("AddRecord: %v", err)
	}

	rec, _ := recorder.GetRecord(id)
	if rec.ToolTrace == nil || len(rec.ToolTrace.Steps) != 5 {
		t.Fatalf("expected all 5 steps retained without a cap, got %#v", rec.ToolTrace)
	}
	if rec.ToolTrace.StepsTruncated {
		t.Error("expected StepsTruncated=false when cap is 0")
	}
}

func TestRecorderUpdateToolTraceAppliesMaxToolTraceSteps(t *testing.T) {
	// Response-side tool traces flow through UpdateToolTrace
	// (see extproc.attachRouterReplayResponse). They must also respect the
	// step cap or a long streaming response could re-introduce the OOM.
	recorder := NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetCapturePolicy(false, false, 4096)
	recorder.SetMaxToolTraceSteps(2)

	id, err := recorder.AddRecord(RoutingRecord{
		ID:        "replay-update-step-cap",
		RequestID: "req-update-step-cap",
	})
	if err != nil {
		t.Fatalf("AddRecord: %v", err)
	}

	steps := makeStepsWithToolNames([]string{"a", "b", "c", "d"})
	if err := recorder.UpdateToolTrace(id, ToolTrace{Steps: steps}); err != nil {
		t.Fatalf("UpdateToolTrace: %v", err)
	}

	rec, _ := recorder.GetRecord(id)
	if rec.ToolTrace == nil || len(rec.ToolTrace.Steps) != 2 {
		t.Fatalf("expected step cap enforced on UpdateToolTrace, got %#v", rec.ToolTrace)
	}
	if rec.ToolTrace.Steps[0].ToolName != "c" || rec.ToolTrace.Steps[1].ToolName != "d" {
		t.Errorf("expected newest steps preserved (c,d), got %+v", rec.ToolTrace.Steps)
	}
	if !rec.ToolTrace.StepsTruncated || rec.ToolTrace.DroppedStepCount != 2 {
		t.Errorf("expected StepsTruncated=true, DroppedStepCount=2; got %v / %d",
			rec.ToolTrace.StepsTruncated, rec.ToolTrace.DroppedStepCount)
	}
}

func TestRecorderSetMaxToolTraceBytesZeroNoTruncation(t *testing.T) {
	recorder := NewRecorder(store.NewMemoryStore(10, 0))
	recorder.SetCapturePolicy(false, false, 4096)
	recorder.SetMaxToolTraceBytes(0) // no limit

	longPrompt := "This is a very long prompt that would be truncated if the limit were active."
	id, err := recorder.AddRecord(RoutingRecord{
		RequestID: "req-no-trunc",
		Prompt:    longPrompt,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	rec, found := recorder.GetRecord(id)
	if !found {
		t.Fatal("record not found")
	}

	if rec.Prompt != longPrompt {
		t.Errorf("expected full prompt, got %q", rec.Prompt)
	}
	if rec.PromptTruncated {
		t.Error("expected PromptTruncated=false")
	}
}
