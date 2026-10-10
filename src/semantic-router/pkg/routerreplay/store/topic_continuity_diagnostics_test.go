package store

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/topiccontinuity"
)

func topicText(role llmprotocol.Role, value string) llmprotocol.Message {
	return llmprotocol.Message{Role: role, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: value}}}
}

// topicContinuityEvaluations runs the real evaluator: one explicit change and,
// with a one-turn window over two prior turns, one beyond-window unknown.
func topicContinuityEvaluations() []topiccontinuity.RuleEvaluation {
	messages := []llmprotocol.Message{
		topicText(llmprotocol.RoleUser, "Refactor the routing module so plugins load lazily"),
		topicText(llmprotocol.RoleAssistant, "Done. The loader now defers plugin initialization."),
		topicText(llmprotocol.RoleUser, "Add tests for the lazy loader"),
		{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
			Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "c1", Name: "run_tests", Arguments: "{}"},
		}}},
		{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{
			Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: "c1"},
		}}},
		topicText(llmprotocol.RoleAssistant, "Tests pass."),
		topicText(llmprotocol.RoleUser, "Unrelated question: how do I renew a passport?"),
	}
	wide := topiccontinuity.HistoryPolicy{
		Limits:           topiccontinuity.Limits{MaxPriorTurns: 8, MaxTurnBytes: 16384, MaxInputBytes: 147456},
		IncludeAssistant: true,
	}
	narrow := wide
	narrow.Limits.MaxPriorTurns = 1
	rules := []topiccontinuity.EvalConfig{
		{Name: "z_rule", Policy: wide, Continuation: 0.35, Change: 0.08},
		{Name: "a_rule", Policy: narrow, Continuation: 0.35, Change: 0.08},
	}
	load := func() ([]llmprotocol.Message, bool) { return messages, true }
	return topiccontinuity.EvaluateAll(context.Background(), load, rules).Rules
}

func TestTopicContinuityRecordsKeepDeclarationOrderAndFields(t *testing.T) {
	records := NewTopicContinuityRecords(topicContinuityEvaluations())
	if len(records) != 2 || records[0].Signal != "z_rule" || records[1].Signal != "a_rule" {
		t.Fatalf("records = %+v", records)
	}
	evaluations := topicContinuityEvaluations()
	first, result := records[0], evaluations[0].Result
	if first.Class != "change" || first.Reason != "change_explicit_marker" || first.Coverage != "full" ||
		first.HistorySource != "original_snapshot" || !first.ExcludedContentPresent ||
		first.InputBytes != result.Features.InputBytes || first.PriorTurnsExamined != 2 ||
		first.Confidence != result.Confidence || first.LiveTerms != result.Features.LiveTerms {
		t.Fatalf("first record = %+v", first)
	}
	if second := records[1]; second.Reason != "unknown_history_beyond_window" || second.Coverage != "window" {
		t.Fatalf("second record = %+v", second)
	}
	if NewTopicContinuityRecords(nil) != nil {
		t.Fatal("no evaluations should produce no records")
	}
}

func TestTopicContinuityRecordJSONRoundTripAndFieldNames(t *testing.T) {
	diagnostics := &RouteDiagnostics{TopicContinuity: NewTopicContinuityRecords(topicContinuityEvaluations())}
	encoded, err := json.Marshal(diagnostics)
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{`"topic_continuity"`, `"classify_us"`, `"prepare_extract_us"`, `"max_raw_score"`} {
		if !strings.Contains(string(encoded), field) {
			t.Fatalf("encoded diagnostics lack %s: %s", field, encoded)
		}
	}
	var decoded RouteDiagnostics
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decoded.TopicContinuity, diagnostics.TopicContinuity) {
		t.Fatal("round trip changed the records")
	}
}

func TestTopicContinuityRecordsAreContentFree(t *testing.T) {
	// Every string field is a bounded enum, version, or rule name: no field may
	// carry conversation text or a digest.
	allowed := map[string]bool{
		"Signal": true, "SchemaVersion": true, "EvaluatorVersion": true, "Class": true,
		"Reason": true, "Coverage": true, "HistorySource": true,
	}
	recordType := reflect.TypeOf(TopicContinuityRecord{})
	for i := 0; i < recordType.NumField(); i++ {
		field := recordType.Field(i)
		if field.Type.Kind() == reflect.String && !allowed[field.Name] {
			t.Fatalf("unexpected string field %s could carry content", field.Name)
		}
	}
}

func TestTopicContinuityCloneIsolationAndOldRecords(t *testing.T) {
	original := &RouteDiagnostics{TopicContinuity: NewTopicContinuityRecords(topicContinuityEvaluations())}
	cloned := cloneRouteDiagnostics(original)
	cloned.TopicContinuity[0].Reason = "mutated"
	if original.TopicContinuity[0].Reason == "mutated" {
		t.Fatal("mutating the clone changed the original")
	}

	var old RouteDiagnostics
	if err := json.Unmarshal([]byte(`{"decision":"route_a"}`), &old); err != nil {
		t.Fatal(err)
	}
	if old.TopicContinuity != nil {
		t.Fatalf("an old record decoded topic continuity: %+v", old.TopicContinuity)
	}
	encoded, err := json.Marshal(old)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), "topic_continuity") {
		t.Fatalf("an old record gained a topic_continuity field: %s", encoded)
	}
}
