package classification

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// The same golden fixtures run against a task-declared label set with no PII
// mapping or reserved class index. A second consumer can reuse the transport and
// alignment contract while retaining its own task-specific input adapter.
func TestHTTPTokenClassifierGenericTaskFixtures(t *testing.T) {
	labels := tasks.TokenLabelSet{
		Labels:  []string{"PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "URL", "ADDRESS", "CREDIT_CARD"},
		Outside: []string{"O"},
	}
	for _, tc := range loadTokenSpansFixtures(t) {
		t.Run(tc.Name, func(t *testing.T) {
			_, cfg := newTokenSpansServer(t, func(inputs string) any {
				if inputs != tc.Text {
					t.Error("task input changed in transport")
				}
				return wireSpans(tc.Spans)
			})
			backend, err := newHTTPTokenClassifierInference(cfg, labels, 0)
			if err != nil {
				t.Fatal(err)
			}
			defer backend.Close()
			result, err := backend.ClassifyTokens(context.Background(), tc.Text)
			if tc.Expect == "reject" {
				assertFixtureRejected(t, tc, result.Entities, err)
			} else {
				assertFixtureAccepted(t, tc, result.Entities, err)
				if !result.HasScores() {
					t.Fatal("scored wire response lost its score availability")
				}
			}
		})
	}
}

func TestHTTPTokenClassifierGenericLabelZeroAndPartialMetadata(t *testing.T) {
	text := "前🙂 unsupported trailing"
	_, cfg := newTokenSpansServer(t, func(string) any {
		return map[string]any{
			"truncated_at": 14,
			"spans":        []map[string]any{{"label": "UNSUPPORTED", "score": 0.75, "text": "unsupported", "start": 3, "end": 14}},
		}
	})
	labels := tasks.TokenLabelSet{Labels: []string{"UNSUPPORTED"}}
	backend, err := newHTTPTokenClassifierInference(cfg, labels, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer backend.Close()
	labels.Labels[0] = "mutated after preparation"
	result, err := backend.ClassifyTokens(context.Background(), text)
	if !errors.Is(err, tasks.ErrTokenSpansTruncated) || len(result.Entities) != 1 || !result.HasScores() {
		t.Fatalf("first task label was treated as outside or partial span lost: %+v, %v", result, err)
	}
	span := result.Entities[0]
	if span.EntityType != "UNSUPPORTED" || span.Start != len("前🙂 ") || text[span.Start:span.End] != "unsupported" || span.Confidence != 0.75 {
		t.Fatalf("generic task changed byte/score semantics: %+v", span)
	}
	if result.TruncatedAt == nil || *result.TruncatedAt != len("前🙂 unsupported") {
		t.Fatalf("truncation must retain a byte offset, got %v", result.TruncatedAt)
	}
}

func TestHTTPTokenClassifierRejectsInvalidTaskLabels(t *testing.T) {
	for _, labels := range []tasks.TokenLabelSet{
		{},
		{Labels: []string{""}},
		{Labels: []string{"B-UNSUPPORTED"}, Outside: []string{"UNSUPPORTED"}},
		{Labels: []string{"UNSUPPORTED"}, Outside: []string{""}},
	} {
		if _, _, err := compileTokenLabels(labels); err == nil {
			t.Errorf("invalid task label declaration accepted: %+v", labels)
		}
	}
}

func TestHTTPTokenClassifierRejectsNonFiniteSpanConfidence(t *testing.T) {
	for _, score := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		if _, err := spanScore(0, "UNSUPPORTED", tokenSpanWire{Score: &score}); err == nil {
			t.Errorf("non-finite confidence accepted: %v", score)
		}
	}
}
