package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestStructureTextBytesUsesUncompressedText(t *testing.T) {
	threshold := 5.0
	classifier, err := NewStructureClassifier([]config.StructureRule{{
		Name: "long-input",
		Feature: config.StructureFeature{
			Type: "count",
			Source: config.StructureSource{
				Type: "text_bytes",
			},
		},
		Predicate: &config.NumericPredicate{GTE: &threshold},
	}})
	if err != nil {
		t.Fatalf("NewStructureClassifier() error = %v", err)
	}

	matches, err := classifier.Classify("tiny", "你好世界")
	if err != nil {
		t.Fatalf("Classify() error = %v", err)
	}
	if len(matches) != 1 {
		t.Fatalf("matches = %v, want one match", matches)
	}
	if matches[0].Value != float64(len("你好世界")) {
		t.Fatalf("byte value = %v, want %d", matches[0].Value, len("你好世界"))
	}
}

func TestStructureTextBytesCountsWhitespaceOnlyText(t *testing.T) {
	threshold := 4.0
	classifier, err := NewStructureClassifier([]config.StructureRule{{
		Name: "whitespace-input",
		Feature: config.StructureFeature{
			Type:   "count",
			Source: config.StructureSource{Type: "text_bytes"},
		},
		Predicate: &config.NumericPredicate{GTE: &threshold},
	}})
	if err != nil {
		t.Fatalf("NewStructureClassifier() error = %v", err)
	}

	matches, err := classifier.Classify("", " \t \n")
	if err != nil {
		t.Fatalf("Classify() error = %v", err)
	}
	if len(matches) != 1 || matches[0].Value != 4 {
		t.Fatalf("matches = %#v, want four raw bytes", matches)
	}
}

func TestStructureEvaluateAllPublishesZeroForEmptyText(t *testing.T) {
	maxBytes := 0.0
	classifier, err := NewStructureClassifier([]config.StructureRule{{
		Name: "empty-input",
		Feature: config.StructureFeature{
			Type:   "count",
			Source: config.StructureSource{Type: "text_bytes"},
		},
		Predicate: &config.NumericPredicate{LTE: &maxBytes},
	}})
	if err != nil {
		t.Fatalf("NewStructureClassifier() error = %v", err)
	}

	results, err := classifier.EvaluateAll("")
	if err != nil {
		t.Fatalf("EvaluateAll() error = %v", err)
	}
	if len(results) != 1 || results[0].Value != 0 || !results[0].Matched {
		t.Fatalf("results = %#v, want matched zero-byte value", results)
	}
}
