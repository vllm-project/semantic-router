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
