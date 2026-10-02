//go:build !windows && cgo && (amd64 || arm64)

package instance

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func groundingFixture(t *testing.T) Options {
	t.Helper()
	options := specialTokenFixture(t, "token")
	options.MaxInputTokens = 32
	configureSequenceFixture(t, options, map[string]any{"max_position_embeddings": 32768, "id2label": map[string]string{"0": "supported", "1": "hallucinated"}, "label2id": map[string]int{"supported": 0, "hallucinated": 1}})
	policy := map[string]any{"max_input_tokens": 8192, "token_threshold": .5, "threshold_comparison": "strictly_greater", "label2id": map[string]int{"supported": 0, "hallucinated": 1}, "input_pair": []string{"User request: {question}\n\n{context}", "answer"}, "answer_offsets": "Unicode code points"}
	data, err := json.Marshal(policy)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(options.ModelPath, "operating_point.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	return options
}

func TestOwnedGroundingPairPolicyAndUnicode(t *testing.T) {
	options := groundingFixture(t)
	model, err := LoadGroundedClassifier(options)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	// The graph emits class-one probability >.5 for known words and exactly
	// .5 for unknown words, proving strict comparison and answer-only spans.
	answer := "é 秘密 test"
	result, err := model.Detect("hello world", "test", answer)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Spans) != 1 || result.Spans[0].Text != "秘密 test" || result.Spans[0].Start != 3 || result.Spans[0].End != len(answer) {
		t.Fatalf("invalid answer byte spans: %+v", result)
	}
	if result.OffsetUnit != "utf8_bytes" || result.Input.Truncated {
		t.Fatalf("invalid grounding metadata: %+v", result)
	}
	info, err := model.Info()
	if err != nil {
		t.Fatal(err)
	}
	if info.TaskLimit != 8192 || info.ModelLimit != 32768 {
		t.Fatalf("task capacity inherited architecture: %+v", info)
	}
	if _, err = model.Detect(strings.Repeat("hello ", 40), "", answer); err == nil {
		t.Fatal("oversize pair accepted")
	} else {
		errorKind(t, err, "input_limit")
	}
	options.Overflow = "truncate_right"
	truncated, err := LoadGroundedClassifier(options)
	if err != nil {
		t.Fatal(err)
	}
	defer truncated.Close()
	result, err = truncated.Detect(strings.Repeat("hello ", 40), "", answer)
	if err != nil || !result.Input.Truncated || len(result.Spans) != 1 || result.Spans[0].Text != "秘密 test" {
		t.Fatalf("only-first truncation changed answer: %+v / %v", result, err)
	}
	if _, err = truncated.Detect("hello", "", strings.Repeat("world ", 40)); err == nil {
		t.Fatal("answer was silently truncated")
	}
	options.MaxInputTokens = 8193
	if invalid, err := LoadGroundedClassifier(options); err == nil {
		_ = invalid.Close()
		t.Fatal("Halu admitted >8192 task budget")
	}
}
