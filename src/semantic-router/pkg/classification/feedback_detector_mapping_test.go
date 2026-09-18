package classification

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestFeedbackDetectorLoadsMappingFromPath(t *testing.T) {
	path := filepath.Join(t.TempDir(), "label_mapping.json")
	content := `{"label_to_idx": {"NEED_CLARIFICATION": 0, "SAT": 1, "WANT_DIFFERENT": 2, "WRONG_ANSWER": 3}, "idx_to_label": {"0": "NEED_CLARIFICATION", "1": "SAT", "2": "WANT_DIFFERENT", "3": "WRONG_ANSWER"}}`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	detector := &FeedbackDetector{config: &config.FeedbackDetectorConfig{FeedbackMappingPath: path}}
	if err := detector.loadMapping(path); err != nil {
		t.Fatalf("loadMapping() error = %v", err)
	}
	if got := detector.mapping.IdxToLabel["1"]; got != FeedbackLabelSatisfied {
		t.Fatalf("IdxToLabel[1] = %q, want %q", got, FeedbackLabelSatisfied)
	}
	if got := detector.mapping.LabelToIdx[FeedbackLabelWrongAnswer]; got != 3 {
		t.Fatalf("LabelToIdx[wrong_answer] = %d, want 3", got)
	}

	configPath := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(configPath, []byte(`{"id2label": {"0": "SAT"}, "label2id": {"SAT": 0}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := detector.loadMapping(configPath); err != nil {
		t.Fatalf("loadMapping(config.json) error = %v", err)
	}
	if got := detector.mapping.IdxToLabel["0"]; got != FeedbackLabelSatisfied {
		t.Fatalf("IdxToLabel[0] = %q, want %q", got, FeedbackLabelSatisfied)
	}

	if err := detector.loadMapping(filepath.Join(t.TempDir(), "missing.json")); err == nil {
		t.Fatal("expected an error for a missing mapping file")
	}
}
