//go:build !windows && cgo && (amd64 || arm64)

package instance

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func configureSequenceFixture(t *testing.T, options Options, changes map[string]any) {
	t.Helper()
	path := filepath.Join(options.ModelPath, "config.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var config map[string]any
	if err = json.Unmarshal(data, &config); err != nil {
		t.Fatal(err)
	}
	for key, value := range changes {
		config[key] = value
	}
	data, err = json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestOwnedSequenceCheckpointCapacityAndFullTokenBudget(t *testing.T) {
	options := specialTokenFixture(t, "sequence")
	configureSequenceFixture(t, options, map[string]any{"max_position_embeddings": 32768})
	options.MaxInputTokens = 0
	legacy, err := LoadSequenceClassifier(options)
	if err != nil {
		t.Fatal(err)
	}
	defer legacy.Close()
	if _, err = legacy.Classify(strings.Repeat("hello ", 511)); err == nil {
		t.Fatal("omitted budget no longer preserves 512")
	}
	options.MaxInputTokens = 32768
	model, err := LoadSequenceClassifier(options)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	text := strings.Repeat("hello ", 32766)
	result, err := model.Classify(text)
	if err != nil {
		t.Fatal(err)
	}
	if result.Input.OriginalTokens != 32768 || result.Input.ProcessedTokens != 32768 || result.Input.Truncated {
		t.Fatalf("incorrect exact context: %+v", result.Input)
	}
	if _, err = model.Classify(text + "hello"); err == nil {
		t.Fatal("32769 tokens were accepted")
	} else {
		errorKind(t, err, "input_limit")
	}
	options.MaxInputTokens = 32769
	if invalid, loadErr := LoadSequenceClassifier(options); loadErr == nil {
		_ = invalid.Close()
		t.Fatal("budget exceeds real capacity")
	}
}

func TestOwnedIndependentScoresAndWindowCoverage(t *testing.T) {
	options := specialTokenFixture(t, "sequence")
	options.MaxInputTokens = 512
	configureSequenceFixture(t, options, map[string]any{"problem_type": "multi_label_classification"})
	if invalid, loadErr := LoadSequenceClassifier(options); loadErr == nil {
		_ = invalid.Close()
		t.Fatal("multi-label head loaded as softmax")
	}
	model, err := LoadLabelScorer(options)
	if err != nil {
		t.Fatal(err)
	}
	shared, err := model.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer shared.Close()
	if err = model.Close(); err != nil {
		t.Fatal(err)
	}
	text := "hello world test hello world"
	scores, err := shared.Score(text)
	if err != nil {
		t.Fatal(err)
	}
	if len(scores.Scores) != 2 || len(scores.Labels) != 2 || scores.Input.ProcessedTokens != 7 {
		t.Fatalf("invalid full output: %+v", scores)
	}
	// Fixture logits are [-mean(ids), mean(ids)]; assert actual sigmoid values.
	want := 1 / (1 + math.Exp(-(0.0+1+2+4+1+2+4)/7))
	if math.Abs(float64(scores.Scores[1])-want) > 1e-6 {
		t.Fatalf("sigmoid mismatch: %+v", scores)
	}
	windows, err := shared.ScoreWindows(text, SequenceWindowOptions{Size: 5, Overlap: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(windows.Windows) != 2 || windows.ContentTokens != 5 || windows.Input.ProcessedTokens != 7 || windows.Input.Truncated {
		t.Fatalf("invalid window usage: %+v", windows)
	}
	if windows.Windows[0].Start != 0 || windows.Windows[0].End != 3 || windows.Windows[1].Start != 2 || windows.Windows[1].End != 5 {
		t.Fatalf("window coverage: %+v", windows.Windows)
	}
	info, err := shared.Info()
	if err != nil {
		t.Fatal(err)
	}
	if info.CompletedInferences != 3 {
		t.Fatalf("expected full score plus two forwards: %+v", info)
	}
	if _, err = shared.ScoreWindows(text, SequenceWindowOptions{Size: 2}); err == nil {
		t.Fatal("special-token-only window accepted")
	}
	if _, err = shared.ScoreWindows(strings.Repeat("hello ", 511), SequenceWindowOptions{Size: 5}); err == nil {
		t.Fatal("full request overflow accepted")
	}
}
