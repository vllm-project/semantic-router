//go:build !windows && cgo && (amd64 || arm64 || riscv64)

package candle_binding

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestOwnedIndependentLabelScoresUseNativeHandle(t *testing.T) {
	path := ownedModelFixture(t, 1)
	configPath := filepath.Join(path, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	var config map[string]any
	if err = json.Unmarshal(data, &config); err != nil {
		t.Fatal(err)
	}
	config["problem_type"] = "multi_label_classification"
	data, err = json.Marshal(config)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(configPath, data, 0o600); err != nil {
		t.Fatal(err)
	}
	options := InstanceOptions{ModelPath: path, ModelType: "modernbert", Overflow: "reject"}
	if invalid, loadErr := LoadSequenceClassifier(options); loadErr == nil {
		_ = invalid.Close()
		t.Fatal("multi-label head accepted by categorical loader")
	}
	model, err := LoadLabelScorer(options)
	if err != nil {
		t.Fatal(err)
	}
	clone, err := model.Clone()
	if err != nil {
		t.Fatal(err)
	}
	defer clone.Close()
	if err = model.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err = model.Score("hello"); !errors.Is(err, ErrInstanceClosed) {
		t.Fatalf("closed owner: %v", err)
	}
	result, err := clone.Score("hello world")
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Scores) != 2 || result.Scores[0]+result.Scores[1] <= 1 {
		t.Fatalf("independent scores were normalized: %+v", result)
	}
	windows, err := clone.ScoreWindows("hello world hello world hello", SequenceWindowOptions{Size: 3, Overlap: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(windows.Windows) != 2 || windows.ContentTokens != 5 || windows.Input.ProcessedTokens != 5 || windows.Input.Truncated {
		t.Fatalf("invalid native window result: %+v", windows)
	}
	if windows.Windows[0].Start != 0 || windows.Windows[0].End != 3 || windows.Windows[1].Start != 2 || windows.Windows[1].End != 5 {
		t.Fatalf("invalid exact token ranges: %+v", windows.Windows)
	}
	if _, err = clone.ScoreWindows("hello", SequenceWindowOptions{Size: 3, Overlap: -1}); err == nil {
		t.Fatal("negative overlap reached unsigned ABI")
	}
}
