package classification

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestLocalClassifierLoadFailureDoesNotReserveGlobalSlot(t *testing.T) {
	for _, name := range []string{"first", "second"} {
		_, err := newLocalLabelClassifier(config.ClassifierSignalRule{Name: name, Type: "local", ModelPath: filepath.Join(t.TempDir(), name), UseCPU: true, Labels: []string{"SAFE", "RISKY"}})
		if err == nil || strings.Contains(err.Error(), "restart the router") {
			t.Fatalf("failed candidate should report its own load failure: %v", err)
		}
	}
}

func TestLocalClassifierMaintainedCPU(t *testing.T) {
	path := os.Getenv("CANDLE_GENERIC_CLASSIFIER_MODEL")
	if path == "" {
		t.Skip("set CANDLE_GENERIC_CLASSIFIER_MODEL to a local checkpoint")
	}
	data, err := os.ReadFile(filepath.Join(path, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var metadata struct {
		ID2Label map[string]string `json:"id2label"`
	}
	if err = json.Unmarshal(data, &metadata); err != nil {
		t.Fatal(err)
	}
	labels := make([]string, len(metadata.ID2Label))
	for index := range labels {
		label, ok := metadata.ID2Label[strconv.Itoa(index)]
		if !ok || label == "" {
			t.Fatalf("missing label for class %d", index)
		}
		labels[index] = label
	}
	classifier, err := newLocalLabelClassifier(config.ClassifierSignalRule{
		Name: "model-compatibility", Type: "local", ModelPath: path, UseCPU: true, Labels: labels,
	})
	if err != nil {
		t.Fatal(err)
	}
	local, ok := classifier.(*localLabelClassifier)
	if !ok {
		t.Fatalf("unexpected local classifier type %T", classifier)
	}
	defer local.Close()
	capability := local.backend.handle.Capability()
	if capability.Provider != "candle" || capability.Device != "cpu" {
		t.Fatalf("expected Candle CPU execution, got %s/%s", capability.Provider, capability.Device)
	}
	for _, text := range []string{
		"Please explain how solar panels produce electricity.",
		"The meeting starts at nine tomorrow.",
	} {
		result, err := classifier.Classify(context.Background(), text)
		if err != nil {
			t.Fatal(err)
		}
		if len(result.Scores) != len(labels) {
			t.Fatalf("incomplete distribution: %v", result.Scores)
		}
		total := 0.0
		for _, label := range labels {
			score, ok := result.Scores[label]
			if !ok || math.IsNaN(score) || math.IsInf(score, 0) || score < 0 || score > 1 {
				t.Fatalf("invalid score for %q: %v", label, result.Scores)
			}
			total += score
		}
		if math.Abs(total-1) > 1e-5 {
			t.Fatalf("distribution sums to %g", total)
		}
		t.Logf("model=%s device=%s input=%q scores=%v", path, capability.Device, text, result.Scores)
	}
}
