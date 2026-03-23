package evaluation

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestExtractMMLUProMetrics(t *testing.T) {
	t.Parallel()
	raw := map[string]interface{}{
		"overall_accuracy":   0.85,
		"total_questions":    100,
		"successful_queries": 98,
		"failed_queries":     2,
		"avg_response_time":  1.5,
		"category_accuracy":  map[string]interface{}{"math": 0.9, "physics": 0.8},
	}
	metrics := extractMMLUProMetrics(raw)
	if metrics["status"] != "success" {
		t.Errorf("status = %v, want success", metrics["status"])
	}
	if metrics["accuracy"] != 0.85 {
		t.Errorf("accuracy = %v, want 0.85", metrics["accuracy"])
	}
	if metrics["overall_accuracy"] != 0.85 {
		t.Errorf("overall_accuracy = %v, want 0.85", metrics["overall_accuracy"])
	}
	switch v := metrics["total_questions"]; val := v.(type) {
	case float64:
		if val != 100 {
			t.Errorf("total_questions = %v, want 100", val)
		}
	case int:
		if val != 100 {
			t.Errorf("total_questions = %v, want 100", val)
		}
	default:
		t.Errorf("total_questions = %v (type %T), want 100", v, v)
	}
}

func TestParseMMLUProOutput(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	subdir := filepath.Join(dir, "model_direct")
	if err := os.MkdirAll(subdir, 0o755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}
	analysis := map[string]interface{}{
		"overall_accuracy":   0.75,
		"total_questions":    50,
		"successful_queries": 50,
		"failed_queries":     0,
		"avg_response_time":  2.0,
		"category_accuracy":  map[string]interface{}{"default": 0.75},
	}
	data, err := json.Marshal(analysis)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if writeErr := os.WriteFile(filepath.Join(subdir, "analysis.json"), data, 0o600); writeErr != nil {
		t.Fatalf("WriteFile: %v", writeErr)
	}

	metrics, err := ParseMMLUProOutput(dir)
	if err != nil {
		t.Fatalf("ParseMMLUProOutput() error = %v", err)
	}
	if metrics["status"] != "success" {
		t.Errorf("status = %v, want success", metrics["status"])
	}
	if metrics["accuracy"] != 0.75 {
		t.Errorf("accuracy = %v, want 0.75", metrics["accuracy"])
	}
}

func TestParseMMLUProOutput_NoAnalysisFile(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	_, err := ParseMMLUProOutput(dir)
	if err == nil {
		t.Fatal("ParseMMLUProOutput() expected error when no analysis.json present")
	}
}

func TestParseSignalEvalOutput(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "signal_eval_domain.json")
	data := []byte(`{"dimension":"domain","total_samples":10,"correct":8,"incorrect":1,"skipped":1,"accuracy":0.8}`)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	metrics, err := ParseSignalEvalOutput(path)
	if err != nil {
		t.Fatalf("ParseSignalEvalOutput() error = %v", err)
	}
	if metrics["status"] != "success" {
		t.Errorf("status = %v, want success", metrics["status"])
	}
	if metrics["accuracy"] != 0.8 {
		t.Errorf("accuracy = %v, want 0.8", metrics["accuracy"])
	}
}
