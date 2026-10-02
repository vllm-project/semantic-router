//go:build !windows && cgo && (amd64 || arm64 || riscv64)

package candle_binding

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

const classifierParityConfidenceTol = 1e-3

var classifierParityTexts = []string{
	"What is the derivative of x squared?",
	"Explain the process of photosynthesis",
}

type classifierParityRow struct {
	Text       string  `json:"text"`
	Class      int     `json:"class"`
	Confidence float32 `json:"confidence"`
}

type classifierParitySnapshot struct {
	Rows []classifierParityRow `json:"rows"`
}

func requiredClassifierModelPath(t *testing.T) string {
	t.Helper()
	path := os.Getenv("CANDLE_CLASSIFIER_MODEL")
	if path == "" {
		path = "../models/Vela-1.0-Encoder-307M-Domain"
	}
	if _, err := os.Stat(filepath.Join(path, "config.json")); err != nil {
		t.Fatalf("required classifier checkpoint missing at %s: %v", path, err)
	}
	return path
}

func classifyParityRows(t *testing.T, modelPath string) []classifierParityRow {
	t.Helper()
	if err := InitMmBert32KIntentClassifierWithMaxSequenceLength(modelPath, true, 512); err != nil {
		t.Fatalf("InitMmBert32KIntentClassifierWithMaxSequenceLength(%s): %v", modelPath, err)
	}
	rows := make([]classifierParityRow, 0, len(classifierParityTexts))
	for _, text := range classifierParityTexts {
		result, err := ClassifyMmBert32KIntent(text)
		if err != nil {
			t.Fatalf("ClassifyMmBert32KIntent(%q): %v", text, err)
		}
		if result.Class < 0 {
			t.Fatalf("ClassifyMmBert32KIntent(%q): invalid class %d", text, result.Class)
		}
		if result.Confidence <= 0 || result.Confidence > 1 {
			t.Fatalf("ClassifyMmBert32KIntent(%q): confidence %f out of (0,1]", text, result.Confidence)
		}
		rows = append(rows, classifierParityRow{
			Text:       text,
			Class:      result.Class,
			Confidence: result.Confidence,
		})
	}
	return rows
}

func TestCandleClassifierParity(t *testing.T) {
	mode := os.Getenv("CANDLE_CLASSIFIER_PARITY_MODE")
	if mode == "" {
		if runtime.GOARCH == "riscv64" {
			t.Fatal("riscv64 requires CANDLE_CLASSIFIER_PARITY_MODE=compare")
		}
		t.Skip("set CANDLE_CLASSIFIER_PARITY_MODE=record or compare")
	}

	got := classifierParitySnapshot{Rows: classifyParityRows(t, requiredClassifierModelPath(t))}
	goldenPath := os.Getenv("CANDLE_CLASSIFIER_PARITY_GOLDEN")
	if goldenPath == "" {
		t.Fatal("CANDLE_CLASSIFIER_PARITY_GOLDEN is required")
	}

	switch mode {
	case "record":
		data, err := json.MarshalIndent(got, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.MkdirAll(filepath.Dir(goldenPath), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(goldenPath, append(data, '\n'), 0o600); err != nil {
			t.Fatal(err)
		}
	case "compare":
		data, err := os.ReadFile(goldenPath)
		if err != nil {
			t.Fatalf("read golden %s: %v", goldenPath, err)
		}
		var want classifierParitySnapshot
		if err := json.Unmarshal(data, &want); err != nil {
			t.Fatal(err)
		}
		if len(want.Rows) != len(got.Rows) {
			t.Fatalf("golden has %d rows, riscv produced %d", len(want.Rows), len(got.Rows))
		}
		for i := range want.Rows {
			if want.Rows[i].Text != got.Rows[i].Text {
				t.Fatalf("row %d text: golden %q, got %q", i, want.Rows[i].Text, got.Rows[i].Text)
			}
			if want.Rows[i].Class != got.Rows[i].Class {
				t.Fatalf("row %d class: golden %d, got %d", i, want.Rows[i].Class, got.Rows[i].Class)
			}
			diff := math.Abs(float64(want.Rows[i].Confidence - got.Rows[i].Confidence))
			if diff > classifierParityConfidenceTol {
				t.Fatalf("row %d confidence: golden %f, got %f, abs diff %g",
					i, want.Rows[i].Confidence, got.Rows[i].Confidence, diff)
			}
		}
	default:
		t.Fatalf("unknown CANDLE_CLASSIFIER_PARITY_MODE %q", mode)
	}
}
