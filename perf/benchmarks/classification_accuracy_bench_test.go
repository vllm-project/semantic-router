//go:build !windows && cgo

package benchmarks

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
)

const (
	INTENT_GOLD_DATASET_PATH = "INTENT_GOLD_DATASET_PATH"
	INTENT_MODEL_PATH_ENV    = "INTENT_MODEL_PATH"
	defaultIntentModelDir    = "models/mmbert32k-intent-classifier-merged"
)

var (
	intentInitOnce sync.Once
	intentInitErr  error
	intentMapping  *classification.CategoryMapping
)

func resolveIntentModelDir() string {
	if path := os.Getenv(INTENT_MODEL_PATH_ENV); path != "" {
		return path
	}
	wd, err := os.Getwd()
	if err != nil {
		return defaultIntentModelDir
	}
	return filepath.Join(wd, "..", "..", defaultIntentModelDir)
}

// initIntentClassifier loads the mmBERT-32K intent classifier and its category mapping.
// The candle binding's init is sync.Once-guarded internally, so calling it once per process is enough.
func initIntentClassifier(b *testing.B) {
	b.Helper()
	intentInitOnce.Do(func() {
		modelDir := resolveIntentModelDir()
		if _, statErr := os.Stat(modelDir); statErr != nil {
			intentInitErr = fmt.Errorf("intent model dir not found at %s: %w", modelDir, statErr)
			return
		}
		mapping, mapErr := classification.LoadCategoryMapping(filepath.Join(modelDir, "category_mapping.json"))
		if mapErr != nil {
			intentInitErr = fmt.Errorf("failed to load category mapping: %w", mapErr)
			return
		}
		if err := candle_binding.InitMmBert32KIntentClassifier(modelDir, true); err != nil {
			intentInitErr = fmt.Errorf("failed to init mmBERT-32K intent classifier: %w", err)
			return
		}
		intentMapping = mapping
	})
	if intentInitErr != nil {
		b.Fatalf("Failed to initialize intent classifier: %v", intentInitErr)
	}
}

type intentGoldExample struct {
	Text       string
	GoldIntent string
}

type intentGoldRow struct {
	Text       string `json:"text"`
	GoldIntent string `json:"gold_intent"`
	Label      string `json:"label"`
	Intent     string `json:"intent"`
}

func resolveIntentGoldDatasetPath() string {
	if path := os.Getenv(INTENT_GOLD_DATASET_PATH); path != "" {
		return path
	}

	// Anchor to perf/testdata/ regardless of cwd. Tests run from perf/benchmarks,
	// but `go test` invocations from elsewhere would otherwise fail to resolve the
	// relative path.
	testdataDir := filepath.Join("..", "testdata")
	if wd, err := os.Getwd(); err == nil {
		testdataDir = filepath.Join(wd, "..", "testdata")
	}

	// Prefer MMLU-derived dataset (exported via perf/scripts/export_mmlu_intent_gold.py)
	mmluPath := filepath.Join(testdataDir, "mmlu_intent_gold.jsonl")
	if _, err := os.Stat(mmluPath); err == nil {
		return mmluPath
	}

	// Fallback starter dataset
	return filepath.Join(testdataDir, "intent_gold.jsonl")
}

func normalizeIntentLabel(label string) string {
	return strings.ToLower(strings.TrimSpace(label))
}

func loadIntentGoldExamples(path string) ([]intentGoldExample, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	examples := make([]intentGoldExample, 0)
	scanner := bufio.NewScanner(file)
	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		var row intentGoldRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			return nil, fmt.Errorf("invalid JSONL at line %d: %w", lineNum, err)
		}

		gold := row.GoldIntent
		if gold == "" {
			gold = row.Label
		}
		if gold == "" {
			gold = row.Intent
		}
		if strings.TrimSpace(row.Text) == "" || strings.TrimSpace(gold) == "" {
			return nil, fmt.Errorf("missing text or gold label at line %d", lineNum)
		}

		examples = append(examples, intentGoldExample{
			Text:       row.Text,
			GoldIntent: gold,
		})
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return examples, nil
}

func BenchmarkClassifyIntent_GoldAccuracy(b *testing.B) {
	initIntentClassifier(b)

	datasetPath := resolveIntentGoldDatasetPath()
	examples, err := loadIntentGoldExamples(datasetPath)
	if err != nil {
		b.Skipf("intent gold dataset not ready (%v); set %s or provide %s", err, INTENT_GOLD_DATASET_PATH, datasetPath)
	}
	if len(examples) == 0 {
		b.Skipf("intent gold dataset is empty: %s", datasetPath)
	}

	normalizedGold := make([]string, len(examples))
	for i := range examples {
		normalizedGold[i] = normalizeIntentLabel(examples[i].GoldIntent)
	}

	b.ReportAllocs()
	b.ResetTimer()

	var accuracy float64
	for i := 0; i < b.N; i++ {
		correct := 0
		for idx, ex := range examples {
			result, classifyErr := candle_binding.ClassifyMmBert32KIntent(ex.Text)
			if classifyErr != nil {
				b.Fatalf("classification failed: %v", classifyErr)
			}
			label, ok := intentMapping.GetCategoryFromIndex(result.Class)
			if !ok {
				continue
			}
			if normalizeIntentLabel(label) == normalizedGold[idx] {
				correct++
			}
		}
		accuracy = float64(correct) / float64(len(examples))
	}

	b.StopTimer()
	b.ReportMetric(accuracy*100, "intent_acc_pct")
	b.Logf("dataset=%s samples=%d intent_accuracy=%.2f%%", datasetPath, len(examples), accuracy*100)
}