//go:build !windows && cgo

package benchmarks

import (
	"os"
	"strings"
	"sync"
	"testing"

	binding "github.com/vllm-project/semantic-router/candle-binding"
)

var (
	prefOnce  sync.Once
	prefError error
	prefPath  string

	adapterOnce sync.Once
	adapterErr  error
)

// set these environment variables to customize paths
// if not set, assume the models are located at project-root/models/
const QWEN3_PREF_MODEL_DIR = "QWEN3_PREF_MODEL_DIR"
const QWEN3_BASE_MODEL_PATH = "QWEN3_BASE_MODEL_PATH"
const QWEN3_CATEGORY_ADAPTER_PATH = "QWEN3_CATEGORY_ADAPTER_PATH"

func resolvePrefModelPath() (string, bool) {
	path := os.Getenv(QWEN3_PREF_MODEL_DIR)
	if path == "" {
		path = "../../models/mom-preference-classifier"
	}
	if _, err := os.Stat(path); err != nil {
		return "", false
	}
	return path, true
}

func initPreference(b *testing.B) {
	prefOnce.Do(func() {
		var ok bool
		prefPath, ok = resolvePrefModelPath()
		if !ok {
			prefError = os.ErrNotExist
			return
		}
		prefError = binding.InitQwen3PreferenceClassifier(prefPath, true)
	})
	if prefError != nil {
		b.Skipf("preference classifier not ready (%v); set QWEN3_PREF_MODEL_DIR or place weights at ./models/mom-preference-classifier", prefError)
	}
}

func benchmarkPreference(b *testing.B, text string, labels []string, expectedLabel string) {
	initPreference(b)

	// one time warm up
	if _, err := binding.ClassifyQwen3Preference(text, labels); err != nil {
		b.Fatalf("warmup failed: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	var lastLabel string

	for i := 0; i < b.N; i++ {
		res, err := binding.ClassifyQwen3Preference(text, labels)
		if err != nil {
			b.Fatalf("preference classification failed: %v", err)
		}
		if res.Class >= 0 && res.Class < len(labels) {
			lastLabel = labels[res.Class]
		}
	}

	b.StopTimer()
	if lastLabel != "" {
		b.Logf("last predicted label: %s", lastLabel)
		if expectedLabel != "" && lastLabel != expectedLabel {
			b.Fatalf("expected label %s, but got %s", expectedLabel, lastLabel)
		}
	}
}

func BenchmarkPreference_ShortLabels(b *testing.B) {
	labels := []string{"code_generation", "bug_fixing", "code_review", "weather_inquiry"}
	text := "Generate some python code for me."
	benchmarkPreference(b, text, labels, "code_generation")
}

func BenchmarkPreference_LongLabels(b *testing.B) {
	labels := []string{
		"biology", "chemistry", "physics", "math", "computer science", "economics",
		"history", "law", "psychology", "engineering", "medicine", "literature",
		"philosophy", "geography", "politics", "art", "music", "sports",
		"finance", "other",
	}
	text := "Explain how reinforcement learning differs from supervised learning."
	benchmarkPreference(b, text, labels, "computer science")
}

func BenchmarkPreference_VeryLongQuery(b *testing.B) {
	labels := []string{"code_generation", "bug_fixing", "code_review", "weather_inquiry"}
	text := strings.Repeat("Generate some python code for me.", 200)
	benchmarkPreference(b, text, labels, "code_generation")
}

// =============================================================================================
// Qwen3 Multi-LoRA adapter benchmarks (category adapter as baseline)
// =============================================================================================

func resolveQwen3BasePath() string {
	if path := os.Getenv(QWEN3_BASE_MODEL_PATH); path != "" {
		return path
	}
	return "../../models/Qwen3-0.6B"
}

func resolveCategoryAdapterPath() string {
	if path := os.Getenv(QWEN3_CATEGORY_ADAPTER_PATH); path != "" {
		return path
	}
	return "../../models/qwen3_generative_classifier_r16"
}

func initCategoryAdapter(b *testing.B) {
	adapterOnce.Do(func() {
		basePath := resolveQwen3BasePath()
		if _, err := os.Stat(basePath); err != nil {
			adapterErr = err
			return
		}

		if err := binding.InitQwen3MultiLoRAClassifier(basePath); err != nil {
			adapterErr = err
			return
		}

		adapterPath := resolveCategoryAdapterPath()
		if _, err := os.Stat(adapterPath); err != nil {
			adapterErr = err
			return
		}
		adapterErr = binding.LoadQwen3LoRAAdapter("category", adapterPath)
	})

	if adapterErr != nil {
		b.Skipf("category adapter not ready (%v); set QWEN3_BASE_MODEL_PATH and QWEN3_CATEGORY_ADAPTER_PATH", adapterErr)
	}
}

func benchmarkCategoryAdapter(b *testing.B, text string) {
	initCategoryAdapter(b)

	// one time warm up
	if _, err := binding.ClassifyWithQwen3Adapter(text, "category"); err != nil {
		b.Fatalf("warmup failed: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		if _, err := binding.ClassifyWithQwen3Adapter(text, "category"); err != nil {
			b.Fatalf("category adapter classification failed: %v", err)
		}
	}
}

func BenchmarkCategoryAdapter_Short(b *testing.B) {
	text := "Generate some python code for me."
	benchmarkCategoryAdapter(b, text)
}

func BenchmarkCategoryAdapter_VeryLongQuery(b *testing.B) {
	text := strings.Repeat("Generate some python code for me.", 200)
	benchmarkCategoryAdapter(b, text)
}
