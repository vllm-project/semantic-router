//go:build !windows && cgo

package benchmarks

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
)

var (
	testTexts = []string{
		"What is the derivative of x^2 + 3x + 5?",
		"How do I implement a binary search tree in Python?",
		"Explain the benefits of cloud computing for businesses",
		"What is the capital of France?",
		"How does photosynthesis work in plants?",
	}

	classifierOnce sync.Once
	classifierErr  error
)

// initClassifier initializes the global unified classifier once
var benchClassifier *classification.UnifiedClassifier

func initClassifier(b *testing.B) {
	classifierOnce.Do(func() {
		// Find the project root (semantic-router-fork)
		wd, err := os.Getwd()
		if err != nil {
			classifierErr = err
			return
		}

		// Navigate up to find the project root
		projectRoot := filepath.Join(wd, "../..")

		// Use auto-discovery to initialize classifier
		modelsDir := filepath.Join(projectRoot, "models")
		c, err := classification.AutoInitializeUnifiedClassifier(modelsDir)
		if err != nil {
			classifierErr = err
			return
		}
		benchClassifier = c
	})

	if classifierErr != nil {
		b.Fatalf("Failed to initialize classifier: %v", classifierErr)
	}
}

// BenchmarkClassifyBatch_Size1 benchmarks single text classification
func BenchmarkClassifyBatch_Size1(b *testing.B) {
	initClassifier(b)
	classifier := benchClassifier

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		text := testTexts[i%len(testTexts)]
		_, err := classifier.ClassifyBatch([]string{text})
		if err != nil {
			b.Fatalf("Classification failed: %v", err)
		}
	}
}

// BenchmarkClassifyBatch_Size10 benchmarks batch of 10 texts
func BenchmarkClassifyBatch_Size10(b *testing.B) {
	initClassifier(b)
	classifier := benchClassifier

	// Prepare batch
	batch := make([]string, 10)
	for i := 0; i < 10; i++ {
		batch[i] = testTexts[i%len(testTexts)]
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := classifier.ClassifyBatch(batch)
		if err != nil {
			b.Fatalf("Classification failed: %v", err)
		}
	}
}

// BenchmarkClassifyBatch_Size50 benchmarks batch of 50 texts
func BenchmarkClassifyBatch_Size50(b *testing.B) {
	initClassifier(b)
	classifier := benchClassifier

	// Prepare batch
	batch := make([]string, 50)
	for i := 0; i < 50; i++ {
		batch[i] = testTexts[i%len(testTexts)]
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := classifier.ClassifyBatch(batch)
		if err != nil {
			b.Fatalf("Classification failed: %v", err)
		}
	}
}

// BenchmarkClassifyBatch_Size100 benchmarks batch of 100 texts
func BenchmarkClassifyBatch_Size100(b *testing.B) {
	initClassifier(b)
	classifier := benchClassifier

	// Prepare batch
	batch := make([]string, 100)
	for i := 0; i < 100; i++ {
		batch[i] = testTexts[i%len(testTexts)]
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := classifier.ClassifyBatch(batch)
		if err != nil {
			b.Fatalf("Classification failed: %v", err)
		}
	}
}

// BenchmarkClassifyBatch_Parallel benchmarks parallel classification
func BenchmarkClassifyBatch_Parallel(b *testing.B) {
	initClassifier(b)
	classifier := benchClassifier

	b.ResetTimer()
	b.ReportAllocs()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			text := testTexts[0]
			_, err := classifier.ClassifyBatch([]string{text})
			if err != nil {
				b.Fatalf("Classification failed: %v", err)
			}
		}
	})
}

// BenchmarkCGOOverhead measures the overhead of CGO calls
func BenchmarkCGOOverhead(b *testing.B) {
	initClassifier(b)
	classifier := benchClassifier

	texts := []string{"Simple test text"}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := classifier.ClassifyBatch(texts)
		if err != nil {
			b.Fatalf("Classification failed: %v", err)
		}
	}
}

// BenchmarkLearnedSignalKernel is the model-backed counterpart to the Router
// orchestration topology benchmark. The unified classifier currently produces
// intent, PII, and security outputs in one native call, so the signal set and
// count are explicit rather than pretending arbitrary learned families have
// interchangeable cost.
func BenchmarkLearnedSignalKernel(b *testing.B) {
	initClassifier(b)
	for _, contextTokens := range []int{128, 512, 2048} {
		benchmarkLearnedSignalContext(b, contextTokens)
	}
}

func benchmarkLearnedSignalContext(b *testing.B, contextTokens int) {
	text := strings.Repeat("routing ", contextTokens)
	for _, classifierBatch := range []int{1, 4, 16} {
		batch := make([]string, classifierBatch)
		for index := range batch {
			batch[index] = text
		}
		name := fmt.Sprintf(
			"classifier_batch=%d/context_tokens=%d/learned_signal_set=unified-intent-pii-security/learned_signals=3/signal_backend=native-unified",
			classifierBatch,
			contextTokens,
		)
		b.Run(name, func(b *testing.B) {
			benchmarkLearnedSignalBatch(b, batch)
		})
	}
}

func benchmarkLearnedSignalBatch(b *testing.B, batch []string) {
	b.ReportAllocs()
	for b.Loop() {
		results, err := benchClassifier.ClassifyBatch(batch)
		if err != nil {
			b.Fatal(err)
		}
		if results == nil || results.BatchSize != len(batch) {
			b.Fatalf("classified batch = %v, want %d", results, len(batch))
		}
	}
	if elapsed := b.Elapsed().Seconds(); elapsed > 0 {
		b.ReportMetric(float64(b.N*len(batch))/elapsed, "texts/s")
	}
	b.ReportMetric(float64(len(batch)), "texts/op")
}
