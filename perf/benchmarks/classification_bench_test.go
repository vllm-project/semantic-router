//go:build !windows && cgo

package benchmarks

import (
	"context"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
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

// The explicit catalog tasks retain independent native resources. A directory
// scan must never substitute an unrelated model left in the download cache.
var benchClassifier *native.LoRABatch

func initClassifier(b *testing.B) {
	b.Helper()
	classifierOnce.Do(func() {
		domain := benchmarkModel(b, "domain", "label_distribution.v1")
		pii := benchmarkModel(b, "pii", "token_spans.v1")
		guard := benchmarkModel(b, "jailbreak", "label_distribution.v1")
		benchClassifier, classifierErr = benchmarkRuntime.LoRABatch(context.Background(), domain, pii, guard)
	})
	if classifierErr != nil {
		if missingBenchModels(classifierErr) {
			b.Skipf("Failed to initialize classifier: %v", classifierErr)
		}
		b.Fatalf("prepare owned Vela classifiers: %v", classifierErr)
	}
	recordModelIdentity(b, "domain", "pii", "jailbreak")
}

// BenchmarkClassifyBatch_Size1 benchmarks single text classification
func BenchmarkClassifyBatch_Size1(b *testing.B) {
	initClassifier(b)
	classifier := benchClassifier

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		text := testTexts[i%len(testTexts)]
		_, err := classifier.ClassifyBatch(context.Background(), "perf", []string{text})
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
		_, err := classifier.ClassifyBatch(context.Background(), "perf", batch)
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
		_, err := classifier.ClassifyBatch(context.Background(), "perf", batch)
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
		_, err := classifier.ClassifyBatch(context.Background(), "perf", batch)
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
			_, err := classifier.ClassifyBatch(context.Background(), "perf", []string{text})
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
		_, err := classifier.ClassifyBatch(context.Background(), "perf", texts)
		if err != nil {
			b.Fatalf("Classification failed: %v", err)
		}
	}
}
