//go:build !windows && cgo

package benchmarks

import (
	"os"
	"path/filepath"
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
		_, err = classification.AutoInitializeUnifiedClassifier(modelsDir)
		if err != nil {
			classifierErr = err
			return
		}
	})

	if classifierErr != nil {
		b.Fatalf("Failed to initialize classifier: %v", classifierErr)
	}
}

// BenchmarkClassifyBatch_Size1 benchmarks single text classification
func BenchmarkClassifyBatch_Size1(b *testing.B) {
	initClassifier(b)
	classifier := classification.GetGlobalUnifiedClassifier()

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
	classifier := classification.GetGlobalUnifiedClassifier()

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
	classifier := classification.GetGlobalUnifiedClassifier()

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
	classifier := classification.GetGlobalUnifiedClassifier()

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
	classifier := classification.GetGlobalUnifiedClassifier()

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
	classifier := classification.GetGlobalUnifiedClassifier()

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
