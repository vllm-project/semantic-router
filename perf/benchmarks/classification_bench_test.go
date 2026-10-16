//go:build !windows

package benchmarks

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
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

// signalClassifier runs the Domain, PII and Guard signals a request evaluates
// together, bundled the way the router bundles one request stage.
type signalClassifier struct {
	domain    *binding.Resolved[string, tasks.LabelDistribution]
	pii       *binding.Resolved[string, tasks.TokenClassificationResult]
	jailbreak *binding.Resolved[string, tasks.LabelDistribution]
}

// ClassifyBatch runs every signal on every text in one bundle.
func (c *signalClassifier) ClassifyBatch(ctx context.Context, recipe string, texts []string) error {
	ctx, bundle := modelservice.WithBundle(ctx, 0)
	defer bundle.Join()()
	errs := make([]error, 3*len(texts))
	modelservice.Fan(ctx, len(errs), func(i int) {
		text := texts[i/3]
		switch i % 3 {
		case 0:
			_, errs[i] = c.domain.Call(ctx, recipe, text)
		case 1:
			_, errs[i] = c.pii.Call(ctx, recipe, text)
		default:
			_, errs[i] = c.jailbreak.Call(ctx, recipe, text)
		}
	})
	return errors.Join(errs...)
}

func (c *signalClassifier) Close() error {
	var errs []error
	if c.domain != nil {
		errs = append(errs, c.domain.Close())
	}
	if c.pii != nil {
		errs = append(errs, c.pii.Close())
	}
	if c.jailbreak != nil {
		errs = append(errs, c.jailbreak.Close())
	}
	return errors.Join(errs...)
}

var benchClassifier *signalClassifier

func initClassifier(b *testing.B) {
	b.Helper()
	classifierOnce.Do(func() {
		// benchmarkModel stops this benchmark (b.Fatalf) inside Do, and Do still
		// counts as done, so later benchmarks fail on this error unless setup ends.
		classifierErr = errors.New("an earlier benchmark stopped while preparing them")
		ctx := context.Background()
		classifier := &signalClassifier{}
		var err error
		if classifier.domain, err = benchmarkRuntime.Sequence(ctx, benchmarkModel(b, "domain", "label_distribution.v1")); err == nil {
			if classifier.pii, err = benchmarkRuntime.Tokens(ctx, benchmarkModel(b, "pii", "token_spans.v1")); err == nil {
				classifier.jailbreak, err = benchmarkRuntime.Sequence(ctx, benchmarkModel(b, "jailbreak", "label_distribution.v1"))
			}
		}
		if err != nil {
			classifierErr = errors.Join(err, classifier.Close())
			return
		}
		benchClassifier = classifier
		classifierErr = nil
	})
	if classifierErr != nil {
		b.Fatalf("prepare the Vela signal classifiers: %v", classifierErr)
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
		err := classifier.ClassifyBatch(context.Background(), "perf", []string{text})
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
		err := classifier.ClassifyBatch(context.Background(), "perf", batch)
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
		err := classifier.ClassifyBatch(context.Background(), "perf", batch)
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
		err := classifier.ClassifyBatch(context.Background(), "perf", batch)
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
			err := classifier.ClassifyBatch(context.Background(), "perf", []string{text})
			if err != nil {
				b.Fatalf("Classification failed: %v", err)
			}
		}
	})
}

// BenchmarkClassifyRuntimeOverhead measures one short request through the
// runtime transport, bundling and result decoding.
func BenchmarkClassifyRuntimeOverhead(b *testing.B) {
	initClassifier(b)
	classifier := benchClassifier

	texts := []string{"Simple test text"}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		err := classifier.ClassifyBatch(context.Background(), "perf", texts)
		if err != nil {
			b.Fatalf("Classification failed: %v", err)
		}
	}
}
