//go:build !windows

package benchmarks

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
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

// presetQuestion is the question an application rule of a preset asks, as
// the Router writes it for the decision model's card to compile.
func presetQuestion(preset string) modelservice.Question {
	canonical, _ := presets.Lookup(preset, presets.StageRequest)
	written := config.PresetSystemOneQuestion(canonical)
	question := modelservice.Question{
		ID: "preset:" + preset, TaskID: modelservice.PresetTask(preset), Stage: canonical.Stage,
		Type: written.Type, Instructions: written.Instructions, Truncate: !canonical.Full,
	}
	for _, choice := range written.Choices {
		question.Choices = append(question.Choices, modelservice.Choice{Key: choice.Key, Description: choice.Description})
	}
	for _, label := range written.Labels {
		question.Labels = append(question.Labels, modelservice.Choice{Key: label.Key, Description: label.Description})
	}
	return question
}

// signalClassifier asks the decision model the Domain, Guard and PII
// questions a request stage asks together, bundled the way the router bundles
// one request stage.
type signalClassifier struct {
	deployment string
	card       modelservice.ModelCard
	questions  []modelservice.Question
}

// ask asks one text's questions in one call, as a request stage does.
func (c *signalClassifier) ask(ctx context.Context, text string) (modelservice.Response, error) {
	return modelservice.ExecuteQuestions(ctx, benchmarkLease, c.deployment, c.card, modelservice.Request{State: text, Questions: c.questions})
}

// ClassifyBatch asks every text's questions in one bundle.
func (c *signalClassifier) ClassifyBatch(ctx context.Context, texts []string) error {
	ctx, bundle := modelservice.WithBundle(ctx, 0)
	defer bundle.Join()()
	errs := make([]error, len(texts))
	modelservice.Fan(ctx, len(texts), func(i int) {
		_, errs[i] = c.ask(ctx, texts[i])
	})
	return errors.Join(errs...)
}

var benchClassifier *signalClassifier

func newSignalClassifier(b *testing.B, deployment string, input config.ModelInputBudget, presetIDs ...string) *signalClassifier {
	b.Helper()
	spec := benchmarkDeployment(b, "decision", deployment, input)
	classifier := &signalClassifier{deployment: spec.Binding.Deployment, card: benchmarkCards["decision"]}
	for _, preset := range presetIDs {
		classifier.questions = append(classifier.questions, presetQuestion(preset))
	}
	return classifier
}

func initClassifier(b *testing.B) {
	b.Helper()
	classifierOnce.Do(func() {
		// benchmarkDeployment stops this benchmark (b.Fatalf) inside Do, and Do
		// still counts as done, so later benchmarks fail on this error unless
		// setup ends.
		classifierErr = errors.New("an earlier benchmark stopped while preparing them")
		benchClassifier = newSignalClassifier(b, "perf-decision", benchmarkInput, presets.Domain, presets.Jailbreak, presets.PII)
		classifierErr = nil
	})
	if classifierErr != nil {
		b.Fatalf("prepare the decision model: %v", classifierErr)
	}
	recordModelIdentity(b, "decision")
}

// BenchmarkClassifyBatch_Size1 benchmarks single text classification
func BenchmarkClassifyBatch_Size1(b *testing.B) {
	initClassifier(b)
	classifier := benchClassifier

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		text := testTexts[i%len(testTexts)]
		err := classifier.ClassifyBatch(context.Background(), []string{text})
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
		err := classifier.ClassifyBatch(context.Background(), batch)
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
		err := classifier.ClassifyBatch(context.Background(), batch)
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
		err := classifier.ClassifyBatch(context.Background(), batch)
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
			err := classifier.ClassifyBatch(context.Background(), []string{text})
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
		err := classifier.ClassifyBatch(context.Background(), texts)
		if err != nil {
			b.Fatalf("Classification failed: %v", err)
		}
	}
}
