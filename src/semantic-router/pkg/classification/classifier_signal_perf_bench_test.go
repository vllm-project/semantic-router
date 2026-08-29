package classification

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

var benchmarkGenericSignalResult atomic.Pointer[SignalResults]

// deterministicBenchmarkLabelClassifier replaces model execution only. It
// still consumes the full input and returns the same label-score shape as a
// learned classifier, allowing CPU CI to isolate Router fanout, aggregation,
// synchronization, and allocation scaling without claiming model latency.
type deterministicBenchmarkLabelClassifier struct{}

func (deterministicBenchmarkLabelClassifier) Classify(
	_ context.Context,
	input string,
) (labelClassification, error) {
	checksum := uint64(1469598103934665603)
	for index := range len(input) {
		checksum ^= uint64(input[index])
		checksum *= 1099511628211
	}
	matched := float64(checksum&1) * 0.1
	return labelClassification{Scores: map[string]float64{
		"no-match": 0.9 - matched,
		"match":    0.1 + matched,
	}}, nil
}

// BenchmarkGenericClassifierSignalTopology is a model-isolated interaction
// matrix. Context sweeps hold request batch at one; batch sweeps hold context
// at 2K tokens. That produces readable curves without a full Cartesian matrix.
func BenchmarkGenericClassifierSignalTopology(b *testing.B) {
	for _, signalCount := range []int{0, 1, 4, 8} {
		benchmarkGenericClassifierSignalCount(b, signalCount)
	}
}

func benchmarkGenericClassifierSignalCount(b *testing.B, signalCount int) {
	classifier, usedSignals := benchmarkGenericSignalClassifier(signalCount)
	for _, contextTokens := range []int{128, 512, 2048, 8192, 32768} {
		benchmarkGenericSignalContext(b, classifier, usedSignals, contextTokens, signalCount)
	}
	for _, requestBatch := range []int{1, 4, 16, 32} {
		benchmarkGenericSignalRequestBatch(b, classifier, usedSignals, requestBatch, signalCount)
	}
}

func benchmarkGenericSignalContext(
	b *testing.B,
	classifier *Classifier,
	usedSignals map[string]bool,
	contextTokens int,
	signalCount int,
) {
	input := strings.Repeat("routing ", contextTokens)
	name := fmt.Sprintf(
		"Context/context_tokens=%d/learned_signals=%d/learned_signal_set=generic-classifier/signal_backend=deterministic_stub",
		contextTokens,
		signalCount,
	)
	b.Run(name, func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			result := benchmarkEvaluateGenericSignals(classifier, usedSignals, input)
			if len(result.MatchedClassifierRules) != signalCount {
				b.Fatalf("matched signals = %d, want %d", len(result.MatchedClassifierRules), signalCount)
			}
			benchmarkGenericSignalResult.Store(result)
		}
		b.ReportMetric(float64(signalCount), "learned_signals/op")
	})
}

func benchmarkGenericSignalRequestBatch(
	b *testing.B,
	classifier *Classifier,
	usedSignals map[string]bool,
	requestBatch int,
	signalCount int,
) {
	input := strings.Repeat("routing ", 2048)
	name := fmt.Sprintf(
		"RequestBatch/request_batch=%d/context_tokens=2048/learned_signals=%d/learned_signal_set=generic-classifier/signal_backend=deterministic_stub",
		requestBatch,
		signalCount,
	)
	b.Run(name, func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			benchmarkEvaluateGenericSignalBatch(b, classifier, usedSignals, input, requestBatch, signalCount)
		}
		if elapsed := b.Elapsed().Seconds(); elapsed > 0 {
			b.ReportMetric(float64(b.N*requestBatch)/elapsed, "requests/s")
		}
		b.ReportMetric(float64(signalCount*requestBatch), "learned_signals/op")
	})
}

func benchmarkGenericSignalClassifier(signalCount int) (*Classifier, map[string]bool) {
	rules := make([]config.ClassifierSignalRule, signalCount)
	classifiers := make(map[string]labelClassifier, signalCount)
	usedSignals := make(map[string]bool, signalCount)
	for index := range signalCount {
		name := fmt.Sprintf("learned-%d", index)
		rules[index] = config.ClassifierSignalRule{
			Name: name, Type: "local", Labels: []string{"no-match", "match"},
		}
		classifiers[name] = deterministicBenchmarkLabelClassifier{}
		usedSignals[config.SignalTypeClassifier+":"+name] = true
	}
	return &Classifier{
		Config: &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{ClassifierRules: rules},
		}},
		genericClassifiers: classifiers,
	}, usedSignals
}

func benchmarkEvaluateGenericSignals(
	classifier *Classifier,
	usedSignals map[string]bool,
	input string,
) *SignalResults {
	result := &SignalResults{
		Metrics: &SignalMetricsCollection{}, SignalConfidences: map[string]float64{},
		SignalValues: map[string]float64{}, SignalErrors: map[string]string{},
	}
	classifier.evaluateGenericClassifierSignals(result, &sync.Mutex{}, input, usedSignals, nil)
	return result
}

func benchmarkEvaluateGenericSignalBatch(
	b *testing.B,
	classifier *Classifier,
	usedSignals map[string]bool,
	input string,
	requestBatch int,
	signalCount int,
) {
	var waitGroup sync.WaitGroup
	waitGroup.Add(requestBatch)
	for range requestBatch {
		go func() {
			defer waitGroup.Done()
			result := benchmarkEvaluateGenericSignals(classifier, usedSignals, input)
			if len(result.MatchedClassifierRules) != signalCount {
				b.Errorf("matched signals = %d, want %d", len(result.MatchedClassifierRules), signalCount)
			}
			benchmarkGenericSignalResult.Store(result)
		}()
	}
	waitGroup.Wait()
}
