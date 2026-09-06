package classification

import (
	"context"
	"errors"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type fakeLabelClassifier struct {
	result labelClassification
	err    error
	calls  *int
}

type blockingLabelClassifier struct {
	started chan<- struct{}
	release <-chan struct{}
}

func (f blockingLabelClassifier) Classify(
	ctx context.Context,
	_ string,
) (labelClassification, error) {
	f.started <- struct{}{}
	select {
	case <-f.release:
		return labelClassification{
			Scores: map[string]float64{"NO": 0, "YES": 1},
		}, nil
	case <-ctx.Done():
		return labelClassification{}, ctx.Err()
	}
}

func (f fakeLabelClassifier) Classify(
	context.Context,
	string,
) (labelClassification, error) {
	if f.calls != nil {
		(*f.calls)++
	}
	return f.result, f.err
}

func TestEvaluateGenericClassifierSignals(t *testing.T) {
	classifier := &Classifier{
		Config: &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{ClassifierRules: []config.ClassifierSignalRule{{
				Name:   "risk",
				Type:   "llm",
				Model:  "risk-model",
				Labels: []string{"SAFE", "RISKY"},
			}}},
		}},
		genericClassifiers: map[string]labelClassifier{
			"risk": fakeLabelClassifier{result: labelClassification{
				Scores: map[string]float64{"SAFE": 0, "RISKY": 1},
			}},
		},
	}
	results := &SignalResults{
		SignalConfidences: map[string]float64{},
		SignalValues:      map[string]float64{},
		SignalErrors:      map[string]string{},
		Metrics:           &SignalMetricsCollection{},
	}

	classifier.evaluateGenericClassifierSignals(
		results,
		&sync.Mutex{},
		"delete production",
		map[string]bool{"classifier:risk": true},
		nil,
	)

	if got := results.SignalValues["classifier:risk:RISKY"]; got != 1 {
		t.Fatalf("RISKY score = %v, want 1", got)
	}
	if len(results.MatchedClassifierRules) != 1 ||
		results.MatchedClassifierRules[0] != "risk:RISKY" {
		t.Fatalf("matched classifiers = %v", results.MatchedClassifierRules)
	}
}

func TestGenericClassifierSignalsRespectDecisionScope(t *testing.T) {
	firstCalls := 0
	secondCalls := 0
	classifier := &Classifier{
		Config: &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{ClassifierRules: []config.ClassifierSignalRule{
				{Name: "first", Type: "llm"},
				{Name: "second", Type: "llm"},
			}},
		}},
		genericClassifiers: map[string]labelClassifier{
			"first": fakeLabelClassifier{
				result: labelClassification{Scores: map[string]float64{"YES": 1}},
				calls:  &firstCalls,
			},
			"second": fakeLabelClassifier{
				result: labelClassification{Scores: map[string]float64{"YES": 1}},
				calls:  &secondCalls,
			},
		},
	}
	results := &SignalResults{
		SignalConfidences: map[string]float64{},
		SignalValues:      map[string]float64{},
		SignalErrors:      map[string]string{},
		Metrics:           &SignalMetricsCollection{},
	}

	classifier.evaluateGenericClassifierSignals(
		results,
		&sync.Mutex{},
		"route me",
		map[string]bool{"classifier:first": true},
		nil,
	)

	if firstCalls != 1 || secondCalls != 0 {
		t.Fatalf("classifier calls = (%d, %d), want (1, 0)", firstCalls, secondCalls)
	}
}

func TestGenericClassifierSignalExposesSanitizedErrorCode(t *testing.T) {
	classifier := &Classifier{
		Config: &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{ClassifierRules: []config.ClassifierSignalRule{{
				Name: "risk",
				Type: "llm",
			}}},
		}},
		genericClassifiers: map[string]labelClassifier{
			"risk": fakeLabelClassifier{err: errors.New("sensitive upstream detail")},
		},
	}
	results := &SignalResults{
		SignalConfidences: map[string]float64{},
		SignalValues:      map[string]float64{},
		SignalErrors:      map[string]string{},
		Metrics:           &SignalMetricsCollection{},
	}

	classifier.evaluateGenericClassifierSignals(
		results,
		&sync.Mutex{},
		"route me",
		map[string]bool{"classifier:risk": true},
		nil,
	)

	if results.SignalErrors["classifier:risk"] != genericClassifierErrorCode {
		t.Fatalf("signal error = %q", results.SignalErrors["classifier:risk"])
	}
}

func TestGenericClassifierSignalUsesConfiguredLabelOrderForTies(t *testing.T) {
	classifier := &Classifier{
		Config: &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{ClassifierRules: []config.ClassifierSignalRule{{
				Name:   "risk",
				Type:   "llm",
				Labels: []string{"SAFE", "RISKY"},
			}}},
		}},
		genericClassifiers: map[string]labelClassifier{
			"risk": fakeLabelClassifier{result: labelClassification{
				Scores: map[string]float64{"RISKY": 0.5, "SAFE": 0.5},
			}},
		},
	}
	results := &SignalResults{
		SignalConfidences: map[string]float64{},
		SignalValues:      map[string]float64{},
		SignalErrors:      map[string]string{},
		Metrics:           &SignalMetricsCollection{},
	}

	classifier.evaluateGenericClassifierSignals(
		results,
		&sync.Mutex{},
		"route me",
		map[string]bool{"classifier:risk": true},
		nil,
	)

	if len(results.MatchedClassifierRules) != 1 ||
		results.MatchedClassifierRules[0] != "risk:SAFE" {
		t.Fatalf("matched labels = %v, want configured first label", results.MatchedClassifierRules)
	}
}

func TestGenericClassifierSignalRejectsNonFiniteScores(t *testing.T) {
	classifier := &Classifier{
		Config: &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{ClassifierRules: []config.ClassifierSignalRule{{
				Name:   "risk",
				Type:   "llm",
				Labels: []string{"SAFE", "RISKY"},
			}}},
		}},
		genericClassifiers: map[string]labelClassifier{
			"risk": fakeLabelClassifier{result: labelClassification{
				Scores: map[string]float64{
					"SAFE":  math.NaN(),
					"RISKY": 0.9,
				},
			}},
		},
	}
	results := &SignalResults{
		SignalConfidences: map[string]float64{},
		SignalValues:      map[string]float64{},
		SignalErrors:      map[string]string{},
		Metrics:           &SignalMetricsCollection{},
	}

	classifier.evaluateGenericClassifierSignals(
		results,
		&sync.Mutex{},
		"route me",
		map[string]bool{"classifier:risk": true},
		nil,
	)

	if results.SignalErrors["classifier:risk"] !=
		genericClassifierInvalidScoreCode {
		t.Fatalf("signal errors = %v", results.SignalErrors)
	}
	if len(results.MatchedClassifierRules) != 0 {
		t.Fatalf("matched labels = %v", results.MatchedClassifierRules)
	}
}

func TestGenericClassifierSignalsRunInParallel(t *testing.T) {
	started := make(chan struct{}, 2)
	release := make(chan struct{})
	classifier := &Classifier{
		Config: &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{ClassifierRules: []config.ClassifierSignalRule{
				{Name: "first", Type: "llm", Labels: []string{"NO", "YES"}},
				{Name: "second", Type: "llm", Labels: []string{"NO", "YES"}},
			}},
		}},
		genericClassifiers: map[string]labelClassifier{
			"first": blockingLabelClassifier{
				started: started,
				release: release,
			},
			"second": blockingLabelClassifier{
				started: started,
				release: release,
			},
		},
	}
	results := &SignalResults{
		SignalConfidences: map[string]float64{},
		SignalValues:      map[string]float64{},
		SignalErrors:      map[string]string{},
		Metrics:           &SignalMetricsCollection{},
	}
	done := make(chan struct{})
	go func() {
		classifier.evaluateGenericClassifierSignals(
			results,
			&sync.Mutex{},
			"route me",
			map[string]bool{
				"classifier:first":  true,
				"classifier:second": true,
			},
			context.Background(),
		)
		close(done)
	}()
	for range 2 {
		select {
		case <-started:
		case <-time.After(time.Second):
			t.Fatal("classifier rules did not start in parallel")
		}
	}
	close(release)
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("parallel classifier evaluation did not complete")
	}
}

func TestGenericClassifierSignalsRespectCancellation(t *testing.T) {
	started := make(chan struct{}, 1)
	release := make(chan struct{})
	classifier := &Classifier{
		Config: &config.RouterConfig{IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{ClassifierRules: []config.ClassifierSignalRule{{
				Name: "risk", Type: "llm", Labels: []string{"NO", "YES"},
			}}},
		}},
		genericClassifiers: map[string]labelClassifier{
			"risk": blockingLabelClassifier{
				started: started,
				release: release,
			},
		},
	}
	results := &SignalResults{
		SignalConfidences: map[string]float64{},
		SignalValues:      map[string]float64{},
		SignalErrors:      map[string]string{},
		Metrics:           &SignalMetricsCollection{},
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	classifier.evaluateGenericClassifierSignals(
		results,
		&sync.Mutex{},
		"route me",
		map[string]bool{"classifier:risk": true},
		ctx,
	)

	if results.SignalErrors["classifier:risk"] !=
		genericClassifierCancelledCode {
		t.Fatalf("signal errors = %v", results.SignalErrors)
	}
}
