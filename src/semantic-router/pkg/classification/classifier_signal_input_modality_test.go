package classification

import (
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func inputModalityTestClassifier(rules ...config.InputModalityRule) *Classifier {
	return &Classifier{Config: &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{InputModalityRules: rules},
		},
	}}
}

func newInputModalityTestResults() *SignalResults {
	return &SignalResults{
		SignalConfidences: map[string]float64{},
		SignalValues:      map[string]float64{},
		Metrics:           &SignalMetricsCollection{},
	}
}

func TestInputModalitySignalMatchesPresentModalities(t *testing.T) {
	classifier := inputModalityTestClassifier(
		config.InputModalityRule{Name: "text_input", Modality: config.InputModalityText},
		config.InputModalityRule{Name: "image_input", Modality: config.InputModalityImage},
		config.InputModalityRule{Name: "audio_input", Modality: config.InputModalityAudio},
		config.InputModalityRule{Name: "video_input", Modality: config.InputModalityVideo},
	)
	results := newInputModalityTestResults()

	classifier.evaluateInputModalitySignal(
		results,
		&sync.Mutex{},
		RequestFacts{InputModality: InputModalityFacts{
			TextContentCount:  1,
			ImageContentCount: 2,
		}},
		map[string]bool{
			"input_modality:text_input":  true,
			"input_modality:image_input": true,
			"input_modality:audio_input": true,
			"input_modality:video_input": true,
		},
	)

	matched := map[string]bool{}
	for _, name := range results.MatchedInputModalityRules {
		matched[name] = true
	}
	if !matched["text_input"] || !matched["image_input"] {
		t.Fatalf("matched input-modality rules = %v, want text_input and image_input", results.MatchedInputModalityRules)
	}
	if matched["audio_input"] || matched["video_input"] {
		t.Fatalf("matched input-modality rules = %v, absent modalities must not match", results.MatchedInputModalityRules)
	}
	if results.SignalValues["input_modality:image_input"] != 2 {
		t.Fatalf("signal values = %v, want image count 2", results.SignalValues)
	}
	if results.SignalValues["input_modality:audio_input"] != 0 {
		t.Fatalf("signal values = %v, want published zero for audio", results.SignalValues)
	}
	if results.SignalConfidences["input_modality:image_input"] != 1.0 {
		t.Fatalf("signal confidences = %v, want 1.0 for image", results.SignalConfidences)
	}
	if results.Metrics.InputModality.Confidence != 1.0 {
		t.Fatalf("input-modality confidence = %v, want 1.0", results.Metrics.InputModality.Confidence)
	}
}

func TestInputModalitySignalNoMatchOnEmptyFacts(t *testing.T) {
	classifier := inputModalityTestClassifier(
		config.InputModalityRule{Name: "image_input", Modality: config.InputModalityImage},
	)
	results := newInputModalityTestResults()

	classifier.evaluateInputModalitySignal(
		results,
		&sync.Mutex{},
		RequestFacts{},
		map[string]bool{"input_modality:image_input": true},
	)

	if len(results.MatchedInputModalityRules) != 0 {
		t.Fatalf("matched input-modality rules = %v, want none", results.MatchedInputModalityRules)
	}
	if results.Metrics.InputModality.Confidence != 0 {
		t.Fatalf("input-modality confidence = %v, want 0", results.Metrics.InputModality.Confidence)
	}
}

func TestInputModalitySignalRespectsRuleScope(t *testing.T) {
	classifier := inputModalityTestClassifier(
		config.InputModalityRule{Name: "image_input", Modality: config.InputModalityImage},
		config.InputModalityRule{Name: "audio_input", Modality: config.InputModalityAudio},
	)
	results := newInputModalityTestResults()

	classifier.evaluateInputModalitySignal(
		results,
		&sync.Mutex{},
		RequestFacts{InputModality: InputModalityFacts{ImageContentCount: 1, AudioContentCount: 1}},
		map[string]bool{"input_modality:audio_input": true},
	)

	if len(results.MatchedInputModalityRules) != 1 || results.MatchedInputModalityRules[0] != "audio_input" {
		t.Fatalf("matched input-modality rules = %v, want only the in-scope audio_input", results.MatchedInputModalityRules)
	}
	if _, published := results.SignalValues["input_modality:image_input"]; published {
		t.Fatalf("signal values = %v, out-of-scope rule must not publish", results.SignalValues)
	}
}
