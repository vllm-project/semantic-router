package classification

import (
	"math"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

func TestJailbreakRiskScore(t *testing.T) {
	// Binary mapping: index 0 = benign, index 1 = jailbreak.
	mapping := &JailbreakMapping{
		LabelToIdx: map[string]int{"benign": 0, "jailbreak": 1},
		IdxToLabel: map[string]string{"0": "benign", "1": "jailbreak"},
	}

	tests := []struct {
		name    string
		mapping *JailbreakMapping
		result  candle_binding.ClassResultWithProbs
		want    float32
	}{
		{
			name:    "distribution present, argmax is jailbreak",
			mapping: mapping,
			result:  candle_binding.ClassResultWithProbs{Class: 1, Confidence: 0.98, Probabilities: []float32{0.02, 0.98}},
			want:    0.98,
		},
		{
			// The bug from #2591: argmax is benign with high confidence; risk_score must be
			// P(jailbreak) = 0.0043, NOT the benign confidence 0.9957.
			name:    "distribution present, argmax is benign with high confidence",
			mapping: mapping,
			result:  candle_binding.ClassResultWithProbs{Class: 0, Confidence: 0.9957, Probabilities: []float32{0.9957, 0.0043}},
			want:    0.0043,
		},
		{
			name:    "no distribution, predicted jailbreak falls back to confidence",
			mapping: mapping,
			result:  candle_binding.ClassResultWithProbs{Class: 1, Confidence: 0.98},
			want:    0.98,
		},
		{
			name:    "no distribution, predicted benign falls back to 1-confidence",
			mapping: mapping,
			result:  candle_binding.ClassResultWithProbs{Class: 0, Confidence: 0.9957},
			want:    0.0043,
		},
		{
			name:    "distribution present but jailbreak label absent falls back to 1-confidence",
			mapping: &JailbreakMapping{LabelToIdx: map[string]int{"benign": 0}, IdxToLabel: map[string]string{"0": "benign"}},
			result:  candle_binding.ClassResultWithProbs{Class: 0, Confidence: 0.9957, Probabilities: []float32{0.9957, 0.0043}},
			want:    0.0043,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := jailbreakRiskScore(tt.mapping, tt.result)
			if math.Abs(float64(got-tt.want)) > 1e-6 {
				t.Errorf("jailbreakRiskScore() = %v, want %v", got, tt.want)
			}
		})
	}
}
