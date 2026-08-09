package looper

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestFormatConfidenceStreamingResponsePublishesOnlySelectedCandidate(t *testing.T) {
	looper := NewConfidenceLooper(&config.LooperConfig{})
	first := &ModelResponse{
		Raw:         []byte("data: first candidate\n\ndata: [DONE]\n\n"),
		Model:       "small",
		IsStreaming: true,
		Usage:       TokenUsage{PromptTokens: 10, CompletionTokens: 2, TotalTokens: 12},
	}
	selected := &ModelResponse{
		Raw:         []byte("data: selected candidate\n\ndata: [DONE]\n\n"),
		Model:       "large",
		IsStreaming: true,
		Usage:       TokenUsage{PromptTokens: 20, CompletionTokens: 3, TotalTokens: 23},
	}

	resp, err := looper.formatConfidenceStreamingResponse(
		&AggregatedResponse{
			Models:          []string{"small", "large"},
			Responses:       []*ModelResponse{first, selected},
			CombinedContent: "selected candidate",
			FinalModel:      "large",
		},
		[]string{"small", "large"},
		2,
	)
	if err != nil {
		t.Fatalf("format confidence streaming response: %v", err)
	}
	if strings.Contains(string(resp.Body), "first candidate") {
		t.Fatalf("confidence stream leaked a rejected candidate: %s", resp.Body)
	}
	if !strings.Contains(string(resp.Body), "selected candidate") {
		t.Fatalf("confidence stream omitted the selected candidate: %s", resp.Body)
	}
	if resp.Usage.TotalTokens != 35 {
		t.Fatalf("confidence usage = %+v, want aggregate total 35", resp.Usage)
	}
}

func TestSortModelRefsBySize(t *testing.T) {
	modelParams := map[string]config.ModelParams{
		"large":  {ParamSize: "70b"},
		"medium": {ParamSize: "13b"},
		"small":  {ParamSize: "7b"},
	}

	models := []config.ModelRef{
		{Model: "large"},
		{Model: "small"},
		{Model: "medium"},
	}

	sorted := sortModelRefsBySize(models, modelParams)

	// Should sort by size (smallest first): small(7b) < medium(13b) < large(70b)
	if len(sorted) != 3 {
		t.Errorf("Expected 3 models, got %d", len(sorted))
	}

	if sorted[0].Model != "small" {
		t.Errorf("Expected small first (7b), got %s", sorted[0].Model)
	}
	if sorted[1].Model != "medium" {
		t.Errorf("Expected medium second (13b), got %s", sorted[1].Model)
	}
	if sorted[2].Model != "large" {
		t.Errorf("Expected large third (70b), got %s", sorted[2].Model)
	}
}

func TestSortModelRefsByCost(t *testing.T) {
	modelParams := map[string]config.ModelParams{
		"expensive": {Pricing: config.ModelPricing{PromptPer1M: 30.0}},
		"medium":    {Pricing: config.ModelPricing{PromptPer1M: 1.0}},
		"cheap":     {Pricing: config.ModelPricing{PromptPer1M: 0.1}},
	}

	models := []config.ModelRef{
		{Model: "expensive"},
		{Model: "cheap"},
		{Model: "medium"},
	}

	sorted := sortModelRefsByCost(models, modelParams)

	// Should sort by cost (cheapest first)
	if len(sorted) != 3 {
		t.Errorf("Expected 3 models, got %d", len(sorted))
	}

	if sorted[0].Model != "cheap" {
		t.Errorf("Expected cheap first (0.1), got %s", sorted[0].Model)
	}
	if sorted[1].Model != "medium" {
		t.Errorf("Expected medium second (1.0), got %s", sorted[1].Model)
	}
	if sorted[2].Model != "expensive" {
		t.Errorf("Expected expensive third (30.0), got %s", sorted[2].Model)
	}
}

func TestConfidenceEvaluator_Creation(t *testing.T) {
	tests := []struct {
		name         string
		method       string
		wantLogprobs bool
		wantSelfVer  bool
		wantAutoMix  bool
	}{
		{"avg_logprob", "avg_logprob", true, false, false},
		{"margin", "margin", true, false, false},
		{"hybrid", "hybrid", true, false, false},
		{"self_verify", "self_verify", false, true, false},
		{"automix_entailment", MethodAutoMixEntailment, false, false, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &config.ConfidenceAlgorithmConfig{
				ConfidenceMethod: tt.method,
			}
			eval := NewConfidenceEvaluator(cfg)

			if eval.NeedsLogprobs() != tt.wantLogprobs {
				t.Errorf("NeedsLogprobs() = %v, want %v", eval.NeedsLogprobs(), tt.wantLogprobs)
			}

			if eval.IsSelfVerify() != tt.wantSelfVer {
				t.Errorf("IsSelfVerify() = %v, want %v", eval.IsSelfVerify(), tt.wantSelfVer)
			}

			if eval.IsAutoMixEntailment() != tt.wantAutoMix {
				t.Errorf("IsAutoMixEntailment() = %v, want %v", eval.IsAutoMixEntailment(), tt.wantAutoMix)
			}
		})
	}
}

func TestConfidenceModelCallStreamingRequiresNoLogprobs(t *testing.T) {
	for _, test := range []struct {
		method string
		want   bool
	}{
		{method: "avg_logprob", want: false},
		{method: "margin", want: false},
		{method: "hybrid", want: false},
		{method: "self_verify", want: true},
		{method: MethodAutoMixEntailment, want: true},
	} {
		t.Run(test.method, func(t *testing.T) {
			evaluator := NewConfidenceEvaluator(&config.ConfidenceAlgorithmConfig{
				ConfidenceMethod: test.method,
			})
			if got := confidenceModelCallStreaming(true, evaluator); got != test.want {
				t.Fatalf("confidenceModelCallStreaming(%q) = %v, want %v", test.method, got, test.want)
			}
		})
	}
}

func TestEvaluate_Logprobs(t *testing.T) {
	eval := &ConfidenceEvaluator{
		Method:    "avg_logprob",
		Threshold: 0.5,
	}

	resp := &ModelResponse{
		Content:        "Test response",
		AverageLogprob: -0.3, // Close to 0 = high confidence
		AverageMargin:  0.3,
	}

	confidence, _ := eval.Evaluate(resp)

	if confidence < 0 || confidence > 1 {
		t.Errorf("Confidence out of range: %f", confidence)
	}

	t.Logf("Logprob -0.3 normalized to confidence: %f", confidence)
}

func TestEvaluate_Margin(t *testing.T) {
	eval := &ConfidenceEvaluator{
		Method:    "margin",
		Threshold: 0.1,
	}

	resp := &ModelResponse{
		Content:        "Test response",
		AverageLogprob: -0.5,
		AverageMargin:  2.0,
	}

	confidence, meets := eval.Evaluate(resp)

	t.Logf("Margin 2.0 normalized to confidence: %f, meets threshold 0.1: %v", confidence, meets)

	if confidence < 0.3 {
		t.Errorf("Margin 2.0 should normalize to ~0.49, got %f", confidence)
	}

	if !meets {
		t.Errorf("Should meet threshold 0.1 with normalized confidence %f", confidence)
	}
}

func TestNormalizeMargin(t *testing.T) {
	tests := []struct {
		margin    float64
		minExpect float64
		maxExpect float64
	}{
		{0.0, 0.0, 0.0},
		{0.5, 0.1, 0.2},
		{2.0, 0.4, 0.6},
		{5.0, 0.8, 1.0},
		{10.0, 0.95, 1.0},
	}

	for _, tt := range tests {
		result := normalizeMargin(tt.margin)
		if result < tt.minExpect || result > tt.maxExpect {
			t.Errorf("normalizeMargin(%f) = %f, want between %f and %f",
				tt.margin, result, tt.minExpect, tt.maxExpect)
		}
	}
}
