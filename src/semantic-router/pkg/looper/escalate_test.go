package looper

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestFormatEscalateStreamingResponsePublishesOnlySelectedCandidate(t *testing.T) {
	looper := NewEscalateLooper(&config.LooperConfig{})
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

	resp, err := looper.formatEscalateStreamingResponse(
		&AggregatedResponse{
			Models:          []string{"small", "large"},
			Responses:       []*ModelResponse{first, selected},
			CombinedContent: "selected candidate",
			FinalModel:      "large",
		},
		[]string{"small", "large"},
		2,
		true,
	)
	if err != nil {
		t.Fatalf("format escalate streaming response: %v", err)
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
	if got := strings.Count(string(resp.Body), `"usage":`); got != 1 {
		t.Fatalf("confidence stream usage chunks = %d, want exactly one: %s", got, resp.Body)
	}
	if usageAt, doneAt := strings.Index(string(resp.Body), `"usage":`), strings.Index(string(resp.Body), "data: [DONE]"); usageAt < 0 || doneAt < 0 || usageAt > doneAt {
		t.Fatalf("aggregate usage must precede DONE: %s", resp.Body)
	}
}

func TestFormatEscalateStreamingResponseReplacesUpstreamUsage(t *testing.T) {
	looper := NewEscalateLooper(&config.LooperConfig{})
	rejected := &ModelResponse{
		Model: "small",
		Usage: TokenUsage{PromptTokens: 10, CompletionTokens: 2, TotalTokens: 12},
	}
	selected := &ModelResponse{
		Raw: []byte(
			"data:{\"id\":\"chatcmpl-selected\",\"object\":\"chat.completion.chunk\",\"created\":7,\"model\":\"large\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"},\"finish_reason\":null}],\"usage\":null}\n\n" +
				"data:{\"id\":\"chatcmpl-selected\",\"object\":\"chat.completion.chunk\",\"created\":7,\"model\":\"large\",\"choices\":[],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":3,\"total_tokens\":23}}\n\n" +
				"data:[DONE]\n\n",
		),
		Model:       "large",
		IsStreaming: true,
		Usage:       TokenUsage{PromptTokens: 20, CompletionTokens: 3, TotalTokens: 23},
	}

	resp, err := looper.formatEscalateStreamingResponse(
		&AggregatedResponse{
			Models:          []string{"small", "large"},
			Responses:       []*ModelResponse{selected},
			UsageResponses:  []*ModelResponse{rejected, selected},
			CombinedContent: "answer",
			FinalModel:      "large",
		},
		[]string{"small", "large"},
		2,
		true,
	)
	if err != nil {
		t.Fatalf("format escalate stream: %v", err)
	}
	if got := strings.Count(string(resp.Body), `"usage":`); got != 1 {
		t.Fatalf("usage chunks = %d, want one replacement: %s", got, resp.Body)
	}
	if got := parseStreamingUsage(resp.Body); got != (TokenUsage{PromptTokens: 30, CompletionTokens: 5, TotalTokens: 35}) {
		t.Fatalf("stream usage = %+v, want aggregate", got)
	}
	if strings.Contains(string(resp.Body), `"total_tokens":23`) {
		t.Fatalf("selected upstream usage leaked into final stream: %s", resp.Body)
	}
	if got := strings.Count(string(resp.Body), "[DONE]"); got != 1 {
		t.Fatalf("DONE markers = %d, want exactly one: %s", got, resp.Body)
	}
	if !strings.Contains(string(resp.Body), "data: [DONE]") || strings.Contains(string(resp.Body), "data:[DONE]") {
		t.Fatalf("stream did not normalize the no-space terminal marker: %s", resp.Body)
	}
}

// A backend may name its model differently from the requested name, as a
// snapshot does; the usage chunk must name the same model as the stream.
func TestStreamWithUsageNamesTheStreamModel(t *testing.T) {
	for _, tc := range []struct {
		name string
		body string
		want string
	}{
		{
			name: "backend reports its own name",
			body: "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":7,\"model\":\"large-2026-01-01\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\n" +
				"data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":7,\"model\":\"large-2026-01-01\",\"choices\":[],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n" +
				"data: [DONE]\n\n",
			want: "large-2026-01-01",
		},
		{
			name: "stream names no model",
			body: "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":7,\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
			want: "large",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			body := StreamWithUsage([]byte(tc.body), "large", TokenUsage{PromptTokens: 4, CompletionTokens: 2, TotalTokens: 6})
			var models []string
			for _, line := range strings.Split(string(body), "\n") {
				var chunk struct {
					Model string          `json:"model"`
					Usage json.RawMessage `json:"usage"`
				}
				if payload, ok := strings.CutPrefix(line, "data: {"); ok && json.Unmarshal([]byte("{"+payload), &chunk) == nil && len(chunk.Usage) > 0 {
					models = append(models, chunk.Model)
				}
			}
			if !slices.Equal(models, []string{tc.want}) {
				t.Fatalf("usage chunk models = %v, want [%s]: %s", models, tc.want, body)
			}
		})
	}
}

func TestFormatEscalateStreamingResponseAddsUsageToSimulatedStream(t *testing.T) {
	looper := NewEscalateLooper(&config.LooperConfig{})
	selected := &ModelResponse{
		Model:   "large",
		Content: "simulated answer",
		Usage:   TokenUsage{PromptTokens: 4, CompletionTokens: 2, TotalTokens: 6},
	}
	resp, err := looper.formatEscalateStreamingResponse(
		&AggregatedResponse{
			Models:          []string{"large"},
			Responses:       []*ModelResponse{selected},
			CombinedContent: selected.Content,
			FinalModel:      selected.Model,
		},
		[]string{"large"},
		1,
		true,
	)
	if err != nil {
		t.Fatalf("format simulated confidence stream: %v", err)
	}
	if got := strings.Count(string(resp.Body), `"usage":`); got != 1 {
		t.Fatalf("simulated usage chunks = %d, want one: %s", got, resp.Body)
	}
	if got := parseStreamingUsage(resp.Body); got != selected.Usage {
		t.Fatalf("simulated usage = %+v, want %+v", got, selected.Usage)
	}
}

func TestFormatEscalateStreamingResponseSuppressesUsageUnlessRequested(t *testing.T) {
	looper := NewEscalateLooper(&config.LooperConfig{})
	selected := &ModelResponse{
		Raw: []byte(
			"data: {\"id\":\"chatcmpl-selected\",\"object\":\"chat.completion.chunk\",\"created\":7,\"model\":\"large\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"},\"finish_reason\":null}],\"usage\":null}\n\n" +
				"data: {\"id\":\"chatcmpl-selected\",\"object\":\"chat.completion.chunk\",\"created\":7,\"model\":\"large\",\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2,\"total_tokens\":6}}\n\n" +
				"data: [DONE]\n\n",
		),
		Model:       "large",
		IsStreaming: true,
		Usage:       TokenUsage{PromptTokens: 4, CompletionTokens: 2, TotalTokens: 6},
	}
	resp, err := looper.formatEscalateStreamingResponse(
		&AggregatedResponse{
			Models:          []string{"large"},
			Responses:       []*ModelResponse{selected},
			CombinedContent: "answer",
			FinalModel:      "large",
		},
		[]string{"large"},
		1,
		false,
	)
	if err != nil {
		t.Fatalf("format escalate stream: %v", err)
	}
	if strings.Contains(string(resp.Body), `"usage":`) {
		t.Fatalf("stream emitted usage without client opt-in: %s", resp.Body)
	}
	if resp.Usage != selected.Usage {
		t.Fatalf("internal accounting = %+v, want %+v", resp.Usage, selected.Usage)
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

// The models escalate in modelRefs order unless order names size or cost.
func TestEscalateOrder(t *testing.T) {
	for _, test := range []struct {
		order string
		want  []string
	}{
		{order: "", want: []string{"large", "cheap", "small"}},
		{order: config.EscalateOrderSize, want: []string{"small", "cheap", "large"}},
		{order: config.EscalateOrderCost, want: []string{"cheap", "small", "large"}},
	} {
		t.Run("order="+test.order, func(t *testing.T) {
			server, calls := newEscalateLogprobBackend(t, map[string]int{"large": 1, "cheap": 1, "small": 1}, true)
			defer server.Close()
			request := escalateRequest(config.EscalateOnErrorLastAttempt, config.ConfidenceMethodAverageLogprob)
			// No answer reaches the threshold, so every model is called.
			request.Algorithm.Escalate.Accept.Confidence.GTE = gte(0.99)
			request.ModelRefs = []config.ModelRef{{Model: "large"}, {Model: "cheap"}, {Model: "small"}}
			request.ModelParams = map[string]config.ModelParams{
				"large": {ParamSize: "70b", Pricing: config.ModelPricing{PromptPer1M: 30}},
				"cheap": {ParamSize: "8b", Pricing: config.ModelPricing{PromptPer1M: 0.1}},
				"small": {ParamSize: "1b", Pricing: config.ModelPricing{PromptPer1M: 1}},
			}
			request.Algorithm.Escalate.Order = test.order
			if _, err := NewEscalateLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), request); err != nil {
				t.Fatalf("Execute() error = %v", err)
			}
			if !slices.Equal(*calls, test.want) {
				t.Fatalf("backend attempt order = %v, want %v", *calls, test.want)
			}
		})
	}
}

// order: size and order: cost read the sizes and prices a loaded configuration
// carries, built-in catalog sizes such as "397B (17B active)" and "1.6T (49B
// active)" included, and try a model with neither last.
func TestEscalateOrderTriesUnknownSizeAndPriceLast(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(`version: v0.5
listeners: []
providers:
  defaults:
    model: small
  models:
    - name: small
      catalog: qwen/qwen3.6-27b
      pricing: {currency: USD, prompt_per_1m: 3}
      backend_refs: [{endpoint: 127.0.0.1:8000, provider: vllm}]
    - name: huge
      catalog: qwen/qwen3.5-397b-a17b
      pricing: {currency: USD, prompt_per_1m: 0.1}
      backend_refs: [{endpoint: 127.0.0.1:8000, provider: vllm}]
    - name: giant
      catalog: deepseek/deepseek-v4-pro
      pricing: {currency: USD, prompt_per_1m: 1}
      backend_refs: [{endpoint: 127.0.0.1:8000, provider: vllm}]
    - name: unknown
      backend_refs: [{endpoint: 127.0.0.1:8000, provider: vllm}]
recipes:
  - name: default
    decisions:
      - name: escalate
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: unknown}, {model: giant}, {model: huge}, {model: small}]
        algorithm:
          type: escalate
          escalate:
            order: size
            accept: {confidence: {method: avg_logprob, gte: 0.99}}
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	for _, test := range []struct {
		order string
		want  []string
	}{
		{order: config.EscalateOrderSize, want: []string{"small", "huge", "giant", "unknown"}},
		{order: config.EscalateOrderCost, want: []string{"huge", "giant", "small", "unknown"}},
	} {
		t.Run(test.order, func(t *testing.T) {
			server, calls := newEscalateLogprobBackend(t, map[string]int{"small": 1, "huge": 1, "giant": 1, "unknown": 1}, true)
			defer server.Close()
			decision := cfg.Decisions[0]
			escalate := *decision.Algorithm.Escalate
			escalate.Order = test.order
			request := escalateRequest(config.EscalateOnErrorLastAttempt, "")
			request.ModelRefs, request.ModelParams = decision.ModelRefs, cfg.ModelConfig
			request.Algorithm.Escalate = &escalate
			// No answer reaches the threshold, so every model is called.
			if _, err := NewEscalateLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), request); err != nil {
				t.Fatalf("Execute() error = %v", err)
			}
			if !slices.Equal(*calls, test.want) {
				t.Fatalf("backend attempt order = %v, want %v", *calls, test.want)
			}
		})
	}
}

func TestConfidenceEvaluator_Creation(t *testing.T) {
	tests := []struct {
		name         string
		method       string
		wantLogprobs bool
		wantSelfVer  bool
	}{
		{"avg_logprob", config.ConfidenceMethodAverageLogprob, true, false},
		{"margin", config.ConfidenceMethodMargin, true, false},
		{"hybrid", config.ConfidenceMethodHybrid, true, false},
		{"self_verify", config.ConfidenceMethodSelfVerify, false, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			eval := NewConfidenceEvaluator(&config.EscalateConfidence{Method: tt.method, GTE: gte(0.6)})

			if eval.NeedsLogprobs() != tt.wantLogprobs {
				t.Errorf("NeedsLogprobs() = %v, want %v", eval.NeedsLogprobs(), tt.wantLogprobs)
			}
			if eval.IsSelfVerify() != tt.wantSelfVer {
				t.Errorf("IsSelfVerify() = %v, want %v", eval.IsSelfVerify(), tt.wantSelfVer)
			}
			if eval.Threshold != 0.6 {
				t.Errorf("Threshold = %v, want gte 0.6", eval.Threshold)
			}
		})
	}
}

// A candidate streams to the client only when no token logprobs are read
// from it: a self_verify answer, never a logprob or System One check.
func TestEscalateCandidatesStreamOnlyWithoutLogprobs(t *testing.T) {
	for _, test := range []struct {
		accept *config.EscalateAccept
		want   bool
	}{
		{accept: &config.EscalateAccept{Confidence: &config.EscalateConfidence{Method: config.ConfidenceMethodAverageLogprob, GTE: gte(0.5)}}, want: false},
		{accept: &config.EscalateAccept{Confidence: &config.EscalateConfidence{Method: config.ConfidenceMethodMargin, GTE: gte(0.5)}}, want: false},
		{accept: &config.EscalateAccept{Confidence: &config.EscalateConfidence{Method: config.ConfidenceMethodSelfVerify, GTE: gte(0.5)}}, want: true},
		{accept: &config.EscalateAccept{SystemOne: &config.EscalateAnswerCheck{GTE: gte(0.5)}}, want: false},
	} {
		acceptance := newEscalateAcceptance(test.accept)
		if got := acceptance.streamsCandidates(true); got != test.want {
			t.Fatalf("streamsCandidates(%s) = %v, want %v", acceptance.kind(), got, test.want)
		}
	}
}

func TestEscalateStreamUsageRequested(t *testing.T) {
	if escalateStreamUsageRequested(nil) {
		t.Fatal("nil request unexpectedly requested usage")
	}
	request := &Request{OriginalRequest: &openai.ChatCompletionNewParams{}}
	if escalateStreamUsageRequested(request) {
		t.Fatal("omitted stream_options unexpectedly requested usage")
	}
	request.OriginalRequest.StreamOptions = openai.ChatCompletionStreamOptionsParam{
		IncludeUsage: openai.Bool(true),
	}
	if !escalateStreamUsageRequested(request) {
		t.Fatal("stream_options.include_usage=true was not honored")
	}
}

func TestEvaluate_Logprobs(t *testing.T) {
	eval := &ConfidenceEvaluator{
		Method:    "avg_logprob",
		Threshold: 0.5,
	}

	resp := &ModelResponse{
		Content:        "Test response",
		Tokens:         []string{"Test", " response"},
		Logprobs:       []float64{-0.2, -0.4},
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
		Content:                "Test response",
		Tokens:                 []string{"Test", " response"},
		Logprobs:               []float64{-0.4, -0.6},
		TopLogprobMargins:      []float64{1.5, 2.5},
		AverageLogprob:         -0.5,
		AverageMargin:          2.0,
		MarginEvidenceComplete: true,
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

func TestEvaluate_UsesFilteredZeroEvidence(t *testing.T) {
	resp := &ModelResponse{
		Tokens: []string{
			`{"name":"x","arguments":{"value":"`,
			`semantic`,
			`"}}`,
		},
		Logprobs:               []float64{-3, 0, -3},
		TopLogprobMargins:      []float64{2, 0, 2},
		AverageLogprob:         -2,
		AverageMargin:          4.0 / 3.0,
		MarginEvidenceComplete: true,
	}
	ApplyTokenFilter(resp, "tool_call_args")
	if resp.FilteredTokenCount != 1 || resp.FilteredAverageLogprob != 0 || resp.FilteredAverageMargin != 0 {
		t.Fatalf("filtered evidence = count %d logprob %v margin %v, want one real zero-valued token",
			resp.FilteredTokenCount, resp.FilteredAverageLogprob, resp.FilteredAverageMargin)
	}

	logprobEvaluator := &ConfidenceEvaluator{
		Method:      "avg_logprob",
		Threshold:   0.9,
		TokenFilter: "tool_call_args",
	}
	confidence, accepted, err := logprobEvaluator.evaluate(resp)
	if err != nil || confidence != 1 || !accepted {
		t.Fatalf("filtered zero logprob evaluation = confidence %v accepted %v err %v, want 1/true/nil", confidence, accepted, err)
	}

	marginEvaluator := &ConfidenceEvaluator{
		Method:      "margin",
		Threshold:   0.1,
		TokenFilter: "tool_call_args",
	}
	confidence, accepted, err = marginEvaluator.evaluate(resp)
	if err != nil || confidence != 0 || accepted {
		t.Fatalf("filtered zero margin evaluation = confidence %v accepted %v err %v, want 0/false/nil", confidence, accepted, err)
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

func TestConfidenceEvaluatorRequiresTokenLogprobs(t *testing.T) {
	for _, method := range []string{"avg_logprob", "margin", "hybrid"} {
		t.Run(method, func(t *testing.T) {
			evaluator := NewConfidenceEvaluator(&config.EscalateConfidence{
				Method: method,
				GTE:    gte(0.5),
			})

			response := &ModelResponse{
				Content:        "backend omitted logprobs",
				AverageLogprob: 0,
			}
			confidence, accepted, err := evaluator.evaluate(response)
			if err == nil || !strings.Contains(err.Error(), "token logprobs") {
				t.Fatalf("evaluate() error = %v, want missing token logprobs", err)
			}
			if confidence != 0 || accepted {
				t.Fatalf("evaluate() = (%v, %v), want zero confidence and rejection", confidence, accepted)
			}
			if confidence, accepted := evaluator.Evaluate(response); confidence != 0 || accepted {
				t.Fatalf("Evaluate() = (%v, %v), want zero confidence and rejection", confidence, accepted)
			}
		})
	}
}

func TestConfidenceEvaluatorAcceptsPresentZeroLogprob(t *testing.T) {
	evaluator := NewConfidenceEvaluator(&config.EscalateConfidence{
		Method: "avg_logprob",
		GTE:    gte(0.9),
	})
	confidence, accepted, err := evaluator.evaluate(&ModelResponse{
		Content:        "certain answer",
		Tokens:         []string{"certain"},
		Logprobs:       []float64{0},
		AverageLogprob: 0,
	})
	if err != nil {
		t.Fatalf("Evaluate() error = %v", err)
	}
	if confidence != 1 || !accepted {
		t.Fatalf("Evaluate() = (%v, %v), want (1, true)", confidence, accepted)
	}
}

func TestConfidenceEvaluatorRejectsFabricatedMarginWithoutAlternatives(t *testing.T) {
	response := &ModelResponse{
		Content:                "backend supplied chosen-token logprobs only",
		Tokens:                 []string{"answer"},
		Logprobs:               []float64{-0.1},
		TopLogprobMargins:      []float64{0},
		AverageLogprob:         -0.1,
		AverageMargin:          2.0,
		MarginEvidenceComplete: false,
	}
	for _, method := range []string{"margin", "hybrid"} {
		t.Run(method, func(t *testing.T) {
			evaluator := NewConfidenceEvaluator(&config.EscalateConfidence{
				Method: method,
				GTE:    gte(0.5),
			})
			confidence, accepted, err := evaluator.evaluate(response)
			if err == nil || !strings.Contains(err.Error(), "top-logprob alternatives") {
				t.Fatalf("evaluate() error = %v, want missing alternatives", err)
			}
			if confidence != 0 || accepted {
				t.Fatalf("evaluate() = (%v, %v), want no confidence", confidence, accepted)
			}
		})
	}
}

func TestParseNonStreamingResponseDoesNotInventMissingMargin(t *testing.T) {
	body := []byte(`{
		"id":"chatcmpl-margin",
		"object":"chat.completion",
		"model":"backend-model",
		"choices":[{
			"index":0,
			"message":{"role":"assistant","content":"answer"},
			"finish_reason":"stop",
			"logprobs":{"content":[{
				"token":"answer",
				"logprob":-0.1,
				"top_logprobs":[{"token":"answer","logprob":-0.1}]
			}]}
		}]
	}`)
	response, err := (&Client{}).parseNonStreamingResponse(body, "logical-model")
	if err != nil {
		t.Fatalf("parse response: %v", err)
	}
	if response.MarginEvidenceComplete {
		t.Fatal("one chosen-token entry was treated as an alternative margin")
	}
	if response.AverageMargin != 0 || !slices.Equal(response.TopLogprobMargins, []float64{0}) {
		t.Fatalf("margin analysis = average %v margins %v, want explicit no-evidence zero", response.AverageMargin, response.TopLogprobMargins)
	}
}
