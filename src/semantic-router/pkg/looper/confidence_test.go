package looper

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/openai/openai-go"

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
		true,
	)
	if err != nil {
		t.Fatalf("format confidence streaming response: %v", err)
	}
	body := wireResponseForTest(t, resp)
	if strings.Contains(string(body), "first candidate") {
		t.Fatalf("confidence stream leaked a rejected candidate: %s", body)
	}
	if !strings.Contains(string(body), "selected candidate") {
		t.Fatalf("confidence stream omitted the selected candidate: %s", body)
	}
	if resp.Usage.TotalTokens != 35 {
		t.Fatalf("confidence usage = %+v, want aggregate total 35", resp.Usage)
	}
	if got := strings.Count(string(body), `"usage":`); got != 1 {
		t.Fatalf("confidence stream usage chunks = %d, want exactly one: %s", got, body)
	}
	if usageAt, doneAt := strings.Index(string(body), `"usage":`), strings.Index(string(body), "data: [DONE]"); usageAt < 0 || doneAt < 0 || usageAt > doneAt {
		t.Fatalf("aggregate usage must precede DONE: %s", body)
	}
}

func TestFormatConfidenceStreamingResponseReplacesUpstreamUsage(t *testing.T) {
	looper := NewConfidenceLooper(&config.LooperConfig{})
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

	resp, err := looper.formatConfidenceStreamingResponse(
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
		t.Fatalf("format confidence stream: %v", err)
	}
	body := wireResponseForTest(t, resp)
	if got := strings.Count(string(body), `"usage":`); got != 1 {
		t.Fatalf("usage chunks = %d, want one replacement: %s", got, body)
	}
	if got := parseStreamingUsage(body); got != (TokenUsage{PromptTokens: 30, CompletionTokens: 5, TotalTokens: 35}) {
		t.Fatalf("stream usage = %+v, want aggregate", got)
	}
	if strings.Contains(string(body), `"total_tokens":23`) {
		t.Fatalf("selected upstream usage leaked into final stream: %s", body)
	}
	if got := strings.Count(string(body), "[DONE]"); got != 1 {
		t.Fatalf("DONE markers = %d, want exactly one: %s", got, body)
	}
	if !strings.Contains(string(body), "data: [DONE]") || strings.Contains(string(body), "data:[DONE]") {
		t.Fatalf("stream did not normalize the no-space terminal marker: %s", body)
	}
}

func TestFormatConfidenceStreamingResponseAddsUsageToSimulatedStream(t *testing.T) {
	looper := NewConfidenceLooper(&config.LooperConfig{})
	selected := &ModelResponse{
		Model:   "large",
		Content: "simulated answer",
		Usage:   TokenUsage{PromptTokens: 4, CompletionTokens: 2, TotalTokens: 6},
	}
	resp, err := looper.formatConfidenceStreamingResponse(
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
	body := wireResponseForTest(t, resp)
	if got := strings.Count(string(body), `"usage":`); got != 1 {
		t.Fatalf("simulated usage chunks = %d, want one: %s", got, body)
	}
	if got := parseStreamingUsage(body); got != selected.Usage {
		t.Fatalf("simulated usage = %+v, want %+v", got, selected.Usage)
	}
}

func TestFormatConfidenceStreamingResponseSuppressesUsageUnlessRequested(t *testing.T) {
	looper := NewConfidenceLooper(&config.LooperConfig{})
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
	resp, err := looper.formatConfidenceStreamingResponse(
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
		t.Fatalf("format confidence stream: %v", err)
	}
	body := wireResponseForTest(t, resp)
	if strings.Contains(string(body), `"usage":`) {
		t.Fatalf("stream emitted usage without client opt-in: %s", body)
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

func TestConfidenceDeclaredEscalationPreservesModelRefOrder(t *testing.T) {
	server, calls := newConfidenceLogprobBackend(t, map[string]int{
		"large": 0,
		"small": 1,
	})
	defer server.Close()

	request := confidenceLogprobRequest("skip")
	request.ModelRefs = []config.ModelRef{{Model: "large"}, {Model: "small"}}
	request.ModelParams = map[string]config.ModelParams{
		"large": {ParamSize: "70b"},
		"small": {ParamSize: "1b"},
	}
	request.Algorithm.Confidence.EscalationOrder = config.ConfidenceEscalationOrderDeclared
	looper := NewConfidenceLooper(&config.LooperConfig{Endpoint: server.URL})

	_, err := looper.Execute(context.Background(), request)
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if !slices.Equal(*calls, []string{"large", "small"}) {
		t.Fatalf("backend attempt order = %v, want declared [large small]", *calls)
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

func TestConfidenceStreamUsageRequested(t *testing.T) {
	if streamUsageRequested(nil) {
		t.Fatal("nil request unexpectedly requested usage")
	}
	request := &Request{executionRequest: &openai.ChatCompletionNewParams{}}
	if streamUsageRequested(request) {
		t.Fatal("omitted stream_options unexpectedly requested usage")
	}
	request.executionRequest.StreamOptions = openai.ChatCompletionStreamOptionsParam{
		IncludeUsage: openai.Bool(true),
	}
	if !streamUsageRequested(request) {
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
			evaluator := NewConfidenceEvaluator(&config.ConfidenceAlgorithmConfig{
				ConfidenceMethod: method,
				Threshold:        0.5,
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
	evaluator := NewConfidenceEvaluator(&config.ConfidenceAlgorithmConfig{
		ConfidenceMethod: "avg_logprob",
		Threshold:        0.9,
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
			evaluator := NewConfidenceEvaluator(&config.ConfidenceAlgorithmConfig{
				ConfidenceMethod: method,
				Threshold:        0.5,
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

func TestConfidenceLooperSkipsMissingLogprobsAndEscalates(t *testing.T) {
	server, calls := newConfidenceLogprobBackend(t, map[string]int{
		"small": 0,
		"large": 1,
	})
	defer server.Close()

	response, err := NewConfidenceLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(
		context.Background(),
		confidenceLogprobRequest("skip"),
	)
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if response.Model != "large" {
		t.Fatalf("response model = %q, want %q", response.Model, "large")
	}
	if !slices.Equal(*calls, []string{"small", "large"}) {
		t.Fatalf("backend calls = %v, want [small large]", *calls)
	}
	if response.Usage.TotalTokens != 6 {
		t.Fatalf("response usage = %+v, want both paid attempts", response.Usage)
	}
	var completion struct {
		Usage TokenUsage `json:"usage"`
	}
	if err := json.Unmarshal(wireResponseForTest(t, response), &completion); err != nil {
		t.Fatalf("decode response body: %v", err)
	}
	if completion.Usage.TotalTokens != 6 {
		t.Fatalf("body usage = %+v, want both paid attempts", completion.Usage)
	}
}

func TestConfidenceLooperFailsFastOnMissingLogprobs(t *testing.T) {
	server, calls := newConfidenceLogprobBackend(t, map[string]int{
		"small": 0,
		"large": 1,
	})
	defer server.Close()

	_, err := NewConfidenceLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(
		context.Background(),
		confidenceLogprobRequest("fail"),
	)
	if err == nil || !strings.Contains(err.Error(), "token logprobs") {
		t.Fatalf("Execute() error = %v, want missing token logprobs", err)
	}
	if !slices.Equal(*calls, []string{"small"}) {
		t.Fatalf("backend calls = %v, want fail-fast [small]", *calls)
	}
	evidence, ok := ExecutionEvidenceFromError(err)
	if !ok {
		t.Fatalf("Execute() error type = %T, want partial confidence evidence", err)
	}
	if !slices.Equal(evidence.ModelsUsed, []string{"small"}) || evidence.Iterations != 1 {
		t.Fatalf("partial evidence = %+v, want one completed small-model call", evidence)
	}
	if evidence.Usage != (TokenUsage{PromptTokens: 2, CompletionTokens: 1, TotalTokens: 3}) {
		t.Fatalf("partial usage = %+v, want paid first call", evidence.Usage)
	}
	if strings.Contains(err.Error(), "small answer") {
		t.Fatalf("partial execution error leaked candidate content: %v", err)
	}
}

func TestConfidenceLooperFailsWhenEveryCandidateOmitsLogprobs(t *testing.T) {
	server, calls := newConfidenceLogprobBackend(t, map[string]int{
		"small": 0,
		"large": 0,
	})
	defer server.Close()

	_, err := NewConfidenceLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(
		context.Background(),
		confidenceLogprobRequest("skip"),
	)
	if err == nil || !strings.Contains(err.Error(), "all models failed") || !strings.Contains(err.Error(), "token logprobs") {
		t.Fatalf("Execute() error = %v, want terminal missing-logprobs failure", err)
	}
	if !slices.Equal(*calls, []string{"small", "large"}) {
		t.Fatalf("backend calls = %v, want [small large]", *calls)
	}
	evidence, ok := ExecutionEvidenceFromError(err)
	if !ok || evidence.Iterations != 2 || evidence.Usage.TotalTokens != 6 || !slices.Equal(evidence.ModelsUsed, []string{"small", "large"}) {
		t.Fatalf("partial evidence = %+v (present=%v), want both paid attempts", evidence, ok)
	}
}

func TestConfidenceLooperTreatsMissingMarginAlternativesAsOnError(t *testing.T) {
	for _, test := range []struct {
		name      string
		onError   string
		wantCalls []string
		wantModel string
		wantError bool
	}{
		{name: "skip escalates", onError: "skip", wantCalls: []string{"small", "large"}, wantModel: "large"},
		{name: "fail stops", onError: "fail", wantCalls: []string{"small"}, wantError: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			server, calls := newConfidenceLogprobBackend(t, map[string]int{
				"small": 1,
				"large": 2,
			})
			defer server.Close()

			response, err := NewConfidenceLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(
				context.Background(),
				confidenceRequest(test.onError, "margin"),
			)
			assertMissingMarginResult(t, response, err, test.wantError, test.wantModel)
			if !slices.Equal(*calls, test.wantCalls) {
				t.Fatalf("backend calls = %v, want %v", *calls, test.wantCalls)
			}
		})
	}
}

func assertMissingMarginResult(t *testing.T, response *Response, err error, wantError bool, wantModel string) {
	t.Helper()
	if wantError {
		if err == nil || !strings.Contains(err.Error(), "top-logprob alternatives") {
			t.Fatalf("Execute() error = %v, want missing alternatives", err)
		}
		evidence, ok := ExecutionEvidenceFromError(err)
		if !ok || evidence.Usage.TotalTokens != 3 || evidence.Iterations != 1 {
			t.Fatalf("partial evidence = %+v (present=%v), want first paid call", evidence, ok)
		}
		return
	}
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if response.Model != wantModel || response.Usage.TotalTokens != 6 {
		t.Fatalf("response = model %q usage %+v, want escalated aggregate", response.Model, response.Usage)
	}
}

func confidenceLogprobRequest(onError string) *Request {
	return confidenceRequest(onError, "avg_logprob")
}

func confidenceRequest(onError, method string) *Request {
	params := openai.ChatCompletionNewParams{
		Model: "auto",
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("answer with calibrated confidence"),
		},
	}
	return &Request{
		executionRequest: &params,
		ModelRefs: []config.ModelRef{
			{Model: "small"},
			{Model: "large"},
		},
		ModelParams: map[string]config.ModelParams{
			"small": {ParamSize: "1b"},
			"large": {ParamSize: "10b"},
		},
		Algorithm: &config.AlgorithmConfig{
			Type: "confidence",
			Confidence: &config.ConfidenceAlgorithmConfig{
				ConfidenceMethod: method,
				Threshold:        0.72,
				OnError:          onError,
			},
		},
		DecisionName: "confidence_logprob_test",
	}
}

func newConfidenceLogprobBackend(t *testing.T, topLogprobsByModel map[string]int) (*httptest.Server, *[]string) {
	t.Helper()
	calls := []string{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var requestBody struct {
			Model    string `json:"model"`
			Logprobs bool   `json:"logprobs"`
		}
		if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
			t.Errorf("decode request: %v", err)
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		calls = append(calls, requestBody.Model)
		if !requestBody.Logprobs {
			t.Errorf("model %q request did not enable logprobs", requestBody.Model)
		}

		choice := map[string]interface{}{
			"index":         0,
			"message":       map[string]interface{}{"role": "assistant", "content": requestBody.Model + " answer"},
			"finish_reason": "stop",
		}
		if topLogprobsByModel[requestBody.Model] > 0 {
			topLogprobs := []map[string]interface{}{
				{"token": "answer", "logprob": -0.1, "bytes": []int{97}},
			}
			if topLogprobsByModel[requestBody.Model] > 1 {
				topLogprobs = append(topLogprobs, map[string]interface{}{
					"token": "alternative", "logprob": -2.1, "bytes": []int{98},
				})
			}
			choice["logprobs"] = map[string]interface{}{
				"content": []map[string]interface{}{
					{
						"token":        "answer",
						"logprob":      -0.1,
						"bytes":        []int{97},
						"top_logprobs": topLogprobs,
					},
				},
			}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":      "chatcmpl-confidence-test",
			"object":  "chat.completion",
			"created": 0,
			"model":   requestBody.Model,
			"choices": []map[string]interface{}{choice},
			"usage": map[string]int{
				"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3,
			},
		})
	}))
	return server, &calls
}
