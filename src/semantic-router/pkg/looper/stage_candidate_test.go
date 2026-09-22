package looper

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func candidateStageRequest() *Request {
	return &Request{
		OriginalRequest: &openai.ChatCompletionNewParams{
			Model: "worker", Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("Summarize the meeting agenda.")},
			MaxCompletionTokens: openai.Int(16),
		},
		RecipeName: "scoped", DecisionName: "meeting",
		CandidateRequirements: &config.CandidateRequirements{Capabilities: config.CandidateCapabilitiesDeclared, Context: config.CandidateContextKnownLimits},
		PermittedModels:       []string{"worker", "planner", "final"},
		ModelRefs:             []config.ModelRef{{Model: "worker"}},
		ModelParams: map[string]config.ModelParams{
			"worker":  {Capabilities: []string{"text"}, ContextWindowSize: 4096, MaxOutputTokens: 512},
			"planner": {Capabilities: []string{"text"}, ContextWindowSize: 4096, MaxOutputTokens: 512},
			"final":   {Capabilities: []string{"text"}, ContextWindowSize: 4096, MaxOutputTokens: 512},
		},
	}
}

func candidateStageClient(t *testing.T) (*Client, *atomic.Int32, <-chan map[string]json.RawMessage) {
	t.Helper()
	var calls atomic.Int32
	bodies := make(chan map[string]json.RawMessage, 16)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var body map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode stage body: %v", err)
		}
		bodies <- body
		if got := r.Header.Get(headers.VSRSelectedRecipe); got != "scoped" {
			t.Errorf("stage lost recipe: %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"stage","object":"chat.completion","created":1,"model":"worker","choices":[{"index":0,"message":{"role":"assistant","content":"Agenda summary."},"finish_reason":"stop"}]}`))
	}))
	t.Cleanup(server.Close)
	client := NewClient(&config.LooperConfig{Endpoint: server.URL})
	t.Cleanup(func() { _ = client.Close() })
	return client, &calls, bodies
}

func dispatchCandidateStage(client *Client, base *Request, stage *openai.ChatCompletionNewParams, model string) error {
	l := newBaseLooper(&config.LooperConfig{}, borrowClient(client))
	_, err := l.dispatchModel(context.Background(), base, stage, ModelTarget{Name: model}, CallOptions{DecisionName: base.DecisionName, Iteration: 1})
	return err
}

func TestLooperCandidateStageRejectsBeforeConnector(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Request, *openai.ChatCompletionNewParams)
	}{
		{"unpermitted helper", func(r *Request, _ *openai.ChatCompletionNewParams) { r.PermittedModels = []string{"planner"} }},
		{"no permission closure", func(r *Request, _ *openai.ChatCompletionNewParams) { r.PermittedModels = nil }},
		{"missing metadata", func(r *Request, _ *openai.ChatCompletionNewParams) { delete(r.ModelParams, "worker") }},
		{"missing capabilities", func(r *Request, _ *openai.ChatCompletionNewParams) {
			p := r.ModelParams["worker"]
			p.Capabilities = nil
			r.ModelParams["worker"] = p
		}},
		{"missing context", func(r *Request, _ *openai.ChatCompletionNewParams) {
			p := r.ModelParams["worker"]
			p.ContextWindowSize = 0
			r.ModelParams["worker"] = p
		}},
		{"missing output metadata", func(r *Request, _ *openai.ChatCompletionNewParams) {
			p := r.ModelParams["worker"]
			p.MaxOutputTokens = 0
			r.ModelParams["worker"] = p
		}},
		{"omitted output is not model capacity", func(_ *Request, s *openai.ChatCompletionNewParams) {
			s.MaxCompletionTokens = (openai.ChatCompletionNewParams{}).MaxCompletionTokens
		}},
		{"output exceeds model", func(_ *Request, s *openai.ChatCompletionNewParams) { s.MaxCompletionTokens = openai.Int(513) }},
		{"generated context growth", func(_ *Request, s *openai.ChatCompletionNewParams) {
			s.Messages = append(s.Messages, openai.UserMessage(strings.Repeat("meeting notes ", 4096)))
		}},
		{"conflicting output fields", func(_ *Request, s *openai.ChatCompletionNewParams) { s.MaxTokens = openai.Int(8) }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			client, calls, _ := candidateStageClient(t)
			base := candidateStageRequest()
			stage := cloneRequest(base.OriginalRequest)
			tc.mutate(base, stage)
			err := dispatchCandidateStage(client, base, stage, "worker")
			if !errors.Is(err, selection.ErrNoEligibleCandidates) || calls.Load() != 0 {
				t.Fatalf("err=%v, calls=%d; expected rejection before connector", err, calls.Load())
			}
			if strings.Contains(err.Error(), "meeting") {
				t.Fatalf("admission error included stage content: %v", err)
			}
		})
	}
}

func TestLooperCandidateUsesCompleteStageBudgetExactlyOnce(t *testing.T) {
	client, calls, _ := candidateStageClient(t)
	base := candidateStageRequest()
	stage := cloneRequest(base.OriginalRequest)
	base.OriginalRequest = &openai.ChatCompletionNewParams{Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage(strings.Repeat("prior agenda ", 1000))}}
	base.BaseContextTokens = 100000
	body, err := json.Marshal(stage)
	if err != nil {
		t.Fatal(err)
	}
	neutral, _, _, err := protocolcodec.NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body)
	if err != nil {
		t.Fatal(err)
	}
	demand := selection.DemandForRequest(&neutral)
	p := base.ModelParams["worker"]
	p.ContextWindowSize = demand.InputTokens + int(*demand.MaxOutputTokens)
	base.ModelParams["worker"] = p
	if err := dispatchCandidateStage(client, base, stage, "worker"); err != nil {
		t.Fatalf("exact stage budget was double counted: %v", err)
	}
	p.ContextWindowSize--
	base.ModelParams["worker"] = p
	if err := dispatchCandidateStage(client, base, stage, "worker"); !errors.Is(err, selection.ErrNoEligibleCandidates) {
		t.Fatalf("one token below actual stage budget accepted: %v", err)
	}
	if calls.Load() != 1 {
		t.Fatalf("connector calls=%d, want 1", calls.Load())
	}
}

func TestLooperCandidateChecksToolAndOutputCapabilities(t *testing.T) {
	for _, raw := range []string{
		`{"model":"worker","messages":[{"role":"user","content":"Read the meeting agenda."}],"max_completion_tokens":16,"tools":[{"type":"function","function":{"name":"agenda","parameters":{"type":"object"}}}]}`,
		`{"model":"worker","messages":[{"role":"user","content":"Return an agenda object."}],"max_completion_tokens":16,"response_format":{"type":"json_object"}}`,
		`{"model":"worker","messages":[{"role":"user","content":"Read the agenda."}],"max_completion_tokens":16,"reasoning_effort":"high"}`,
		`{"model":"worker","messages":[{"role":"user","content":"Read the agenda."}],"max_completion_tokens":16,"logit_bias":{"1":1}}`,
	} {
		client, calls, _ := candidateStageClient(t)
		base := candidateStageRequest()
		var stage openai.ChatCompletionNewParams
		if err := json.Unmarshal([]byte(raw), &stage); err != nil {
			t.Fatal(err)
		}
		if err := dispatchCandidateStage(client, base, &stage, "worker"); !errors.Is(err, selection.ErrNoEligibleCandidates) {
			t.Fatalf("unsupported stage accepted: %v", err)
		}
		if calls.Load() != 0 {
			t.Fatalf("unsupported stage reached connector")
		}
	}
}

func TestLooperCandidateStageOutputOverridesAndPolicyCap(t *testing.T) {
	for _, stage := range []string{"planner", "worker", "final", "fusion"} {
		t.Run(stage, func(t *testing.T) {
			client, calls, bodies := candidateStageClient(t)
			base := candidateStageRequest()
			base.OriginalRequest.MaxCompletionTokens = (openai.ChatCompletionNewParams{}).MaxCompletionTokens
			base.OriginalRequest.MaxTokens = openai.Int(8)
			limit := 24
			base.MaxTokensLimit = &limit
			var err error
			if stage == "fusion" {
				l := newFusionLooper(&config.LooperConfig{}, borrowClient(client))
				_, err = l.callFusionModel(context.Background(), base, base.OriginalRequest, fusionExecutionConfig{MaxCompletionTokens: 48}, "final", false, false, 1, config.FusionModelOverride{})
			} else {
				l := newWorkflowsLooper(&config.LooperConfig{}, borrowClient(client))
				_, err = l.callWorkflowModel(context.Background(), base.OriginalRequest, workflowsExecutionConfig{PlannerModel: "planner", PlannerMaxCompletionTokens: 64, MaxCompletionTokens: 48}, stage, false, 1, base)
			}
			if err != nil {
				t.Fatalf("stage failed: %v", err)
			}
			body := <-bodies
			if string(body["max_completion_tokens"]) != "24" || body["max_tokens"] != nil || calls.Load() != 1 {
				t.Fatalf("stage output override/cap was not transmitted once: %v", body)
			}
			if base.OriginalRequest.MaxTokens.Value != 8 || base.OriginalRequest.MaxCompletionTokens.Valid() {
				t.Fatal("stage mutated original caller sampling")
			}
		})
	}
}

func TestLooperCandidateConfidenceRetainsNativeEvidenceAndRejectsLargeVerifier(t *testing.T) {
	client, calls, bodies := candidateStageClient(t)
	base := candidateStageRequest()
	l := newBaseLooper(&config.LooperConfig{}, borrowClient(client))
	_, _, err := l.startConfidenceModelAttempt(context.Background(), base, base.OriginalRequest, "worker", "candidate", "worker", false, 1, &LogprobsConfig{Enabled: true, TopLogprobs: 9}, "")
	if err != nil {
		t.Fatalf("native confidence evidence rejected: %v", err)
	}
	body := <-bodies
	if string(body["logprobs"]) != "true" || string(body["top_logprobs"]) != "5" {
		t.Fatalf("native evidence lost: %v", body)
	}
	verifier := cloneRequest(base.OriginalRequest)
	verifier.Messages = []openai.ChatCompletionMessageParamUnion{openai.UserMessage(strings.Repeat("review agenda ", 4096))}
	_, _, err = l.startConfidenceModelAttempt(context.Background(), base, verifier, "worker", "self_verifier", "verifier", false, 2, nil, "")
	if !errors.Is(err, selection.ErrNoEligibleCandidates) || calls.Load() != 1 {
		t.Fatalf("large verifier dispatched: %v calls=%d", err, calls.Load())
	}
}

func TestLooperCandidateKeepsLegacyAbsentAndDeclaredAliases(t *testing.T) {
	client, calls, _ := candidateStageClient(t)
	base := candidateStageRequest()
	base.CandidateRequirements = nil
	base.PermittedModels = nil
	base.ModelParams = nil
	if err := dispatchCandidateStage(client, base, base.OriginalRequest, "legacy"); err != nil {
		t.Fatal(err)
	}
	base = candidateStageRequest()
	base.ModelRefs[0].LoRAName = "worker-adapter"
	base.PermittedModels = append(base.PermittedModels, "worker-adapter")
	if err := dispatchCandidateStage(client, base, base.OriginalRequest, "worker-adapter"); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 2 {
		t.Fatalf("calls=%d", calls.Load())
	}
	base.PermittedModels = append(base.PermittedModels, "unregistered")
	if err := dispatchCandidateStage(client, base, base.OriginalRequest, "unregistered"); !errors.Is(err, selection.ErrNoEligibleCandidates) {
		t.Fatalf("invented alias accepted: %v", err)
	}
}
