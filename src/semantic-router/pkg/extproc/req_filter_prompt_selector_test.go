package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestDecisionPromptSelectorCallsConcreteHelperModel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-vsr-looper-request") != "true" {
			t.Fatalf("missing internal looper header")
		}
		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if body["model"] != "router-small" {
			t.Fatalf("model = %v", body["model"])
		}
		if body["max_completion_tokens"] != float64(promptSelectorMaxCompletionTokens) {
			t.Fatalf("max_completion_tokens = %v", body["max_completion_tokens"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-router",
			"object":"chat.completion",
			"created":1,
			"model":"router-small",
			"choices":[{
				"index":0,
				"message":{
					"role":"assistant",
					"content":"{\"selected_model\":\"reasoning-large\",\"rationale\":\"Hard task\"}"
				},
				"finish_reason":"stop"
			}],
			"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}
		}`))
	}))
	defer server.Close()

	router := &OpenAIRouter{Config: &config.RouterConfig{
		Looper: config.LooperConfig{Endpoint: server.URL},
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"general-small":   {Description: "Fast model"},
			"reasoning-large": {Description: "Hard reasoning"},
		}},
	}}
	selector := router.newDecisionPromptSelector(config.PromptSelectionConfig{
		Model:        "router-small",
		Instructions: "Choose by difficulty.",
	})

	result, err := selector.Select(t.Context(), &selection.SelectionContext{
		Query: "solve this proof",
		CandidateModels: []config.ModelRef{
			{Model: "general-small"},
			{Model: "reasoning-large"},
		},
	})
	if err != nil {
		t.Fatalf("Select() error = %v", err)
	}
	if result.SelectedModel != "reasoning-large" {
		t.Fatalf("SelectedModel = %q", result.SelectedModel)
	}
	if result.HelperPromptTokens != 10 ||
		result.HelperCompletionTokens != 4 ||
		result.HelperTotalTokens != 14 {
		t.Fatalf("helper usage = %#v", result)
	}
}

func TestRecordSelectionFallbackPersistsBoundedReason(t *testing.T) {
	selection.InitializeMetrics()
	requestContext := &RequestContext{}
	selectionContext := &selection.SelectionContext{
		DecisionName:    "prompt-route",
		CandidateModels: []config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
	}
	fallback := &selectionContext.CandidateModels[0]
	counter := selection.ModelSelectionFallbackTotal.WithLabelValues(
		string(selection.MethodPrompt),
		selectionContext.DecisionName,
		selectionFallbackError,
	)
	before := testutil.ToFloat64(counter)

	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	router.recordSelectionFallback(
		selection.MethodPrompt,
		selectionFallbackError,
		selectionContext,
		nil,
		fallback,
		nil,
		requestContext,
	)

	if requestContext.VSRSelectionReasoning != selectionFallbackError {
		t.Fatalf(
			"selection reasoning = %q, want %q",
			requestContext.VSRSelectionReasoning,
			selectionFallbackError,
		)
	}
	if got := testutil.ToFloat64(counter); got != before+1 {
		t.Fatalf("fallback counter = %v, want %v", got, before+1)
	}
}

func TestPromptSelectionReasoningIsContentFree(t *testing.T) {
	const secret = "synthetic-secret-from-request"
	reason := selectionReasoningForDiagnostics(
		selection.MethodPrompt,
		"selected because request contained "+secret,
	)
	if reason != "prompt selector selected declared candidate" {
		t.Fatalf("reason = %q", reason)
	}
	if strings.Contains(reason, secret) {
		t.Fatal("prompt reasoning leaked model-controlled request content")
	}
}

func TestSelectionFallbackReasonClassifiesPromptFailures(t *testing.T) {
	for _, testCase := range []struct {
		err  error
		want string
	}{
		{context.Canceled, selectionFallbackCancelled},
		{context.DeadlineExceeded, selectionFallbackTimeout},
		{
			fmt.Errorf("%w: malformed", selection.ErrPromptInvalidOutput),
			selectionFallbackInvalidOutput,
		},
		{
			fmt.Errorf("%w: missing", selection.ErrPromptUndeclaredCandidate),
			selectionFallbackUndeclaredCandidate,
		},
		{
			fmt.Errorf("%w: unavailable", selection.ErrPromptInvocation),
			selectionFallbackInvocation,
		},
	} {
		if got := selectionFallbackReasonForError(testCase.err); got != testCase.want {
			t.Fatalf("reason = %q, want %q", got, testCase.want)
		}
	}
}

func TestFastResponseDecisionSkipsPromptHelper(t *testing.T) {
	payload, err := config.NewStructuredPayload(
		config.FastResponsePluginConfig{Message: "done"},
	)
	if err != nil {
		t.Fatalf("create fast-response payload: %v", err)
	}
	useReasoning := false
	result := &decision.DecisionResult{Decision: &config.Decision{
		Name: "fast",
		ModelRefs: []config.ModelRef{
			{
				Model: "model-a",
				ModelReasoningControl: config.ModelReasoningControl{
					UseReasoning: &useReasoning,
				},
			},
			{
				Model: "model-b",
				ModelReasoningControl: config.ModelReasoningControl{
					UseReasoning: &useReasoning,
				},
			},
		},
		Algorithm: &config.AlgorithmConfig{
			Type: "prompt",
			Prompt: &config.PromptSelectionConfig{
				Model:        "unreachable-helper",
				Instructions: "Choose.",
			},
		},
		Plugins: []config.DecisionPlugin{{
			Type:          config.DecisionPluginFastResponse,
			Configuration: payload,
		}},
	}}
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	requestContext := &RequestContext{}

	selected, _, err := router.selectDecisionRuntimeModel(
		result,
		"fast",
		"query",
		"",
		1,
		requestContext,
	)
	if err != nil {
		t.Fatalf("selectDecisionRuntimeModel() error = %v", err)
	}

	if selected != "model-a" ||
		requestContext.VSRSelectionMethod != "fast_response" {
		t.Fatalf(
			"selected=%q method=%q",
			selected,
			requestContext.VSRSelectionMethod,
		)
	}
}

func TestDecisionPromptSelectorCancelsHelperRequest(t *testing.T) {
	started := make(chan struct{})
	cancelled := make(chan struct{})
	releaseHandler := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(
		func(_ http.ResponseWriter, request *http.Request) {
			close(started)
			select {
			case <-request.Context().Done():
				close(cancelled)
			case <-releaseHandler:
			}
		},
	))
	defer func() {
		select {
		case <-releaseHandler:
		default:
			close(releaseHandler)
		}
		server.Close()
	}()
	router := &OpenAIRouter{Config: &config.RouterConfig{
		Looper: config.LooperConfig{Endpoint: server.URL},
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"model-a": {},
			"model-b": {},
		}},
	}}
	selector := router.newDecisionPromptSelector(
		config.PromptSelectionConfig{
			Model:          "helper",
			Instructions:   "Choose.",
			TimeoutSeconds: 30,
		},
	)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error)
	go func() {
		_, err := selector.Select(ctx, &selection.SelectionContext{
			Query: "cancel",
			CandidateModels: []config.ModelRef{
				{Model: "model-a"},
				{Model: "model-b"},
			},
		})
		done <- err
	}()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("helper request did not start")
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("selector error = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("selector did not return after cancellation")
	}
	select {
	case <-cancelled:
	default:
		close(releaseHandler)
	}
}

func TestPromptHelperTelemetryPersistsInReplayDiagnostics(t *testing.T) {
	ctx := &RequestContext{
		VSRPromptHelperModel:            "helper",
		VSRPromptHelperPromptTokens:     10,
		VSRPromptHelperCompletionTokens: 4,
		VSRPromptHelperTotalTokens:      14,
		VSRPromptHelperLatencyMs:        25,
	}

	diagnostics := buildReplayRouteDiagnostics(
		ctx,
		"vllm-sr/auto",
		"model-a",
		"prompt-route",
		0,
		100,
	)

	if diagnostics.PromptHelperModel != "helper" ||
		diagnostics.PromptHelperTotalTokens != 14 ||
		diagnostics.PromptHelperLatencyMs != 25 {
		t.Fatalf("prompt helper diagnostics = %#v", diagnostics)
	}
}
