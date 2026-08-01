package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
			}]
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

	recordSelectionFallback(
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
