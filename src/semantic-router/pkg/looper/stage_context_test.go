package looper

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestValidateLooperStageContextRejectsGeneratedPromptGrowth(t *testing.T) {
	original := &openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hello")},
	}
	stage := appendFusionStageMessage(original, strings.Repeat("evidence ", 40))
	req := &Request{
		OriginalRequest:   original,
		BaseContextTokens: 100,
		ModelRefs:         []config.ModelRef{{Model: "worker"}},
		ModelParams: map[string]config.ModelParams{
			"worker": {ContextWindowSize: 120},
		},
	}

	err := validateLooperStageContext(req, stage, "worker")
	var contextErr *StageContextWindowError
	if !errors.As(err, &contextErr) {
		t.Fatalf("expected StageContextWindowError, got %v", err)
	}
	if contextErr.EstimatedTokens <= contextErr.ContextWindow {
		t.Fatalf("estimated tokens %d did not exceed window %d", contextErr.EstimatedTokens, contextErr.ContextWindow)
	}
	if strings.Contains(err.Error(), "evidence") {
		t.Fatalf("stage error leaked request content: %v", err)
	}
}

func TestValidateLooperStageContextKeepsCompatibilityWithoutWindowMetadata(t *testing.T) {
	original := &openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hello")},
	}
	stage := appendFusionStageMessage(original, strings.Repeat("large ", 100))
	err := validateLooperStageContext(&Request{
		OriginalRequest:   original,
		BaseContextTokens: 100,
		ModelRefs:         []config.ModelRef{{Model: "worker"}},
		ModelParams:       map[string]config.ModelParams{"worker": {}},
	}, stage, "worker")
	if err != nil {
		t.Fatalf("missing context metadata should remain eligible: %v", err)
	}
}

func TestValidateLooperStageContextAccountsForLargerStageOutputReserve(t *testing.T) {
	original := &openai.ChatCompletionNewParams{
		Messages:            []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hello")},
		MaxCompletionTokens: openai.Int(8),
	}
	stage := cloneRequest(original)
	stage.MaxCompletionTokens = openai.Int(64)
	req := &Request{
		OriginalRequest:   original,
		BaseContextTokens: 100,
		ModelRefs:         []config.ModelRef{{Model: "worker"}},
		ModelParams: map[string]config.ModelParams{
			"worker": {ContextWindowSize: 150},
		},
	}

	err := validateLooperStageContext(req, stage, "worker")
	var contextErr *StageContextWindowError
	if !errors.As(err, &contextErr) {
		t.Fatalf("expected output reserve to exceed the stage context, got %v", err)
	}
	if contextErr.EstimatedTokens != 156 {
		t.Fatalf("estimated tokens = %d, want 156", contextErr.EstimatedTokens)
	}
}

func TestLooperStageContextUsesMixedScriptTokenEstimate(t *testing.T) {
	original := &openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hello")},
	}
	stage := appendFusionStageMessage(original, strings.Repeat("界", 20))

	addedTokens, err := looperStageAddedMessageTokens(original, stage)
	if err != nil {
		t.Fatalf("estimate generated stage tokens: %v", err)
	}
	if addedTokens < 30 {
		t.Fatalf("mixed-script token estimate = %d, want at least 30", addedTokens)
	}
}

func TestWorkflowsRejectsOversizedGeneratedStageBeforeBackendDispatch(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		calls.Add(1)
	}))
	defer server.Close()

	_, err := NewWorkflowsLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(
		context.Background(),
		&Request{
			OriginalRequest: &openai.ChatCompletionNewParams{
				Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("solve")},
			},
			BaseContextTokens: 100,
			ModelRefs:         []config.ModelRef{{Model: "worker"}},
			ModelParams: map[string]config.ModelParams{
				"worker": {ContextWindowSize: 110},
			},
			Algorithm: &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmWorkflows,
				Workflows: &config.WorkflowsAlgorithmConfig{
					Mode:        config.WorkflowModeStatic,
					MaxSteps:    1,
					MaxParallel: 1,
					OnError:     config.WorkflowOnErrorFail,
					Roles: []config.WorkflowRoleConfig{{
						Name:   "worker",
						Models: []string{"worker"},
						Prompt: strings.Repeat("inspect ", 30),
					}},
				},
			},
		},
	)

	var contextErr *StageContextWindowError
	if !errors.As(err, &contextErr) {
		t.Fatalf("expected generated stage context error, got %v", err)
	}
	if got := calls.Load(); got != 0 {
		t.Fatalf("backend received %d calls after context gate rejection", got)
	}
}
