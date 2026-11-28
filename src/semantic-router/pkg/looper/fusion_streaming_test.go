package looper

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestPrepareStreamingSynthesis_OneCallMode(t *testing.T) {
	var (
		seenMu     sync.Mutex
		seenModels []string
	)
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		seenMu.Lock()
		seenModels = append(seenModels, model)
		seenMu.Unlock()
		switch model {
		case "panel-a":
			return "panel A perspective", http.StatusOK
		case "panel-b":
			return "panel B perspective", http.StatusOK
		default:
			t.Errorf("unexpected model called during prepare: %s", model)
			return "unexpected", http.StatusInternalServerError
		}
	})
	defer server.Close()

	req := newFusionTestRequest()
	req.IsStreaming = true
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:          "judge",
			AnalysisModels: []string{"panel-a", "panel-b"},
			AnalysisMode:   config.FusionAnalysisModeOneCall,
		},
	}

	looper := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL})
	plan, err := looper.PrepareStreamingSynthesis(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, plan)

	// Verify target model
	assert.Equal(t, "judge", plan.TargetModel)
	assert.Equal(t, "judge", plan.SynthesisParams.Model)

	// Verify only panel models were called, judge was NOT called yet (to be dispatched by Envoy)
	seenMu.Lock()
	assert.ElementsMatch(t, []string{"panel-a", "panel-b"}, seenModels)
	seenMu.Unlock()

	// Verify prompt contains panel responses
	messages := plan.SynthesisParams.Messages
	require.NotEmpty(t, messages)
	lastMsgJSON, err := json.Marshal(messages[len(messages)-1])
	require.NoError(t, err)
	assert.Contains(t, string(lastMsgJSON), "panel A perspective")
	assert.Contains(t, string(lastMsgJSON), "panel B perspective")

	// Verify models used & accounting (only intermediate panel calls executed so far; judge runs later via Envoy)
	assert.ElementsMatch(t, []string{"panel-a", "panel-b"}, plan.ModelsUsed)
	assert.NotContains(t, plan.ModelsUsed, "judge")
	assert.Equal(t, 2, plan.Iterations)
	// panel-a (10/2/12) + panel-b (20/3/23)
	assert.Equal(t, TokenUsage{PromptTokens: 30, CompletionTokens: 5, TotalTokens: 35}, plan.IntermediateUsage)
	require.NotNil(t, plan.Trace)
	fusionTrace, ok := plan.Trace.(*FusionTrace)
	require.True(t, ok, "trace should be *FusionTrace")
	assert.Equal(t, config.FusionAnalysisModeOneCall, fusionTrace.AnalysisMode)
	assert.Equal(t, "judge", fusionTrace.JudgeModel)
	assert.Len(t, fusionTrace.Responses, 2)
	assert.Nil(t, plan.QuorumOutcome)
}

func TestPrepareStreamingSynthesis_SeparateAnalysisMode(t *testing.T) {
	var (
		seenMu      sync.Mutex
		seenModels  []string
		judgePrompt string
	)
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		seenMu.Lock()
		seenModels = append(seenModels, model)
		if model == "judge" {
			judgePrompt = prompt
		}
		seenMu.Unlock()
		switch model {
		case "panel-a":
			return "panel A perspective", http.StatusOK
		case "panel-b":
			return "panel B perspective", http.StatusOK
		case "judge":
			// Intermediate judge analysis call
			return `{"consensus":["common fact"],"contradictions":[],"partial_coverage":[],"unique_insights":["insight 1"],"blind_spots":[]}`, http.StatusOK
		default:
			t.Errorf("unexpected model called during prepare: %s", model)
			return "unexpected", http.StatusInternalServerError
		}
	})
	defer server.Close()

	req := newFusionTestRequest()
	req.IsStreaming = true
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:          "judge",
			AnalysisModels: []string{"panel-a", "panel-b"},
			AnalysisMode:   config.FusionAnalysisModeSeparate,
		},
	}

	looper := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL})
	plan, err := looper.PrepareStreamingSynthesis(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, plan)

	// Verify judge was called once for structured analysis, but not for final synthesis
	seenMu.Lock()
	require.Len(t, seenModels, 3)
	assert.ElementsMatch(t, []string{"panel-a", "panel-b"}, seenModels[:2])
	assert.Equal(t, "judge", seenModels[2])
	capturedJudgePrompt := judgePrompt
	seenMu.Unlock()
	require.Contains(t, capturedJudgePrompt, "return only valid JSON")

	// Verify trace contains structured analysis
	require.NotNil(t, plan.Trace)
	fusionTrace, ok := plan.Trace.(*FusionTrace)
	require.True(t, ok, "trace should be *FusionTrace")
	require.NotNil(t, fusionTrace.Analysis)
	assert.Equal(t, []string{"common fact"}, fusionTrace.Analysis.Consensus)

	// Verify synthesis prompt incorporates analysis and intermediate accounting includes panels + judge analysis
	lastMsgJSON, err := json.Marshal(plan.SynthesisParams.Messages[len(plan.SynthesisParams.Messages)-1])
	require.NoError(t, err)
	assert.Contains(t, string(lastMsgJSON), "common fact")
	assert.Equal(t, TokenUsage{PromptTokens: 60, CompletionTokens: 9, TotalTokens: 69}, plan.IntermediateUsage)

	// Verify synthesis params & intermediate accounting (2 panels + 1 judge analysis = 3 intermediate calls)
	assert.Equal(t, "judge", plan.TargetModel)
	assert.Equal(t, "judge", plan.SynthesisParams.Model)
	assert.ElementsMatch(t, []string{"panel-a", "panel-b", "judge"}, plan.ModelsUsed)
	assert.Equal(t, 3, plan.Iterations)
}

func TestPrepareStreamingSynthesis_QuorumFallback(t *testing.T) {
	// All panel models fail -> trigger quorum fallback
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		return "panel error", http.StatusInternalServerError
	})
	defer server.Close()

	temp := 0.5
	req := newFusionTestRequest()
	req.IsStreaming = true
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:                "judge",
			AnalysisModels:       []string{"panel-a", "panel-b"},
			QuorumFailurePolicy:  config.FusionQuorumFailurePolicyFallback,
			QuorumFallbackTarget: "backup-model",
			Temperature:          &temp,
			MaxCompletionTokens:  512,
		},
	}

	looper := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL})
	plan, err := looper.PrepareStreamingSynthesis(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, plan)

	// Fallback target is selected for direct upstream streaming
	assert.Equal(t, "backup-model", plan.TargetModel)
	assert.Equal(t, "backup-model", plan.SynthesisParams.Model)
	assert.Equal(t, temp, plan.SynthesisParams.Temperature.Value)
	assert.Equal(t, int64(512), plan.SynthesisParams.MaxCompletionTokens.Value)

	// Quorum outcome evidence is preserved (only intermediate dispatched panels in plan.ModelsUsed; backup-model is dispatched later via Envoy)
	assert.Nil(t, plan.Trace)
	require.NotNil(t, plan.QuorumOutcome)
	assert.Equal(t, FusionQuorumFallbackReady, plan.QuorumOutcome.Disposition)
	assert.Equal(t, "backup-model", plan.QuorumOutcome.FallbackTarget)
	assert.ElementsMatch(t, []string{"panel-a", "panel-b"}, plan.ModelsUsed)
	assert.NotContains(t, plan.ModelsUsed, "backup-model")
	assert.Equal(t, 2, plan.Iterations)

	// Verify original request was cloned and not mutated in place
	assert.Equal(t, "vllm-sr/fusion", req.OriginalRequest.Model)
	assert.Len(t, req.OriginalRequest.Messages, 1)
	require.Len(t, plan.SynthesisParams.Messages, 1)
	msgJSON, err := json.Marshal(plan.SynthesisParams.Messages[0])
	require.NoError(t, err)
	assert.Contains(t, string(msgJSON), "compare the options")
}

func TestPrepareStreamingSynthesis_PartialQuorumFallbackPreservesTokens(t *testing.T) {
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		if model == "panel-a" {
			return "panel a partial answer", http.StatusOK
		}
		return "panel error", http.StatusInternalServerError
	})
	defer server.Close()

	req := belowQuorumFusionRequest(belowQuorumFusionConfig(
		config.FusionQuorumFailurePolicyFallback, "backup-model"))
	req.IsStreaming = true

	looper := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL})
	plan, err := looper.PrepareStreamingSynthesis(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, plan)

	assert.Nil(t, plan.Trace)
	assert.Equal(t, "backup-model", plan.TargetModel)
	assert.Equal(t, TokenUsage{PromptTokens: 10, CompletionTokens: 2, TotalTokens: 12}, plan.IntermediateUsage)
	require.NotNil(t, plan.QuorumOutcome)
	assert.Equal(t, FusionQuorumFallbackReady, plan.QuorumOutcome.Disposition)
	assert.Equal(t, 1, plan.QuorumOutcome.UsableCount)
	assert.Equal(t, 2, plan.QuorumOutcome.RequiredCount)
	assert.ElementsMatch(t, []string{"panel-a", "panel-b", "panel-c"}, plan.ModelsUsed)
	assert.NotContains(t, plan.ModelsUsed, "backup-model")
	assert.Equal(t, 3, plan.Iterations)
	assert.Equal(t, "vllm-sr/fusion", req.OriginalRequest.Model)
}

func TestPrepareStreamingSynthesis_QuorumFailureAborts(t *testing.T) {
	// Panel fails and policy is fail
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		return "panel error", http.StatusInternalServerError
	})
	defer server.Close()

	req := newFusionTestRequest()
	req.IsStreaming = true
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:               "judge",
			AnalysisModels:      []string{"panel-a", "panel-b"},
			QuorumFailurePolicy: config.FusionQuorumFailurePolicyFail,
		},
	}

	looper := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL})
	plan, err := looper.PrepareStreamingSynthesis(context.Background(), req)
	require.Error(t, err)
	assert.Nil(t, plan)
	assert.True(t, strings.Contains(err.Error(), "quorum") || strings.Contains(err.Error(), "panel"))
}

func TestPrepareStreamingSynthesis_PreservesSamplingParams(t *testing.T) {
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		return "panel ok", http.StatusOK
	})
	defer server.Close()

	temp := 0.7
	req := newFusionTestRequest()
	req.IsStreaming = true
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:               "judge",
			AnalysisModels:      []string{"panel-a"},
			AnalysisMode:        config.FusionAnalysisModeOneCall,
			Temperature:         &temp,
			MaxCompletionTokens: 1024,
		},
	}

	looper := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL})
	plan, err := looper.PrepareStreamingSynthesis(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, plan)

	assert.Equal(t, temp, plan.SynthesisParams.Temperature.Value)
	assert.Equal(t, int64(1024), plan.SynthesisParams.MaxCompletionTokens.Value)
}

func TestPrepareStreamingSynthesis_CancelledDuringPanelReturnsQuorumCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	secondPanelStarted := make(chan struct{})
	var callSeq atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		switch callSeq.Add(1) {
		case 1:
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"id":      "chatcmpl-test",
				"object":  "chat.completion",
				"created": 1,
				"model":   payload.Model,
				"choices": []map[string]interface{}{{
					"index":         0,
					"message":       map[string]interface{}{"role": "assistant", "content": "panel one done"},
					"finish_reason": "stop",
				}},
				"usage": map[string]int64{"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
			})
		case 2:
			close(secondPanelStarted)
			<-r.Context().Done()
		default:
			writeFusionTestCompletion(w, payload.Model, "err", http.StatusInternalServerError)
		}
	}))
	defer server.Close()

	go func() {
		<-secondPanelStarted
		cancel()
	}()

	req := newFusionTestRequest()
	req.IsStreaming = true
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:                  "judge",
			AnalysisModels:         []string{"panel-a", "panel-b"},
			MaxConcurrent:          1,
			MinSuccessfulResponses: 2,
			QuorumFailurePolicy:    config.FusionQuorumFailurePolicyFallback,
			QuorumFallbackTarget:   "backup-model",
		},
	}

	looper := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL})
	plan, err := looper.PrepareStreamingSynthesis(ctx, req)
	require.Error(t, err)
	assert.Nil(t, plan)

	var quorumErr *FusionQuorumError
	require.ErrorAs(t, err, &quorumErr)
	outcome, ok := FusionQuorumOutcomeFromError(err)
	require.True(t, ok)
	assert.Equal(t, FusionQuorumCancelled, outcome.Disposition)
	assert.Equal(t, TokenUsage{PromptTokens: 10, CompletionTokens: 2, TotalTokens: 12}, quorumErr.Evidence().Usage)
}

func TestPrepareStreamingSynthesis_FallbackWithoutTargetErrors(t *testing.T) {
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		return "panel error", http.StatusInternalServerError
	})
	defer server.Close()

	req := newFusionTestRequest()
	req.IsStreaming = true
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:               "judge",
			AnalysisModels:      []string{"panel-a", "panel-b"},
			QuorumFailurePolicy: config.FusionQuorumFailurePolicyFallback,
		},
	}

	looper := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL})
	plan, err := looper.PrepareStreamingSynthesis(context.Background(), req)
	require.Error(t, err)
	assert.Nil(t, plan)
	assert.Contains(t, err.Error(), "without a target")

	outcome, ok := FusionQuorumOutcomeFromError(err)
	require.True(t, ok)
	assert.Equal(t, FusionQuorumFailed, outcome.Disposition)
}

func TestPrepareStreamingSynthesis_GroundingOutputContractMaxTokensToolsAndRecursion(t *testing.T) {
	var seenPanelTools atomic.Bool
	var decodeErrMu sync.Mutex
	var decodeErr error
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model string        `json:"model"`
			Tools []interface{} `json:"tools"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			decodeErrMu.Lock()
			if decodeErr == nil {
				decodeErr = err
			}
			decodeErrMu.Unlock()
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if len(payload.Tools) > 0 {
			seenPanelTools.Store(true)
		}
		writeFusionTestCompletion(w, payload.Model, "panel grounded answer", http.StatusOK)
	}))
	defer server.Close()

	var params openai.ChatCompletionNewParams
	require.NoError(t, json.Unmarshal([]byte(`{
		"model":"vllm-sr/fusion",
		"max_tokens":200,
		"messages":[{"role":"user","content":"compare options with context: Sydney is a city"}],
		"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],
		"tool_choice":"auto"
	}`), &params))

	req := &Request{
		OriginalRequest: &params,
		DecisionName:    "fusion-grounding",
		ModelParams:     fusionTestModelParams(),
		OutputContract:  "Return concise JSON summary",
		Grounding: &GroundingBackends{
			Detect: func(_ context.Context, _, _, _ string) (GroundingEvidence, error) {
				prob := float32(0.1)
				return GroundingEvidence{Unsupported: false, Probability: &prob}, nil
			},
		},
		Algorithm: &config.AlgorithmConfig{
			Type: "fusion",
			Fusion: &config.FusionAlgorithmConfig{
				Model:               "judge",
				AnalysisModels:      []string{"panel-a", "panel-b"},
				AnalysisMode:        config.FusionAnalysisModeOneCall,
				MaxCompletionTokens: 1024,
				Grounding: &config.FusionGroundingConfig{
					Enabled:   true,
					Reference: config.FusionGroundingReferencePanel,
					Policy:    config.FusionGroundingPolicyWeight,
					MinScore:  0.1,
				},
			},
		},
	}

	looper := NewFusionLooper(&config.LooperConfig{
		Endpoint:         server.URL,
		EntrypointModels: []string{"vllm-sr/fusion"},
	})
	plan, err := looper.PrepareStreamingSynthesis(context.Background(), req)
	require.NoError(t, err)
	require.NotNil(t, plan)
	decodeErrMu.Lock()
	require.NoError(t, decodeErr)
	decodeErrMu.Unlock()

	assert.False(t, seenPanelTools.Load(), "panel calls must strip tools")
	assert.Len(t, plan.SynthesisParams.Tools, 1, "final synthesis must preserve tools")
	assert.False(t, plan.SynthesisParams.MaxTokens.Valid(), "legacy max_tokens must be cleared when max_completion_tokens is set")
	assert.Equal(t, int64(1024), plan.SynthesisParams.MaxCompletionTokens.Value)

	lastMsgJSON, err := json.Marshal(plan.SynthesisParams.Messages[len(plan.SynthesisParams.Messages)-1])
	require.NoError(t, err)
	assert.Contains(t, string(lastMsgJSON), "Return concise JSON summary")
	assert.Contains(t, string(lastMsgJSON), "Weight each panel answer by its groundedness score")

	fusionTrace, ok := plan.Trace.(*FusionTrace)
	require.True(t, ok)
	require.NotNil(t, fusionTrace.Grounding)

	// EntrypointModels recursion guard
	recursiveReq := newFusionTestRequest()
	recursiveReq.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:          "vllm-sr/fusion",
			AnalysisModels: []string{"panel-a"},
		},
	}
	_, recErr := looper.PrepareStreamingSynthesis(context.Background(), recursiveReq)
	require.Error(t, recErr)
	assert.Contains(t, recErr.Error(), "cannot be used as a judge or analysis model")
}
