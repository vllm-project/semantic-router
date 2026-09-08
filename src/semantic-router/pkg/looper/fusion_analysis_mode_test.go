package looper

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestFusionAnalysisModeCachedPanelExecution(t *testing.T) {
	includeNothing := false
	tests := []struct {
		name             string
		mode             string
		wantJudgeCalls   int
		wantIterations   int
		wantTotalTokens  float64
		wantPromptMarker string
		wantAnalysis     bool
	}{
		{
			name:             "separate",
			mode:             config.FusionAnalysisModeSeparate,
			wantJudgeCalls:   2,
			wantIterations:   4,
			wantTotalTokens:  103,
			wantPromptMarker: "Structured analysis:",
			wantAnalysis:     true,
		},
		{
			name:             "one_call",
			mode:             config.FusionAnalysisModeOneCall,
			wantJudgeCalls:   1,
			wantIterations:   3,
			wantTotalTokens:  69,
			wantPromptMarker: "Compare the panel responses",
		},
		{
			name:             "none",
			mode:             config.FusionAnalysisModeNone,
			wantJudgeCalls:   1,
			wantIterations:   3,
			wantTotalTokens:  69,
			wantPromptMarker: "Synthesize a final answer directly",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var judgePrompts []string
			server := newFusionStubServer(t, func(model, prompt string) (string, int) {
				require.Equal(t, "judge", model, "cached panel must bypass live panel calls")
				judgePrompts = append(judgePrompts, prompt)
				if strings.Contains(prompt, "return only valid JSON") {
					return `{"consensus":["agree"],"contradictions":[],"partial_coverage":[],"unique_insights":[],"blind_spots":[]}`, http.StatusOK
				}
				return "final answer", http.StatusOK
			})
			defer server.Close()

			req := newFusionTestRequest()
			req.CachedPanel = cachedTestPanel()
			req.Algorithm = &config.AlgorithmConfig{
				Type: config.DecisionAlgorithmFusion,
				Fusion: &config.FusionAlgorithmConfig{
					Model:                        "judge",
					AnalysisModels:               []string{"panel-a", "panel-b"},
					AnalysisMode:                 tc.mode,
					IncludeAnalysis:              boolPtr(tc.wantAnalysis),
					IncludeIntermediateResponses: &includeNothing,
				},
			}

			resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
			require.NoError(t, err)
			assert.Equal(t, tc.wantIterations, resp.Iterations)
			require.Len(t, judgePrompts, tc.wantJudgeCalls)
			assert.Contains(t, judgePrompts[len(judgePrompts)-1], tc.wantPromptMarker)

			var body map[string]interface{}
			require.NoError(t, json.Unmarshal(resp.Body, &body))
			usage := body["usage"].(map[string]interface{})
			assert.Equal(t, tc.wantTotalTokens, usage["total_tokens"])

			trace, ok := resp.IntermediateResponses.(*FusionTrace)
			require.True(t, ok, "internal trace type = %T", resp.IntermediateResponses)
			assert.Equal(t, tc.mode, trace.AnalysisMode)
			assert.Equal(t, tc.wantAnalysis, trace.Analysis != nil)
			assert.Empty(t, trace.Responses)

			publicTrace, hasPublicTrace := body["fusion"].(map[string]interface{})
			assert.Equal(t, tc.wantAnalysis, hasPublicTrace)
			if hasPublicTrace {
				assert.Equal(t, tc.mode, publicTrace["analysis_mode"])
			}
		})
	}
}

func TestFusionAnalysisModeIncludeAnalysisIsTraceOnly(t *testing.T) {
	for _, includeAnalysis := range []bool{false, true} {
		t.Run(map[bool]string{false: "hidden", true: "visible"}[includeAnalysis], func(t *testing.T) {
			var judgeCalls atomic.Int64
			server := newFusionStubServer(t, func(model, prompt string) (string, int) {
				require.Equal(t, "judge", model)
				judgeCalls.Add(1)
				if strings.Contains(prompt, "return only valid JSON") {
					return `{"consensus":[],"contradictions":[],"partial_coverage":[],"unique_insights":[],"blind_spots":[]}`, http.StatusOK
				}
				return "final answer", http.StatusOK
			})
			defer server.Close()

			req := newFusionTestRequest()
			req.CachedPanel = cachedTestPanel()
			req.Algorithm = &config.AlgorithmConfig{Type: config.DecisionAlgorithmFusion, Fusion: &config.FusionAlgorithmConfig{
				Model: "judge", AnalysisModels: []string{"panel-a", "panel-b"},
				AnalysisMode: config.FusionAnalysisModeSeparate, IncludeAnalysis: &includeAnalysis,
			}}
			_, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
			require.NoError(t, err)
			assert.Equal(t, int64(2), judgeCalls.Load())
		})
	}
}

func TestFusionAnalysisModeTraceEnvelopeCompatibility(t *testing.T) {
	includeNothing := false
	for _, mode := range []string{
		config.FusionAnalysisModeSeparate,
		config.FusionAnalysisModeOneCall,
		config.FusionAnalysisModeNone,
	} {
		t.Run(mode, func(t *testing.T) {
			server := newFusionStubServer(t, func(model, prompt string) (string, int) {
				if strings.Contains(prompt, "return only valid JSON") {
					return `{"consensus":[],"contradictions":[],"partial_coverage":[],"unique_insights":[],"blind_spots":[]}`, http.StatusOK
				}
				return "final answer", http.StatusOK
			})
			defer server.Close()

			req := newFusionTestRequest()
			req.CachedPanel = cachedTestPanel()
			req.Algorithm = &config.AlgorithmConfig{Type: config.DecisionAlgorithmFusion, Fusion: &config.FusionAlgorithmConfig{
				Model:                        "judge",
				AnalysisModels:               []string{"panel-a", "panel-b"},
				AnalysisMode:                 mode,
				IncludeAnalysis:              &includeNothing,
				IncludeIntermediateResponses: &includeNothing,
			}}
			resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
			require.NoError(t, err)

			var body map[string]interface{}
			require.NoError(t, json.Unmarshal(resp.Body, &body))
			assert.NotContains(t, body, "fusion", "mode alone must not emit a public trace envelope")

			trace, ok := resp.IntermediateResponses.(*FusionTrace)
			require.True(t, ok, "internal trace type = %T", resp.IntermediateResponses)
			assert.Equal(t, mode, trace.AnalysisMode)
			assert.Nil(t, trace.Analysis)
			assert.Empty(t, trace.Responses)
		})
	}
}

func TestFusionAnalysisModeFailureContracts(t *testing.T) {
	t.Run("separate analysis transport failure falls back", func(t *testing.T) {
		var judgeCalls atomic.Int64
		server := newFusionStubServer(t, func(model, prompt string) (string, int) {
			require.Equal(t, "judge", model)
			call := judgeCalls.Add(1)
			if call == 1 {
				return "analysis unavailable", http.StatusBadGateway
			}
			return "fallback final", http.StatusOK
		})
		defer server.Close()

		resp, err := executeCachedFusionMode(t, server.URL, config.FusionAnalysisModeSeparate, "")
		require.NoError(t, err)
		assert.Equal(t, "fallback final", extractMessageContent(t, resp.Body))
		assert.Equal(t, int64(2), judgeCalls.Load())
		assert.Equal(t, TokenUsage{PromptTokens: 60, CompletionTokens: 9, TotalTokens: 69}, resp.Usage)
	})

	t.Run("separate analysis parse failure records raw evidence and falls back", func(t *testing.T) {
		var judgeCalls atomic.Int64
		server := newFusionStubServer(t, func(model, prompt string) (string, int) {
			require.Equal(t, "judge", model)
			if judgeCalls.Add(1) == 1 {
				return "not structured json", http.StatusOK
			}
			return "fallback final", http.StatusOK
		})
		defer server.Close()

		resp, err := executeCachedFusionMode(t, server.URL, config.FusionAnalysisModeSeparate, "")
		require.NoError(t, err)
		assert.Equal(t, "fallback final", extractMessageContent(t, resp.Body))
		assert.Equal(t, int64(2), judgeCalls.Load())
		assert.Equal(t, TokenUsage{PromptTokens: 90, CompletionTokens: 13, TotalTokens: 103}, resp.Usage)

		var body struct {
			Fusion struct {
				Analysis *FusionAnalysis `json:"analysis"`
			} `json:"fusion"`
		}
		require.NoError(t, json.Unmarshal(resp.Body, &body))
		require.NotNil(t, body.Fusion.Analysis)
		assert.True(t, body.Fusion.Analysis.ParseFailed)
		assert.Equal(t, "not structured json", body.Fusion.Analysis.Raw)
	})

	t.Run("separate final synthesis failure is terminal", func(t *testing.T) {
		var judgeCalls atomic.Int64
		server := newFusionStubServer(t, func(model, prompt string) (string, int) {
			require.Equal(t, "judge", model)
			if judgeCalls.Add(1) == 1 {
				return `{"consensus":["agree"],"contradictions":[],"partial_coverage":[],"unique_insights":[],"blind_spots":[]}`, http.StatusOK
			}
			return "final synthesis unavailable", http.StatusBadGateway
		})
		defer server.Close()

		_, err := executeCachedFusionMode(t, server.URL, config.FusionAnalysisModeSeparate, "")
		require.ErrorContains(t, err, "fusion final synthesis failed")
		assert.Equal(t, int64(2), judgeCalls.Load())
	})

	for _, mode := range []string{config.FusionAnalysisModeOneCall, config.FusionAnalysisModeNone} {
		t.Run(mode+" terminal failure", func(t *testing.T) {
			var judgeCalls atomic.Int64
			server := newFusionStubServer(t, func(model, prompt string) (string, int) {
				require.Equal(t, "judge", model)
				judgeCalls.Add(1)
				return "terminal failure", http.StatusBadGateway
			})
			defer server.Close()

			_, err := executeCachedFusionMode(t, server.URL, mode, "")
			require.ErrorContains(t, err, "fusion final synthesis failed")
			assert.Equal(t, int64(1), judgeCalls.Load())
		})
	}
}

func TestFusionAnalysisModeCustomSynthesisTemplateOwnsTerminalPrompt(t *testing.T) {
	for _, mode := range []string{config.FusionAnalysisModeOneCall, config.FusionAnalysisModeNone} {
		t.Run(mode, func(t *testing.T) {
			var prompts []string
			server := newFusionStubServer(t, func(model, prompt string) (string, int) {
				prompts = append(prompts, prompt)
				return "custom final", http.StatusOK
			})
			defer server.Close()

			req := newFusionTestRequest()
			req.CachedPanel = cachedTestPanel()
			req.Algorithm = &config.AlgorithmConfig{Type: config.DecisionAlgorithmFusion, Fusion: &config.FusionAlgorithmConfig{
				Model:             "judge",
				AnalysisModels:    []string{"panel-a", "panel-b"},
				AnalysisMode:      mode,
				SynthesisTemplate: "mode analysis={{analysis}} original={{original}} panel={{responses}}",
			}}
			resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
			require.NoError(t, err)
			assert.Equal(t, "custom final", extractMessageContent(t, resp.Body))
			require.Len(t, prompts, 1)
			assert.Contains(t, prompts[0], "analysis= original=compare the options")
			assert.Contains(t, prompts[0], "panel=Response 1 (panel-a)")
			assert.NotContains(t, prompts[0], "Final answer:")
		})
	}
}

func TestFusionAnalysisModeRemainsDecisionOwned(t *testing.T) {
	dst := fusionExecutionConfig{AnalysisMode: config.FusionAnalysisModeOneCall}
	mergeFusionRequestConfig(&dst, &config.FusionRequestConfig{
		Model:            "request-judge",
		AnalysisTemplate: "request analysis {{responses}}",
	})
	assert.Equal(t, config.FusionAnalysisModeOneCall, dst.AnalysisMode)
}

func TestFusionAnalysisModeRejectsInvalidTemplateBeforeCalls(t *testing.T) {
	var calls atomic.Int64
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		calls.Add(1)
		return "unexpected", http.StatusOK
	})
	defer server.Close()

	_, err := executeCachedFusionMode(t, server.URL, config.FusionAnalysisModeOneCall, "compare {{responses}}")
	require.ErrorContains(t, err, "analysis_template requires analysis_mode=\"separate\"")
	assert.Zero(t, calls.Load())
}

func TestFusionAnalysisModeTerminalCallsPreserveTools(t *testing.T) {
	for _, mode := range []string{
		config.FusionAnalysisModeSeparate,
		config.FusionAnalysisModeOneCall,
		config.FusionAnalysisModeNone,
	} {
		t.Run(mode, func(t *testing.T) {
			var (
				mu           sync.Mutex
				observations []fusionToolCallObservation
			)
			server := newFusionToolCallServer(t, func(observation fusionToolCallObservation) {
				mu.Lock()
				observations = append(observations, observation)
				mu.Unlock()
			})
			defer server.Close()

			var params openai.ChatCompletionNewParams
			require.NoError(t, json.Unmarshal([]byte(`{
				"model":"vllm-sr/fusion",
				"messages":[{"role":"user","content":"search before answering"}],
				"tools":[{"type":"function","function":{"name":"search","parameters":{"type":"object"}}}],
				"tool_choice":"auto"
			}`), &params))
			req := &Request{
				OriginalRequest: &params,
				DecisionName:    "fusion-test",
				CachedPanel:     cachedTestPanel(),
				Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmFusion, Fusion: &config.FusionAlgorithmConfig{
					Model: "judge", AnalysisModels: []string{"panel-a", "panel-b"}, AnalysisMode: mode,
				}},
			}
			_, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
			require.NoError(t, err)

			mu.Lock()
			got := append([]fusionToolCallObservation(nil), observations...)
			mu.Unlock()
			wantCalls := 1
			if mode == config.FusionAnalysisModeSeparate {
				wantCalls = 2
			}
			require.Len(t, got, wantCalls)
			for i, observation := range got {
				wantTools := i == len(got)-1
				assert.Equal(t, wantTools, observation.hasTools)
			}
		})
	}
}

func TestFusionAnalysisModeStreamingToolCallSuppressesModeOnlyTrace(t *testing.T) {
	server := newFusionToolCallServer(t, nil)
	defer server.Close()

	includeNothing := false
	req := newFusionTestRequest()
	req.IsStreaming = true
	req.CachedPanel = cachedTestPanel()
	req.Algorithm = &config.AlgorithmConfig{Type: config.DecisionAlgorithmFusion, Fusion: &config.FusionAlgorithmConfig{
		Model:                        "judge",
		AnalysisModels:               []string{"panel-a", "panel-b"},
		AnalysisMode:                 config.FusionAnalysisModeOneCall,
		IncludeAnalysis:              &includeNothing,
		IncludeIntermediateResponses: &includeNothing,
	}}

	resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, "text/event-stream", resp.ContentType)
	assert.NotContains(t, string(resp.Body), `"fusion":`)
	assert.Contains(t, string(resp.Body), `"finish_reason":"tool_calls"`)
	trace, ok := resp.IntermediateResponses.(*FusionTrace)
	require.True(t, ok, "internal trace type = %T", resp.IntermediateResponses)
	assert.Equal(t, config.FusionAnalysisModeOneCall, trace.AnalysisMode)
}

func executeCachedFusionMode(t *testing.T, endpoint, mode, analysisTemplate string) (*Response, error) {
	t.Helper()
	req := newFusionTestRequest()
	req.CachedPanel = cachedTestPanel()
	req.Algorithm = &config.AlgorithmConfig{Type: config.DecisionAlgorithmFusion, Fusion: &config.FusionAlgorithmConfig{
		Model:            "judge",
		AnalysisModels:   []string{"panel-a", "panel-b"},
		AnalysisMode:     mode,
		AnalysisTemplate: analysisTemplate,
	}}
	return NewFusionLooper(&config.LooperConfig{Endpoint: endpoint}).Execute(context.Background(), req)
}

func boolPtr(value bool) *bool { return &value }
