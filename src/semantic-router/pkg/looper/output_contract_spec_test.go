package looper

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func looperAnswerContent(t *testing.T, body []byte) string {
	t.Helper()
	var completion struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal(body, &completion))
	require.Len(t, completion.Choices, 1)
	return completion.Choices[0].Message.Content
}

// TestReMoMAppliesChoiceOutputContract renders the synthesis's choice by the
// decision's choice contract, as Fusion and Flow do.
func TestReMoMAppliesChoiceOutputContract(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload := decodeReMoMTestRequest(t, r)
		writeReMoMTestCompletion(t, w, payload.Model, map[string]string{"model-a": "C", "model-final": "B"}[payload.Model])
	}))
	defer server.Close()
	req := newReMoMTestRequest(&config.ReMoMAlgorithmConfig{
		BreadthSchedule:   []int{1},
		ModelDistribution: remomDistributionRoundRobin,
		SynthesisModel:    "model-final",
	}, false, []config.ModelRef{{Model: "model-a"}, {Model: "model-final"}})
	req.OutputContractSpec = singleChoiceOutputContractSpecForTest()
	req.OutputContractSpec.Render = &config.OutputContractRenderSpec{
		Mode: config.OutputContractRenderModeTemplate, Template: "The answer is {choice}.",
	}

	response, err := NewReMoMLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)

	require.NoError(t, err)
	assert.Equal(t, "The answer is B.", looperAnswerContent(t, response.Body))
}

// TestFusionDereferencesSelectedPanelResponse serves the panel response the
// judge selects by its number, as ReMoM serves the selected reference.
func TestFusionDereferencesSelectedPanelResponse(t *testing.T) {
	for _, mode := range []string{config.FusionAnalysisModeSeparate, config.FusionAnalysisModeOneCall} {
		t.Run(mode, func(t *testing.T) {
			server := newFusionStubServer(t, func(model, prompt string) (string, int) {
				switch model {
				case "panel-a":
					return "panel a answer", http.StatusOK
				case "panel-b":
					return "panel b answer", http.StatusOK
				}
				if strings.Contains(prompt, "return only valid JSON") {
					return `{"consensus":["Paris"],"contradictions":[],"partial_coverage":[],"unique_insights":[],"blind_spots":[]}`, http.StatusOK
				}
				return "2", http.StatusOK
			})
			defer server.Close()
			req := newFusionTestRequest()
			req.Algorithm = &config.AlgorithmConfig{Type: "fusion", Fusion: &config.FusionAlgorithmConfig{
				Model: "judge", AnalysisModels: []string{"panel-a", "panel-b"}, AnalysisMode: mode,
			}}
			req.OutputContractSpec = referenceDereferenceOutputContractSpecForTest()

			response, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)

			require.NoError(t, err)
			assert.Equal(t, "panel b answer", looperAnswerContent(t, response.Body))
		})
	}
}

// TestReferenceSelectionNeverServesReasoning keeps the judge's own answer
// when the reference it selects carries only reasoning: a panel's or a
// round's reasoning is never published as the answer, buffered or streamed.
func TestReferenceSelectionNeverServesReasoning(t *testing.T) {
	const private = "private scratchwork"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload := decodeReMoMTestRequest(t, r)
		switch payload.Model {
		case "panel-a", "model-reasoning":
			writeReMoMTestCompletionWithReasoning(t, w, payload.Model, "", private, TokenUsage{})
		case "panel-b":
			writeReMoMTestCompletion(t, w, payload.Model, "panel b answer")
		default:
			writeReMoMTestCompletion(t, w, payload.Model, "1")
		}
	}))
	defer server.Close()
	runs := map[string]func(streaming bool) (*Response, error){
		"fusion": func(streaming bool) (*Response, error) {
			req := newFusionTestRequest()
			// The panel trace, which publishes panel reasoning when enabled, is
			// off, so the answer is the only place reasoning could appear.
			req.Algorithm = &config.AlgorithmConfig{Type: "fusion", Fusion: &config.FusionAlgorithmConfig{
				Model: "judge", AnalysisModels: []string{"panel-a", "panel-b"}, AnalysisMode: config.FusionAnalysisModeOneCall,
				IncludeIntermediateResponses: new(bool),
			}}
			req.IsStreaming = streaming
			req.OutputContractSpec = referenceDereferenceOutputContractSpecForTest()
			return NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
		},
		"remom": func(streaming bool) (*Response, error) {
			req := newReMoMTestRequest(&config.ReMoMAlgorithmConfig{
				BreadthSchedule: []int{1}, ModelDistribution: remomDistributionRoundRobin, SynthesisModel: "model-final",
			}, streaming, []config.ModelRef{{Model: "model-reasoning"}, {Model: "model-final"}})
			req.OutputContractSpec = referenceDereferenceOutputContractSpecForTest()
			return NewReMoMLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
		},
	}
	for name, run := range runs {
		for _, streaming := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/streaming=%t", name, streaming), func(t *testing.T) {
				response, err := run(streaming)
				require.NoError(t, err)
				assert.NotContains(t, string(response.Body), private)
				if !streaming {
					assert.Equal(t, "1", looperAnswerContent(t, response.Body))
				}
			})
		}
	}
}

// TestFusionReferenceSelectionKeepsJudgeToolCalls leaves a judge answer that
// calls a tool as it is, as the choice contract does.
func TestFusionReferenceSelectionKeepsJudgeToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload := decodeReMoMTestRequest(t, r)
		if payload.Model != "judge" {
			writeReMoMTestCompletion(t, w, payload.Model, payload.Model+" answer")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"c","object":"chat.completion","created":0,"model":"judge","choices":[{"index":0,` +
			`"message":{"role":"assistant","content":"2","tool_calls":[{"id":"call_lookup","type":"function",` +
			`"function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`))
	}))
	defer server.Close()
	var tools openai.ChatCompletionNewParams
	require.NoError(t, json.Unmarshal([]byte(`{"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]}`), &tools))
	req := newFusionTestRequest()
	req.OriginalRequest.Tools = tools.Tools
	req.Algorithm = &config.AlgorithmConfig{Type: "fusion", Fusion: &config.FusionAlgorithmConfig{
		Model: "judge", AnalysisModels: []string{"panel-a", "panel-b"}, AnalysisMode: config.FusionAnalysisModeOneCall,
	}}
	req.OutputContractSpec = referenceDereferenceOutputContractSpecForTest()

	response, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)

	require.NoError(t, err)
	var completion struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				ToolCalls []struct {
					ID string `json:"id"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	require.NoError(t, json.Unmarshal(response.Body, &completion))
	require.Len(t, completion.Choices, 1)
	assert.Equal(t, "tool_calls", completion.Choices[0].FinishReason)
	require.Len(t, completion.Choices[0].Message.ToolCalls, 1)
	assert.Equal(t, "call_lookup", completion.Choices[0].Message.ToolCalls[0].ID)
}
