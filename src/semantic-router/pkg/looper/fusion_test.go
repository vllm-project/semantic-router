package looper

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestFusionLooperExecutesPanelJudgeAndFinal(t *testing.T) {
	var seenModels []string
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		seenModels = append(seenModels, model)
		switch model {
		case "panel-a":
			return "panel a answer", http.StatusOK
		case "panel-b":
			return "panel b answer", http.StatusOK
		case "judge":
			if strings.Contains(prompt, "return only valid JSON") {
				return `{"consensus":["both answer"],"contradictions":[],"partial_coverage":[],"unique_insights":["b"],"blind_spots":[]}`, http.StatusOK
			}
			assert.Contains(t, prompt, "Structured analysis:")
			return "final fusion answer", http.StatusOK
		default:
			return "unexpected model", http.StatusInternalServerError
		}
	})
	defer server.Close()

	req := newFusionTestRequest()
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:          "judge",
			AnalysisModels: []string{"panel-a", "panel-b"},
		},
	}

	resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
	require.NoError(t, err)
	require.Equal(t, "application/json", resp.ContentType)
	assert.Equal(t, "fusion", resp.AlgorithmType)
	assert.Equal(t, "judge", resp.Model)
	assert.Equal(t, []string{"panel-a", "panel-b", "judge"}, resp.ModelsUsed)
	assert.Equal(t, 4, resp.Iterations)
	assert.ElementsMatch(t, []string{"panel-a", "panel-b", "judge", "judge"}, seenModels)

	var body map[string]interface{}
	require.NoError(t, json.Unmarshal(resp.Body, &body))
	assert.Equal(t, "judge", body["model"])
	fusionTrace := body["fusion"].(map[string]interface{})
	assert.Len(t, fusionTrace["responses"], 2)
	assert.Equal(t, "judge", fusionTrace["judge_model"])
}

func TestFusionLooperRecordsPartialPanelFailures(t *testing.T) {
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		switch model {
		case "panel-a":
			return "panel a answer", http.StatusOK
		case "panel-b":
			return "failed", http.StatusBadGateway
		case "judge":
			if strings.Contains(prompt, "return only valid JSON") {
				return `{"consensus":["a"],"contradictions":[],"partial_coverage":[],"unique_insights":[],"blind_spots":["b failed"]}`, http.StatusOK
			}
			return "final from partial panel", http.StatusOK
		default:
			return "unexpected model", http.StatusInternalServerError
		}
	})
	defer server.Close()

	req := newFusionTestRequest()
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:          "judge",
			AnalysisModels: []string{"panel-a", "panel-b"},
		},
	}

	resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, []string{"panel-a", "panel-b", "judge"}, resp.ModelsUsed)
	assert.Equal(t, 4, resp.Iterations)

	var body map[string]interface{}
	require.NoError(t, json.Unmarshal(resp.Body, &body))
	fusionTrace := body["fusion"].(map[string]interface{})
	require.Len(t, fusionTrace["failed_models"], 1)
	failed := fusionTrace["failed_models"].([]interface{})[0].(map[string]interface{})
	assert.Equal(t, "panel-b", failed["model"])
}

func TestFusionLooperAllPanelFailuresReturnError(t *testing.T) {
	server := newFusionStubServer(t, func(model, prompt string) (string, int) {
		return "failed", http.StatusBadGateway
	})
	defer server.Close()

	req := newFusionTestRequest()
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:          "judge",
			AnalysisModels: []string{"panel-a", "panel-b"},
		},
	}

	_, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "all 2 analysis models failed")
}

func TestFusionLooperUsesDecisionModelRefs(t *testing.T) {
	looper := NewFusionLooper(&config.LooperConfig{Endpoint: "http://looper"})
	req := newFusionTestRequest()
	req.ModelRefs = []config.ModelRef{
		{Model: "panel-a"},
		{Model: "panel-b"},
	}
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model: "judge",
		},
	}

	cfg := looper.resolveFusionExecutionConfig(req)

	assert.Equal(t, "judge", cfg.Model)
	assert.Equal(t, []string{"panel-a", "panel-b"}, cfg.AnalysisModels)
}

func TestFusionLooperRejectsInvalidOnError(t *testing.T) {
	req := newFusionTestRequest()
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:          "judge",
			AnalysisModels: []string{"panel-a"},
			OnError:        "ignore",
		},
	}

	_, err := NewFusionLooper(&config.LooperConfig{Endpoint: "http://looper"}).Execute(context.Background(), req)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "fusion on_error must be")
}

func TestFusionAnalysisPromptRequestsCompactJSON(t *testing.T) {
	prompt := buildFusionAnalysisPrompt(fusionExecutionConfig{}, "question", []*ModelResponse{
		{Model: "panel-a", Content: "answer"},
	})

	assert.Contains(t, prompt, "return only valid JSON")
	assert.Contains(t, prompt, "Return compact JSON only")
	assert.Contains(t, prompt, "no markdown")
	assert.Contains(t, prompt, "no code fences")
	assert.Contains(t, prompt, "at most two concise strings")
}

func TestParseFusionAnalysisAcceptsFencedJSON(t *testing.T) {
	analysis, err := parseFusionAnalysis("```json\n{\"consensus\":[\"agree\"],\"contradictions\":[],\"partial_coverage\":[],\"unique_insights\":[],\"blind_spots\":[]}\n```")

	require.NoError(t, err)
	require.NotNil(t, analysis)
	assert.Equal(t, []string{"agree"}, analysis.Consensus)
	assert.False(t, analysis.ParseFailed)
}

func newFusionTestRequest() *Request {
	params := openai.ChatCompletionNewParams{
		Model: "vllm-sr/fusion",
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("compare the options"),
		},
	}
	return &Request{
		OriginalRequest: &params,
		DecisionName:    "fusion-test",
	}
}

func newFusionStubServer(
	t *testing.T,
	respond func(model string, prompt string) (content string, status int),
) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "1", r.Header.Get("x-vsr-fusion-depth"))
		var payload struct {
			Model    string `json:"model"`
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		prompt := ""
		if len(payload.Messages) > 0 {
			prompt = payload.Messages[len(payload.Messages)-1].Content
		}
		content, status := respond(payload.Model, prompt)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":      "chatcmpl-test",
			"object":  "chat.completion",
			"created": 1,
			"model":   payload.Model,
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": content,
					},
					"finish_reason": "stop",
				},
			},
		})
	}))
}
