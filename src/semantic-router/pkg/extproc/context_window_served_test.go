package extproc

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/parity"
)

// longWindowConfig is a recipe with a long-context lane, as production's, on
// a model with a 1,048,576-token window. The lane's context band makes the
// Router count requests with its learned token estimate.
const longWindowConfig = `
version: v0.5
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    timeout: 30s
providers:
  models:
    - name: long-model
      backend_refs:
        - name: long-primary
          endpoint: BACKEND
          protocol: http
          provider: vllm
      context_window_size: 1048576
recipes:
  - name: default
    signals:
      context:
        - name: long_context
          min_tokens: 200K
    decisions:
      - name: long_context
        priority: 20
        rules:
          operator: AND
          conditions:
            - type: context
              name: long_context
        modelRefs:
          - model: long-model
      - name: default_route
        priority: 10
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: long-model
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
global:
  services:
    response_api:
      enabled: true
      store_backend: memory
      ttl_seconds: 3600
      max_responses: 100
`

const englishSentence = "The committee reviewed the quarterly budget and agreed to fund the new library wing next spring. "

// englishPromptOfTokens is English prose that a tokenizer packing 4.18 bytes
// a token, as DeepSeek's does English, counts at tokens.
func englishPromptOfTokens(tokens int) string {
	size := tokens * 418 / 100
	return strings.Repeat(englishSentence, size/len(englishSentence)+1)[:size]
}

func chatRequestBody(t *testing.T, text string) string {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"model":    "vllm-sr/auto",
		"messages": []map[string]string{{"role": "user", "content": text}},
	})
	require.NoError(t, err)
	return string(body)
}

const servedChatCompletion = `{"id":"chatcmpl-long","object":"chat.completion","created":1700000000,` +
	`"model":"long-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},` +
	`"finish_reason":"stop"}]}`

func longWindowRouter(t *testing.T, yaml string) *OpenAIRouter {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(yaml))
	require.NoError(t, err)
	return newParityRouter(t, cfg)
}

func runLongWindowCase(t *testing.T, router *OpenAIRouter, name, path, body string) *parity.Record {
	t.Helper()
	headers := [][]string{{"content-type", "application/json"}}
	if path == "/v1/messages" {
		headers = append(headers, []string{"anthropic-version", "2023-06-01"})
	}
	return parity.NewRecorder(router, routing.DefaultOptions).Run(context.Background(), parity.Case{
		Name:    name,
		Request: parity.CaseRequest{Method: "POST", Path: path, Headers: headers, Body: body},
		Upstream: &parity.Upstream{
			Status: 200, Headers: [][]string{{"content-type", "application/json"}}, Body: servedChatCompletion,
		},
	})
}

func requireServedByLongModel(t *testing.T, record *parity.Record) {
	t.Helper()
	require.Empty(t, record.Error)
	require.NotNil(t, record.Response)
	require.Equal(t, 200, record.Response.Status, "refused: %s", record.Response.Body)
	require.Equal(t, "long-model", record.Route)
	require.NotNil(t, record.Upstream)
}

// A single English message of 1,040,000 tokens fits a 1,048,576-token window.
// The Router counted it at a quarter-token a byte, 1,087,880, and refused it
// with 422 before any backend could count it (#6319).
func TestAPromptThatFitsTheWindowIsServed(t *testing.T) {
	router := longWindowRouter(t, strings.Replace(longWindowConfig, "BACKEND", "127.0.0.1:18100", 1))
	record := runLongWindowCase(t, router, "english-1040000", "/v1/chat/completions",
		chatRequestBody(t, englishPromptOfTokens(1_040_000)))
	requireServedByLongModel(t, record)
}

// A Router that has seen dense traffic (Chinese prose, JSON, logs) learned a
// lower bytes-per-token ratio. English of 1,000,000 tokens must still be
// served: the learned ratio may place context bands, never refuse (#6319).
func TestDenseTrafficDoesNotLowerTheWindow(t *testing.T) {
	router := longWindowRouter(t, strings.Replace(longWindowConfig, "BACKEND", "127.0.0.1:18100", 1))
	recipe, ok := router.Config.RecipeByName("default")
	require.True(t, ok)
	for range 40 {
		ctx := &RequestContext{VSRContextTextBytes: 290_000}
		ctx.Routing.SelectRecipe(recipe)
		router.calibrateTokenEstimator(ctx, 100_000)
	}
	ctx := &RequestContext{}
	ctx.Routing.SelectRecipe(recipe)
	_, conservative, samples, calibrated := router.classifierForRequest(ctx).TokenCalibrationRatio("")
	require.True(t, calibrated, "after %d samples", samples)
	require.Less(t, conservative, 3.0)

	record := runLongWindowCase(t, router, "english-after-dense", "/v1/chat/completions",
		chatRequestBody(t, englishPromptOfTokens(1_000_000)))
	requireServedByLongModel(t, record)
}
