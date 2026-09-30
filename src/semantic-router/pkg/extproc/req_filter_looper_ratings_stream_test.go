package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestRatingsStreamingSurvivesFinalExtProcBoundary(t *testing.T) {
	for _, includeUsage := range []bool{false, true} {
		t.Run(fmt.Sprintf("include_usage_%t", includeUsage), func(t *testing.T) {
			assertRatingsStreamingBoundary(t, includeUsage)
		})
	}
}

func assertRatingsStreamingBoundary(t *testing.T, includeUsage bool) {
	t.Helper()
	text := strings.Repeat("中文🚀", 50)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(strings.ReplaceAll(string(extProcStreamFixture(llmprotocol.OpenAIChatV1)), "hello", text)))
	}))
	defer server.Close()
	router := &OpenAIRouter{Config: &config.RouterConfig{Looper: config.LooperConfig{Endpoint: server.URL}}}
	decision := &config.Decision{
		Name: "ratings-stream", ModelRefs: []config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
		Algorithm: &config.AlgorithmConfig{Type: "ratings", Ratings: &config.RatingsAlgorithmConfig{OnError: "fail"}},
	}
	request := testNeutralRequest("public-model", "hello")
	request.Stream = true
	request.StreamOptions.IncludeUsage = &includeUsage
	ctx := &RequestContext{
		SourceFormat: llmprotocol.OpenAIChatV1, SemanticRequest: request,
		ExpectStreamingResponse: true, TraceContext: context.Background(), Headers: map[string]string{},
	}
	response, err := router.handleLooperExecution(context.Background(), request, decision, ctx)
	require.NoError(t, err)
	immediate := response.GetImmediateResponse()
	require.EqualValues(t, 200, immediate.GetStatus().GetCode(), string(immediate.GetBody()))
	require.Contains(t, string(immediate.GetBody()), "data: [DONE]\n\n")
	if includeUsage {
		require.Contains(t, string(immediate.GetBody()), `"total_tokens":6`)
	} else {
		require.NotContains(t, string(immediate.GetBody()), `"usage"`)
	}
	texts, stops := ratingsStreamChoices(t, immediate.GetBody())
	require.Equal(t, map[int]string{0: text, 1: text}, texts)
	require.Equal(t, map[int]string{0: "stop", 1: "stop"}, stops)
	require.NotNil(t, ctx.SemanticResponse)
	require.Len(t, ctx.SemanticResponse.Alternatives, 1)
	require.EqualValues(t, 6, *ctx.SemanticResponse.Usage.Total.Value)
	require.Equal(t, text, ctx.SemanticResponse.Output[0].Content[0].Text)
	require.Equal(t, text, ctx.SemanticResponse.Alternatives[0][0].Content[0].Text)
	var first map[string]interface{}
	frame := strings.Split(string(immediate.GetBody()), "\n\n")[0]
	require.NoError(t, json.Unmarshal([]byte(strings.TrimPrefix(frame, "data: ")), &first))
	require.Equal(t, ctx.SemanticResponse.ID, first["id"])
}

func ratingsStreamChoices(t *testing.T, body []byte) (map[int]string, map[int]string) {
	t.Helper()
	texts, stops := map[int]string{}, map[int]string{}
	for _, frame := range strings.Split(string(body), "\n\n") {
		data := strings.TrimPrefix(frame, "data: ")
		if data == "" || data == "[DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Index int `json:"index"`
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
				Finish string `json:"finish_reason"`
			} `json:"choices"`
		}
		require.NoError(t, json.Unmarshal([]byte(data), &chunk))
		for _, choice := range chunk.Choices {
			texts[choice.Index] += choice.Delta.Content
			if choice.Finish != "" {
				stops[choice.Index] = choice.Finish
			}
		}
	}
	return texts, stops
}
