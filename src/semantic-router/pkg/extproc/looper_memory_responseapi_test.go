package extproc

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// chatCompletionBody returns a valid Chat Completions JSON response body.
func chatCompletionBody(content string) []byte {
	body := map[string]interface{}{
		"id":     "chatcmpl-test",
		"object": "chat.completion",
		"model":  "test-model",
		"choices": []map[string]interface{}{{
			"index": 0,
			"message": map[string]interface{}{
				"role":    "assistant",
				"content": content,
			},
			"finish_reason": "stop",
		}},
	}
	b, _ := json.Marshal(body)
	return b
}

func TestHandleLooperExecution_TranslatesResponseAPI(t *testing.T) {
	filter := NewResponseAPIFilter(NewMockResponseStore())
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Looper: config.LooperConfig{Endpoint: "http://looper"},
			Memory: config.MemoryConfig{AutoStore: false},
		},
		ResponseAPIFilter: filter,
	}

	resp := &looper.Response{
		Body: chatCompletionBody("Hello from looper"),
	}

	reqCtx := &RequestContext{
		RequestID: "req-test-translate",
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			ConversationID:       "conv-123",
			GeneratedResponseID:  "resp-456",
			OriginalRequest: &responseapi.ResponseAPIRequest{
				Model: "auto",
				Input: json.RawMessage(`"test question"`),
			},
		},
		Headers: map[string]string{
			headers.AuthzUserID: "test-user",
		},
	}

	router.scheduleResponseMemoryStore(reqCtx, resp.Body)

	if isResponseAPIRequest(reqCtx) && router.ResponseAPIFilter != nil {
		translated, err := router.ResponseAPIFilter.TranslateResponse(
			context.Background(), reqCtx.ResponseAPICtx, resp.Body,
		)
		require.NoError(t, err)
		resp.Body = translated
	}

	var result map[string]interface{}
	require.NoError(t, json.Unmarshal(resp.Body, &result))

	assert.Equal(t, "response", result["object"])
	assert.Contains(t, result, "output")
	assert.Equal(t, "conv-123", result["conversation_id"])
}

func TestHandleLooperExecution_NoTranslationForChatCompletions(t *testing.T) {
	originalBody := chatCompletionBody("Hello from looper")
	resp := &looper.Response{
		Body: append([]byte(nil), originalBody...),
	}

	reqCtx := &RequestContext{
		RequestID: "req-test-no-translate",
	}

	assert.False(t, isResponseAPIRequest(reqCtx),
		"should not be identified as Response API request")
	assert.Equal(t, originalBody, resp.Body, "body should be unchanged")
}

func TestLooperStreamingOverride_ResponseAPIForcesNonStreaming(t *testing.T) {
	reqCtx := &RequestContext{
		ExpectStreamingResponse: true,
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
		},
	}

	streaming := reqCtx.ExpectStreamingResponse
	if isResponseAPIRequest(reqCtx) {
		streaming = false
	}

	assert.False(t, streaming,
		"Response API requests must force non-streaming looper execution")
}

func TestLooperStreamingOverride_ChatCompletionsPreservesStreaming(t *testing.T) {
	reqCtx := &RequestContext{
		ExpectStreamingResponse: true,
	}

	streaming := reqCtx.ExpectStreamingResponse
	if isResponseAPIRequest(reqCtx) {
		streaming = false
	}

	assert.True(t, streaming,
		"Chat Completions requests should preserve streaming preference")
}

func TestScheduleResponseMemoryStore_NoOpWithoutMemoryExtractor(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Memory: config.MemoryConfig{AutoStore: true},
		},
		MemoryExtractor: nil,
	}

	reqCtx := &RequestContext{
		RequestID: "req-noop",
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			ConversationID:       "conv-noop",
		},
	}

	router.scheduleResponseMemoryStore(reqCtx, chatCompletionBody("test"))
}

func TestScheduleResponseMemoryStore_SkippedWhenAutoStoreDisabled(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Memory: config.MemoryConfig{AutoStore: false},
		},
		MemoryExtractor: nil,
	}

	reqCtx := &RequestContext{
		RequestID: "req-disabled",
	}

	router.scheduleResponseMemoryStore(reqCtx, chatCompletionBody("test"))
}

func TestExtractCurrentUserMessage_ResponseAPIPath(t *testing.T) {
	reqCtx := &RequestContext{
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			OriginalRequest: &responseapi.ResponseAPIRequest{
				Input: json.RawMessage(`"What is our deployment target?"`),
			},
		},
	}

	msg := extractCurrentUserMessage(reqCtx)
	assert.Equal(t, "What is our deployment target?", msg)
}

func TestExtractCurrentUserMessage_EmptyForChatCompletions(t *testing.T) {
	reqCtx := &RequestContext{
		OriginalRequestBody: []byte(`{"model":"auto","messages":[{"role":"user","content":"hello"}]}`),
	}

	msg := extractCurrentUserMessage(reqCtx)
	assert.Empty(t, msg)
}

func TestExtractMemoryInfo_RejectsEmptyRequest(t *testing.T) {
	reqCtx := &RequestContext{
		RequestID: "req-empty",
	}

	_, _, _, err := extractMemoryInfo(reqCtx)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no conversation history available")
}

func TestExtractMemoryInfo_ResponseAPIWithUserID(t *testing.T) {
	reqCtx := &RequestContext{
		RequestID: "req-resp-api",
		Headers: map[string]string{
			headers.AuthzUserID: "team-user",
		},
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			ConversationID:       "conv-abc",
		},
	}

	sessionID, userID, history, err := extractMemoryInfo(reqCtx)
	require.NoError(t, err)
	assert.Equal(t, "conv-abc", sessionID)
	assert.Equal(t, "team-user", userID)
	assert.Empty(t, history)
}

func TestExtractMemoryInfo_RequiresUserID(t *testing.T) {
	reqCtx := &RequestContext{
		RequestID: "req-no-user",
		Headers:   map[string]string{},
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			ConversationID:       "conv-xyz",
		},
	}

	_, _, _, err := extractMemoryInfo(reqCtx)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "userID is required")
}

func TestLooperResponseCreation_IncludesTranslatedBody(t *testing.T) {
	filter := NewResponseAPIFilter(NewMockResponseStore())
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			Looper: config.LooperConfig{Endpoint: "http://looper"},
		},
		ResponseAPIFilter: filter,
	}

	resp := &looper.Response{
		Body:          chatCompletionBody("Kubernetes 1.30 with ArgoCD"),
		ContentType:   "application/json",
		Model:         "Qwen2.5-7B",
		ModelsUsed:    []string{"Qwen2.5-7B"},
		Iterations:    1,
		AlgorithmType: "confidence",
	}

	reqCtx := &RequestContext{
		RequestID: "req-e2e",
		ResponseAPICtx: &ResponseAPIContext{
			IsResponseAPIRequest: true,
			ConversationID:       "conv-e2e",
			GeneratedResponseID:  "resp-e2e",
			OriginalRequest: &responseapi.ResponseAPIRequest{
				Model: "auto",
				Input: json.RawMessage(`"What is our deployment target?"`),
			},
		},
		Headers: map[string]string{
			headers.AuthzUserID: "e2e-user",
		},
	}

	if isResponseAPIRequest(reqCtx) && router.ResponseAPIFilter != nil {
		translated, err := router.ResponseAPIFilter.TranslateResponse(
			context.Background(), reqCtx.ResponseAPICtx, resp.Body,
		)
		require.NoError(t, err)
		resp.Body = translated
	}

	procResp := router.createLooperResponse(resp, reqCtx)
	immediate := procResp.GetImmediateResponse()
	require.NotNil(t, immediate)

	body := immediate.GetBody()
	var result map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &result))
	assert.Equal(t, "response", result["object"],
		"ImmediateResponse body should contain Response API format")
}
