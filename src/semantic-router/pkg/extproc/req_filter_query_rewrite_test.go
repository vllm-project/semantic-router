package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type mockChatRequest struct {
	Model    string            `json:"model"`
	Messages []mockChatMessage `json:"messages"`
}

type mockChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type mockChatResponse struct {
	Choices []mockChatChoice `json:"choices"`
}

type mockChatChoice struct {
	Message mockChatMessage `json:"message"`
}

func createMockRouterConfig(serverURL string) *config.RouterConfig {
	var address string
	var port int
	if serverURL != "" {
		hostPort := serverURL[7:] // skip "http://"
		for i := len(hostPort) - 1; i >= 0; i-- {
			if hostPort[i] == ':' {
				address = hostPort[:i]
				portStr := hostPort[i+1:]
				for _, c := range portStr {
					port = port*10 + int(c-'0')
				}
				break
			}
		}
	}

	return &config.RouterConfig{
		ExternalModels: []config.ExternalModelConfig{
			{
				Provider:  "vllm",
				ModelRole: config.ModelRoleMemoryRewrite,
				ModelEndpoint: config.ClassifierVLLMEndpoint{
					Address: address,
					Port:    port,
				},
				ModelName:      "test-model",
				TimeoutSeconds: 5,
			},
		},
	}
}

func TestBuildSearchQuery_NoExternalModel(t *testing.T) {
	history := []ConversationMessage{
		{Role: "user", Content: "Planning vacation to Hawaii"},
	}

	result, err := BuildSearchQuery(context.Background(), history, "How much?", nil)
	require.NoError(t, err)
	assert.Equal(t, "How much?", result, "should return original when no routerCfg")

	routerCfg := &config.RouterConfig{}
	result, err = BuildSearchQuery(context.Background(), nil, "test query", routerCfg)
	require.NoError(t, err)
	assert.Equal(t, "test query", result, "should return original when no external models")
}

func TestBuildSearchQuery_WithMockLLM(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1/chat/completions", r.URL.Path)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

		var req mockChatRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		require.NoError(t, err)

		assert.Equal(t, "test-model", req.Model)
		assert.Len(t, req.Messages, 2)
		assert.Equal(t, "system", req.Messages[0].Role)
		assert.Equal(t, "user", req.Messages[1].Role)
		assert.Contains(t, req.Messages[1].Content, "How much?")
		assert.Contains(t, req.Messages[1].Content, "Hawaii")

		resp := mockChatResponse{
			Choices: []mockChatChoice{
				{Message: mockChatMessage{Content: "What is the budget for the Hawaii vacation?"}},
			},
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)

	history := []ConversationMessage{
		{Role: "user", Content: "Planning vacation to Hawaii"},
		{Role: "assistant", Content: "Hawaii sounds great! What's your budget?"},
	}

	result, err := BuildSearchQuery(context.Background(), history, "How much?", routerCfg)
	require.NoError(t, err)
	assert.Equal(t, "What is the budget for the Hawaii vacation?", result)
}

func TestBuildSearchQuery_SelfContainedQuery(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := mockChatResponse{
			Choices: []mockChatChoice{
				{Message: mockChatMessage{Content: "What is the capital of France?"}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)

	result, err := BuildSearchQuery(context.Background(), nil, "What is the capital of France?", routerCfg)
	require.NoError(t, err)
	assert.Equal(t, "What is the capital of France?", result, "self-contained query should remain unchanged")
}

func TestBuildSearchQuery_LLMError_FallbackToOriginal(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)
	result, err := BuildSearchQuery(context.Background(), nil, "original query", routerCfg)
	require.NoError(t, err)
	assert.Equal(t, "original query", result, "should fallback to original on error")
}

func TestBuildSearchQuery_CleanupQuotes(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := mockChatResponse{
			Choices: []mockChatChoice{
				{Message: mockChatMessage{Content: `"What is my budget for Hawaii?"`}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)
	result, err := BuildSearchQuery(context.Background(), nil, "How much?", routerCfg)
	require.NoError(t, err)
	assert.Equal(t, "What is my budget for Hawaii?", result, "should strip quotes")
}

func TestBuildSearchQuery_CleanupWhitespace(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := mockChatResponse{
			Choices: []mockChatChoice{
				{Message: mockChatMessage{Content: "  What is the budget?  \n"}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)

	result, err := BuildSearchQuery(context.Background(), nil, "How much?", routerCfg)
	require.NoError(t, err)
	assert.Equal(t, "What is the budget?", result, "should trim whitespace")
}

func TestBuildSearchQuery_EmptyChoices(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := mockChatResponse{
			Choices: []mockChatChoice{},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	routerCfg := createMockRouterConfig(server.URL)

	result, err := BuildSearchQuery(context.Background(), nil, "original query", routerCfg)
	require.NoError(t, err)
	assert.Equal(t, "original query", result, "should fallback on empty choices")
}
