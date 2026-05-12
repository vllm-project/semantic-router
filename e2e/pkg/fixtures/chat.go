package fixtures

import (
	"context"
	"encoding/json"
	"net/http"
	"time"
)

// ChatMessage is the minimal OpenAI chat message shape used by E2E contracts.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatTool is a minimal OpenAI-format tool definition for E2E requests.
type ChatTool struct {
	Type     string       `json:"type"`
	Function ChatToolFunc `json:"function"`
}

// ChatToolFunc is the function payload for a chat tool.
type ChatToolFunc struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// ChatCompletionsRequest is the typed request for /v1/chat/completions.
type ChatCompletionsRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`
	// User is optional; forwarded for per-user routing and session correlation in tests.
	User string `json:"user,omitempty"`
	// Tools is optional; used by tool_selection filter-mode E2E contracts.
	Tools []ChatTool `json:"tools,omitempty"`
	// ToolChoice is optional; when omitted the gateway uses default auto behavior.
	ToolChoice json.RawMessage `json:"tool_choice,omitempty"`
}

// ChatCompletionsClient talks to the routed chat-completions API.
type ChatCompletionsClient struct {
	baseURL    string
	httpClient *http.Client
}

// NewChatCompletionsClient binds a chat client to a port-forward session.
func NewChatCompletionsClient(session *ServiceSession, timeout time.Duration) *ChatCompletionsClient {
	return &ChatCompletionsClient{
		baseURL:    session.BaseURL(),
		httpClient: session.HTTPClient(timeout),
	}
}

// Create sends a typed chat-completions request.
func (c *ChatCompletionsClient) Create(
	ctx context.Context,
	request ChatCompletionsRequest,
	headers map[string]string,
) (*HTTPResponse, error) {
	return doJSONRequest(ctx, c.httpClient, http.MethodPost, c.baseURL+"/v1/chat/completions", request, headers)
}
