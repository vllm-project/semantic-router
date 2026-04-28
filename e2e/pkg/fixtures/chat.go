package fixtures

import (
	"context"
	"net/http"
	"time"
)

// ChatMessage is the minimal OpenAI chat message shape used by E2E contracts.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatCompletionsRequest is the typed request for /v1/chat/completions.
type ChatCompletionsRequest struct {
	Model    string        `json:"model"`
	Messages []ChatMessage `json:"messages"`
	// User is optional; forwarded for per-user routing and session correlation in tests.
	User string `json:"user,omitempty"`
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
