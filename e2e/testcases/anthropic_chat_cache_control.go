package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("anthropic-chat-cache-control", pkgtestcases.TestCase{
		Description: "OpenAI-compatible cache_control survives buffered and streaming dispatch to Anthropic",
		Tags:        []string{"anthropic", "cache", "protocol-codec", "streaming"},
		Fn:          testAnthropicChatCacheControl,
	})
}

func testAnthropicChatCacheControl(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	backendOpts := opts
	backendOpts.ServiceConfig = pkgtestcases.ServiceConfig{
		Namespace:   "anthropic-backend-system",
		Name:        "anthropic-backend-qwen",
		ServicePort: "8080",
	}
	backendSession, err := fixtures.OpenServiceSession(ctx, client, backendOpts)
	if err != nil {
		return err
	}
	defer backendSession.Close()

	sessionID := fmt.Sprintf("chat-cache-%d", time.Now().UnixNano())
	request := map[string]any{
		"model": "MoM",
		"messages": []any{map[string]any{
			"role": "user",
			"content": []any{map[string]any{
				"type": "text", "text": "Reusable context",
				"cache_control": map[string]any{"type": "ephemeral", "ttl": "5m"},
			}},
		}},
	}
	if err := validateBufferedAnthropicChatCache(ctx, session, request, sessionID); err != nil {
		return err
	}
	if err := validateStreamedAnthropicChatCache(ctx, session, backendSession, request, sessionID); err != nil {
		return err
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"buffered_cache_hit": true, "streaming_marker_preserved": true})
	}
	return nil
}

func validateBufferedAnthropicChatCache(
	ctx context.Context,
	session *fixtures.ServiceSession,
	request map[string]any,
	sessionID string,
) error {
	for attempt := 0; attempt < 2; attempt++ {
		body, err := sendChatCacheRequest(ctx, session, request, sessionID)
		if err != nil {
			return err
		}
		if attempt == 0 {
			continue
		}
		cached, err := chatCachedInputTokens(body)
		if err != nil {
			return err
		}
		if cached <= 0 {
			return fmt.Errorf("repeat request reported no cached input tokens: %s", truncateString(string(body), 600))
		}
	}
	return nil
}

func validateStreamedAnthropicChatCache(
	ctx context.Context,
	session *fixtures.ServiceSession,
	backendSession *fixtures.ServiceSession,
	request map[string]any,
	sessionID string,
) error {
	streamRequest := cloneMap(request)
	streamRequest["stream"] = true
	stream, err := sendProtocolMatrixRequestWithHeaders(
		ctx,
		session,
		"/v1/chat/completions",
		streamRequest,
		true,
		map[string]string{"x-vsr-test-session-id": sessionID},
	)
	if err != nil {
		return err
	}
	if !bytes.Contains(stream, []byte("chat.completion.chunk")) || !bytes.Contains(stream, []byte("data: [DONE]")) {
		return fmt.Errorf("cache-marked Chat stream is invalid: %s", truncateString(string(stream), 600))
	}

	forwarded, err := lastAnthropicShimRequest(ctx, backendSession, sessionID)
	if err != nil {
		return err
	}
	if !hasForwardedCacheMarker(forwarded, "ephemeral", "5m") {
		return fmt.Errorf("streaming dispatch lost cache_control before the Anthropic backend: %s", truncateString(string(forwarded), 800))
	}
	return nil
}

func hasForwardedCacheMarker(body []byte, markerType, ttl string) bool {
	var debug struct {
		Body struct {
			Messages []struct {
				Content []struct {
					CacheControl *struct {
						Type string `json:"type"`
						TTL  string `json:"ttl"`
					} `json:"cache_control"`
				} `json:"content"`
			} `json:"messages"`
		} `json:"body"`
	}
	if json.Unmarshal(body, &debug) != nil {
		return false
	}
	for _, message := range debug.Body.Messages {
		for _, content := range message.Content {
			if content.CacheControl != nil && content.CacheControl.Type == markerType && content.CacheControl.TTL == ttl {
				return true
			}
		}
	}
	return false
}

func sendChatCacheRequest(
	ctx context.Context,
	session *fixtures.ServiceSession,
	body map[string]any,
	sessionID string,
) ([]byte, error) {
	return sendProtocolMatrixRequestWithHeaders(
		ctx,
		session,
		"/v1/chat/completions",
		body,
		false,
		map[string]string{"x-vsr-test-session-id": sessionID},
	)
}

func chatCachedInputTokens(body []byte) (int64, error) {
	var response struct {
		Usage struct {
			PromptTokensDetails struct {
				CachedTokens int64 `json:"cached_tokens"`
			} `json:"prompt_tokens_details"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return 0, err
	}
	return response.Usage.PromptTokensDetails.CachedTokens, nil
}

func lastAnthropicShimRequest(
	ctx context.Context,
	session *fixtures.ServiceSession,
	sessionID string,
) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, session.BaseURL()+"/debug/last-request", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("x-vsr-test-session-id", sessionID)
	resp, err := session.HTTPClient(15 * time.Second).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		return nil, readErr
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("debug request returned HTTP %d: %s", resp.StatusCode, truncateString(string(body), 500))
	}
	return body, nil
}

func cloneMap(source map[string]any) map[string]any {
	result := make(map[string]any, len(source)+1)
	for key, value := range source {
		result[key] = value
	}
	return result
}
