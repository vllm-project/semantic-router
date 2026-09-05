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

const (
	promptCacheEnabledProbe  = "__PROMPT_CACHE_ENABLED__"
	promptCacheDisabledProbe = "__PROMPT_CACHE_DISABLED__"
)

func init() {
	pkgtestcases.Register("anthropic-prompt-cache-policy", pkgtestcases.TestCase{
		Description: "Route-local policy injects bounded Anthropic cache markers and preserves caller markers",
		Tags:        []string{"anthropic", "cache", "plugin", "streaming"},
		Fn:          testAnthropicPromptCachePolicy,
	})
}

func testAnthropicPromptCachePolicy(
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

	if err := validatePromptCachePolicyInsertion(ctx, session, backendSession, false, opts.Verbose); err != nil {
		return err
	}
	if err := validatePromptCachePolicyInsertion(ctx, session, backendSession, true, opts.Verbose); err != nil {
		return err
	}
	if err := validatePromptCachePolicyEmptyInstruction(ctx, session, backendSession, opts.Verbose); err != nil {
		return err
	}
	if err := validatePromptCachePolicyCallerPrecedence(ctx, session, backendSession, opts.Verbose); err != nil {
		return err
	}
	if err := validateDisabledPromptCachePolicy(ctx, session, backendSession, opts.Verbose); err != nil {
		return err
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"buffered_inserted":  true,
			"streaming_inserted": true,
			"empty_skipped":      true,
			"caller_preserved":   true,
			"disabled_unchanged": true,
		})
	}
	return nil
}

func validatePromptCachePolicyInsertion(
	ctx context.Context,
	session *fixtures.ServiceSession,
	backendSession *fixtures.ServiceSession,
	stream bool,
	verbose bool,
) error {
	sessionID := fmt.Sprintf("prompt-cache-insert-%t-%d", stream, time.Now().UnixNano())
	request := promptCachePolicyRequest(promptCacheEnabledProbe, stream)
	response, err := sendPromptCachePolicyRequest(ctx, session, request, sessionID, stream)
	if err != nil {
		return err
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("prompt cache insertion returned HTTP %d: %s", response.StatusCode, truncateString(string(response.Body), 600))
	}
	if stream && (!bytes.Contains(response.Body, []byte("chat.completion.chunk")) ||
		!bytes.Contains(response.Body, []byte("data: [DONE]"))) {
		return fmt.Errorf("prompt cache stream is invalid: %s", truncateString(string(response.Body), 600))
	}
	if err := expectPromptCacheReceipt(response.Headers, "inserted", "", "2", ""); err != nil {
		return fmt.Errorf("prompt cache insertion receipt: %w", err)
	}
	forwarded, err := lastProviderSimulatorRequest(ctx, backendSession, sessionID)
	if err != nil {
		return err
	}
	providerRequest, err := decodePromptCacheProviderRequest(forwarded)
	if err != nil {
		return err
	}
	if err := validateInjectedPromptCacheMarkers(providerRequest, forwarded); err != nil {
		return fmt.Errorf("prompt cache insertion changed provider request: %w", err)
	}
	if verbose {
		mode := "buffered"
		if stream {
			mode = "streaming"
		}
		fmt.Printf(
			"[PromptCache] mode=%s action=%q reason=%q inserted=%q preserved=%q instruction_markers=%d tool_markers=%d message_markers=%d instruction_ttl=%q tool_ttl=%q\n",
			mode,
			response.Headers.Get("x-vsr-prompt-cache-action"),
			response.Headers.Get("x-vsr-prompt-cache-reason"),
			response.Headers.Get("x-vsr-prompt-cache-inserted"),
			response.Headers.Get("x-vsr-prompt-cache-preserved"),
			promptCacheBlockMarkerCount(providerRequest.System),
			promptCacheToolMarkerCount(providerRequest.Tools),
			promptCacheMessageMarkerCount(providerRequest.Messages),
			promptCacheBlockMarkerTTL(providerRequest.System),
			promptCacheToolMarkerTTL(providerRequest.Tools),
		)
	}
	return nil
}

func validatePromptCachePolicyEmptyInstruction(
	ctx context.Context,
	session *fixtures.ServiceSession,
	backendSession *fixtures.ServiceSession,
	verbose bool,
) error {
	sessionID := fmt.Sprintf("prompt-cache-empty-%d", time.Now().UnixNano())
	request := promptCachePolicyRequest(promptCacheEnabledProbe, false)
	system := request["messages"].([]any)[0].(map[string]any)["content"].([]any)
	request["messages"].([]any)[0].(map[string]any)["content"] = append(
		system,
		map[string]any{"type": "text", "text": ""},
	)
	delete(request, "tools")

	response, err := sendPromptCachePolicyRequest(ctx, session, request, sessionID, false)
	if err != nil {
		return err
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("empty instruction request returned HTTP %d: %s", response.StatusCode, truncateString(string(response.Body), 600))
	}
	if err := expectPromptCacheReceipt(response.Headers, "inserted", "", "1", ""); err != nil {
		return fmt.Errorf("empty instruction receipt: %w", err)
	}
	forwarded, err := lastProviderSimulatorRequest(ctx, backendSession, sessionID)
	if err != nil {
		return err
	}
	providerRequest, err := decodePromptCacheProviderRequest(forwarded)
	if err != nil {
		return err
	}
	if len(providerRequest.System) != 3 ||
		!matchesCacheControl(providerRequest.System[1].CacheControl, "ephemeral", "1h") ||
		providerRequest.System[2].CacheControl != nil {
		return fmt.Errorf("empty instruction marker mismatch: %s", truncateString(string(forwarded), 800))
	}
	if verbose {
		fmt.Printf(
			"[PromptCache] mode=empty-trailing action=%q inserted=%q instruction_markers=%d empty_marked=%t\n",
			response.Headers.Get("x-vsr-prompt-cache-action"),
			response.Headers.Get("x-vsr-prompt-cache-inserted"),
			promptCacheBlockMarkerCount(providerRequest.System),
			providerRequest.System[2].CacheControl != nil,
		)
	}
	return nil
}

func validatePromptCachePolicyCallerPrecedence(
	ctx context.Context,
	session *fixtures.ServiceSession,
	backendSession *fixtures.ServiceSession,
	verbose bool,
) error {
	sessionID := fmt.Sprintf("prompt-cache-preserve-%d", time.Now().UnixNano())
	request := promptCachePolicyRequest(promptCacheEnabledProbe, false)
	system := request["messages"].([]any)[0].(map[string]any)["content"].([]any)
	system[0].(map[string]any)["cache_control"] = map[string]any{
		"type": "ephemeral",
		"ttl":  "5m",
	}

	response, err := sendPromptCachePolicyRequest(ctx, session, request, sessionID, false)
	if err != nil {
		return err
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("caller marker request returned HTTP %d: %s", response.StatusCode, truncateString(string(response.Body), 600))
	}
	if err := expectPromptCacheReceipt(response.Headers, "preserved", "caller_markers", "", "1"); err != nil {
		return fmt.Errorf("caller marker receipt: %w", err)
	}
	forwarded, err := lastProviderSimulatorRequest(ctx, backendSession, sessionID)
	if err != nil {
		return err
	}
	providerRequest, err := decodePromptCacheProviderRequest(forwarded)
	if err != nil {
		return err
	}
	if err := validatePreservedPromptCacheMarker(providerRequest, forwarded); err != nil {
		return fmt.Errorf("caller marker precedence failed: %w", err)
	}
	if verbose {
		fmt.Printf(
			"[PromptCache] mode=caller action=%q reason=%q inserted=%q preserved=%q instruction_markers=%d tool_markers=%d message_markers=%d instruction_ttl=%q\n",
			response.Headers.Get("x-vsr-prompt-cache-action"),
			response.Headers.Get("x-vsr-prompt-cache-reason"),
			response.Headers.Get("x-vsr-prompt-cache-inserted"),
			response.Headers.Get("x-vsr-prompt-cache-preserved"),
			promptCacheBlockMarkerCount(providerRequest.System),
			promptCacheToolMarkerCount(providerRequest.Tools),
			promptCacheMessageMarkerCount(providerRequest.Messages),
			promptCacheBlockMarkerTTL(providerRequest.System),
		)
	}
	return nil
}

func validateDisabledPromptCachePolicy(
	ctx context.Context,
	session *fixtures.ServiceSession,
	backendSession *fixtures.ServiceSession,
	verbose bool,
) error {
	sessionID := fmt.Sprintf("prompt-cache-disabled-%d", time.Now().UnixNano())
	response, err := sendPromptCachePolicyRequest(
		ctx,
		session,
		promptCachePolicyRequest(promptCacheDisabledProbe, false),
		sessionID,
		false,
	)
	if err != nil {
		return err
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("disabled prompt cache request returned HTTP %d: %s", response.StatusCode, truncateString(string(response.Body), 600))
	}
	if err := expectPromptCacheReceipt(response.Headers, "", "", "", ""); err != nil {
		return fmt.Errorf("disabled prompt cache receipt: %w", err)
	}
	forwarded, err := lastProviderSimulatorRequest(ctx, backendSession, sessionID)
	if err != nil {
		return err
	}
	providerRequest, err := decodePromptCacheProviderRequest(forwarded)
	if err != nil {
		return err
	}
	if err := validateNoPromptCacheMarkers(providerRequest, forwarded); err != nil {
		return fmt.Errorf("disabled prompt cache changed provider request: %w", err)
	}
	if verbose {
		fmt.Printf(
			"[PromptCache] mode=disabled action=%q reason=%q inserted=%q preserved=%q instruction_markers=%d tool_markers=%d message_markers=%d\n",
			response.Headers.Get("x-vsr-prompt-cache-action"),
			response.Headers.Get("x-vsr-prompt-cache-reason"),
			response.Headers.Get("x-vsr-prompt-cache-inserted"),
			response.Headers.Get("x-vsr-prompt-cache-preserved"),
			promptCacheBlockMarkerCount(providerRequest.System),
			promptCacheToolMarkerCount(providerRequest.Tools),
			promptCacheMessageMarkerCount(providerRequest.Messages),
		)
	}
	return nil
}

func promptCachePolicyRequest(probe string, stream bool) map[string]any {
	return map[string]any{
		"model":      "MoM",
		"max_tokens": 16,
		"stream":     stream,
		"messages": []any{
			map[string]any{
				"role": "system",
				"content": []any{
					map[string]any{"type": "text", "text": "Stable preface"},
					map[string]any{"type": "text", "text": "Reusable instructions"},
				},
			},
			map[string]any{
				"role":    "user",
				"content": probe + " answer with one word",
			},
		},
		"tools": []any{
			promptCachePolicyTool("lookup", "Look up a record"),
			promptCachePolicyTool("search", "Search records"),
		},
	}
}

func promptCachePolicyTool(name string, description string) map[string]any {
	return map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        name,
			"description": description,
			"parameters": map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
		},
	}
}

func sendPromptCachePolicyRequest(
	ctx context.Context,
	session *fixtures.ServiceSession,
	body map[string]any,
	sessionID string,
	stream bool,
) (*localChatCompletionResponse, error) {
	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		session.BaseURL()+"/v1/chat/completions",
		bytes.NewReader(encoded),
	)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-vsr-debug", "true")
	req.Header.Set("x-vsr-test-session-id", sessionID)
	if stream {
		req.Header.Set("Accept", "text/event-stream")
	}
	resp, err := session.HTTPClient(45 * time.Second).Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	return &localChatCompletionResponse{
		StatusCode: resp.StatusCode,
		Headers:    resp.Header,
		Body:       responseBody,
	}, nil
}

func expectPromptCacheReceipt(
	headers http.Header,
	action string,
	reason string,
	inserted string,
	preserved string,
) error {
	expected := map[string]string{
		"x-vsr-prompt-cache-action":    action,
		"x-vsr-prompt-cache-reason":    reason,
		"x-vsr-prompt-cache-inserted":  inserted,
		"x-vsr-prompt-cache-preserved": preserved,
	}
	for name, want := range expected {
		if got := headers.Get(name); got != want {
			return fmt.Errorf("%s = %q, want %q", name, got, want)
		}
	}
	return nil
}

func validateInjectedPromptCacheMarkers(
	forwarded promptCacheProviderRequest,
	body []byte,
) error {
	if len(forwarded.System) != 2 ||
		forwarded.System[0].CacheControl != nil ||
		!matchesCacheControl(forwarded.System[1].CacheControl, "ephemeral", "1h") {
		return fmt.Errorf("instruction marker mismatch: %s", truncateString(string(body), 800))
	}
	if len(forwarded.Tools) != 2 ||
		forwarded.Tools[0].CacheControl != nil ||
		!matchesCacheControl(forwarded.Tools[1].CacheControl, "ephemeral", "1h") {
		return fmt.Errorf("tool marker mismatch: %s", truncateString(string(body), 800))
	}
	if promptCacheMessageMarkerCount(forwarded.Messages) != 0 {
		return fmt.Errorf("message content was marked: %s", truncateString(string(body), 800))
	}
	return nil
}

func validatePreservedPromptCacheMarker(
	forwarded promptCacheProviderRequest,
	body []byte,
) error {
	if len(forwarded.System) != 2 ||
		!matchesCacheControl(forwarded.System[0].CacheControl, "ephemeral", "5m") ||
		forwarded.System[1].CacheControl != nil {
		return fmt.Errorf("caller instruction marker mismatch: %s", truncateString(string(body), 800))
	}
	if promptCacheToolMarkerCount(forwarded.Tools) != 0 ||
		promptCacheMessageMarkerCount(forwarded.Messages) != 0 {
		return fmt.Errorf("router inserted markers beside caller marker: %s", truncateString(string(body), 800))
	}
	return nil
}

func validateNoPromptCacheMarkers(
	forwarded promptCacheProviderRequest,
	body []byte,
) error {
	if promptCacheBlockMarkerCount(forwarded.System) != 0 ||
		promptCacheToolMarkerCount(forwarded.Tools) != 0 ||
		promptCacheMessageMarkerCount(forwarded.Messages) != 0 {
		return fmt.Errorf("unexpected cache marker: %s", truncateString(string(body), 800))
	}
	return nil
}

type promptCacheProviderRequest struct {
	System   []forwardedCacheBlock `json:"system"`
	Messages []struct {
		Content []forwardedCacheBlock `json:"content"`
	} `json:"messages"`
	Tools []struct {
		CacheControl *forwardedCacheControl `json:"cache_control"`
	} `json:"tools"`
}

func decodePromptCacheProviderRequest(body []byte) (promptCacheProviderRequest, error) {
	var debug struct {
		Body promptCacheProviderRequest `json:"body"`
	}
	if err := json.Unmarshal(body, &debug); err != nil {
		return promptCacheProviderRequest{}, fmt.Errorf("decode provider request: %w", err)
	}
	return debug.Body, nil
}

func promptCacheBlockMarkerCount(blocks []forwardedCacheBlock) int {
	count := 0
	for _, block := range blocks {
		if block.CacheControl != nil {
			count++
		}
	}
	return count
}

func promptCacheBlockMarkerTTL(blocks []forwardedCacheBlock) string {
	for _, block := range blocks {
		if block.CacheControl != nil {
			return block.CacheControl.TTL
		}
	}
	return ""
}

func promptCacheMessageMarkerCount(messages []struct {
	Content []forwardedCacheBlock `json:"content"`
}) int {
	count := 0
	for _, message := range messages {
		count += promptCacheBlockMarkerCount(message.Content)
	}
	return count
}

func promptCacheToolMarkerCount(tools []struct {
	CacheControl *forwardedCacheControl `json:"cache_control"`
}) int {
	count := 0
	for _, tool := range tools {
		if tool.CacheControl != nil {
			count++
		}
	}
	return count
}

func promptCacheToolMarkerTTL(tools []struct {
	CacheControl *forwardedCacheControl `json:"cache_control"`
}) string {
	for _, tool := range tools {
		if tool.CacheControl != nil {
			return tool.CacheControl.TTL
		}
	}
	return ""
}
