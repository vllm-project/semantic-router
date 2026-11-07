package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/google/uuid"
	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const toolResultMediaWarning = "approximated;tool-result images and files sent as a user message after the results;tool_result.content"

func init() {
	pkgtestcases.Register("protocol-codec-tool-result-media-opt-in", pkgtestcases.TestCase{
		Description: "An image a coding agent read reaches a Chat backend as a user message after the tool result when the operator accepts that loss",
		Tags:        []string{"protocol-codec", "anthropic", "agents"},
		Fn:          testProtocolCodecToolResultMediaOptIn,
	})
}

// testProtocolCodecToolResultMediaOptIn sends the turn after an agent read an
// image file: the image is the tool result. Chat tool messages carry text only,
// so the opt-in Router of the response-api profile must send the image as a user
// message right after the tool message.
func testProtocolCodecToolResultMediaOptIn(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	routerOpts := opts
	routerOpts.ServiceConfig = pkgtestcases.ServiceConfig{
		Namespace: ToolResultOptInNamespace, Name: ToolResultOptInRelease, ServicePort: ToolResultOptInPort,
	}
	session, err := fixtures.OpenServiceSession(ctx, client, routerOpts)
	if err != nil {
		return err
	}
	defer session.Close()
	provider, err := openProtocolCodecProviderSession(ctx, client, opts, "openai.chat.v1")
	if err != nil {
		return err
	}
	defer provider.Close()
	for _, stream := range []bool{false, true} {
		body := map[string]any{
			"model": "mock/opt-in-chat", "max_tokens": 64, "stream": stream,
			"messages": []any{
				map[string]any{"role": "user", "content": "Describe plot.png"},
				map[string]any{"role": "assistant", "content": []any{
					map[string]any{"type": "tool_use", "id": "call_1", "name": "Read", "input": map[string]any{"file_path": "plot.png"}},
				}},
				map[string]any{"role": "user", "content": []any{
					map[string]any{"type": "tool_result", "tool_use_id": "call_1", "content": []any{
						map[string]any{"type": "image", "source": map[string]any{"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="}},
					}},
				}},
			},
		}
		sessionID := "tool-result-media-opt-in-" + uuid.NewString()
		result, err := sendProtocolMatrixRaw(ctx, session, "/v1/messages", body, stream,
			map[string]string{"x-vsr-test-session-id": sessionID})
		if err != nil {
			return fmt.Errorf("opt-in tool-result image request (stream=%t): %w", stream, err)
		}
		if result.StatusCode != http.StatusOK {
			return fmt.Errorf("opted-in tool-result image was not dispatched (stream=%t): status=%d body=%s",
				stream, result.StatusCode, truncateString(string(result.Body), 500))
		}
		if warnings := result.Headers.Get("x-vsr-protocol-warnings"); !strings.Contains(warnings, toolResultMediaWarning) {
			return fmt.Errorf("x-vsr-protocol-warnings = %q, want %q (stream=%t)", warnings, toolResultMediaWarning, stream)
		}
		raw, err := lastProviderSimulatorRequest(ctx, provider, sessionID)
		if err != nil {
			return err
		}
		if err := assertImageFollowsToolMessage(raw); err != nil {
			return fmt.Errorf("%w (stream=%t): %s", err, stream, truncateString(string(raw), 800))
		}
	}
	return nil
}

func assertImageFollowsToolMessage(raw []byte) error {
	// The simulator returns the captured request under body.
	var captured struct {
		Body struct {
			Messages []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		} `json:"body"`
	}
	if err := json.Unmarshal(raw, &captured); err != nil {
		return fmt.Errorf("backend request is not Chat JSON: %w", err)
	}
	request := captured.Body
	for i, message := range request.Messages {
		if message.Role != "tool" {
			continue
		}
		if strings.Contains(string(message.Content), "image_url") {
			return fmt.Errorf("the tool message still carries the image")
		}
		if i+1 >= len(request.Messages) || request.Messages[i+1].Role != "user" ||
			!strings.Contains(string(request.Messages[i+1].Content), "data:image/png;base64,aW1hZ2U=") {
			return fmt.Errorf("the image does not follow the tool message as a user message")
		}
		return nil
	}
	return fmt.Errorf("the backend request has no tool message")
}
