package testcases

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"strings"

	"github.com/google/uuid"
	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const deferLoadingLossWarning = "dropped;tools are sent in full instead of loading on demand and tool references reach the backend as the tool name;tools.defer_loading"

func init() {
	pkgtestcases.Register("protocol-codec-defer-loading-opt-in", pkgtestcases.TestCase{
		Description: "A Claude Code tool search turn reaches a Chat backend with every tool in full when the operator accepts the defer_loading loss",
		Tags:        []string{"protocol-codec", "anthropic", "agents"},
		Fn:          testProtocolCodecDeferLoadingOptIn,
	})
}

// testProtocolCodecDeferLoadingOptIn sends the turn after a tool search the way
// Claude Code does: deferred tools carry defer_loading and the search result is
// a tool_reference. The opt-in Router of the response-api profile accepts the
// loss, so the backend must see the tool in full and the reference as its name.
func testProtocolCodecDeferLoadingOptIn(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
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
				map[string]any{"role": "user", "content": "Fetch the page"},
				map[string]any{"role": "assistant", "content": []any{
					map[string]any{"type": "tool_use", "id": "call_1", "name": "ToolSearch", "input": map[string]any{"query": "select:WebFetch"}},
				}},
				map[string]any{"role": "user", "content": []any{
					map[string]any{"type": "tool_result", "tool_use_id": "call_1", "content": []any{
						map[string]any{"type": "tool_reference", "tool_name": "WebFetch"},
					}},
					map[string]any{"type": "text", "text": "Tool loaded."},
				}},
			},
			"tools": []any{
				map[string]any{"name": "ToolSearch", "input_schema": map[string]any{"type": "object"}},
				map[string]any{"name": "WebFetch", "input_schema": map[string]any{"type": "object"}, "defer_loading": true},
			},
		}
		sessionID := "defer-loading-opt-in-" + uuid.NewString()
		result, err := sendProtocolMatrixRaw(ctx, session, "/v1/messages", body, stream,
			map[string]string{"x-vsr-test-session-id": sessionID})
		if err != nil {
			return fmt.Errorf("opt-in tool search request (stream=%t): %w", stream, err)
		}
		if result.StatusCode != http.StatusOK {
			return fmt.Errorf("opted-in tool search turn was not dispatched (stream=%t): status=%d body=%s",
				stream, result.StatusCode, truncateString(string(result.Body), 500))
		}
		if warnings := result.Headers.Get("x-vsr-protocol-warnings"); !strings.Contains(warnings, deferLoadingLossWarning) {
			return fmt.Errorf("x-vsr-protocol-warnings = %q, want %q (stream=%t)", warnings, deferLoadingLossWarning, stream)
		}
		raw, err := lastProviderSimulatorRequest(ctx, provider, sessionID)
		if err != nil {
			return err
		}
		if bytes.Contains(raw, []byte("defer_loading")) || bytes.Contains(raw, []byte("tool_reference")) ||
			bytes.Count(raw, []byte("WebFetch")) < 2 {
			return fmt.Errorf("backend must receive WebFetch as a full tool and as the search result (stream=%t): %s",
				stream, truncateString(string(raw), 800))
		}
	}
	return nil
}
