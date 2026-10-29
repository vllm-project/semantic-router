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

// The response-api profile deploys this second, standalone Router with
// global.router.allow_lossy_translation: [tool_result.is_error]. Its main
// Router keeps the default, so both behaviours run against one simulator.
const (
	ToolResultOptInRelease   = "semantic-router-opt-in"
	ToolResultOptInNamespace = "vllm-sr-opt-in-system"
	ToolResultOptInPort      = "8899"
)

const toolResultLossWarning = "approximated;target format cannot represent tool-result failure status;tool_result.is_error"

func init() {
	pkgtestcases.Register("protocol-codec-tool-result-error-opt-in", pkgtestcases.TestCase{
		Description: "A failed tool result reaches Chat and Responses backends with a protocol warning when the operator accepts that loss",
		Tags:        []string{"protocol-codec", "anthropic", "agents"},
		Fn:          testProtocolCodecToolResultErrorOptIn,
	})
}

func testProtocolCodecToolResultErrorOptIn(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	routerOpts := opts
	routerOpts.ServiceConfig = pkgtestcases.ServiceConfig{
		Namespace: ToolResultOptInNamespace, Name: ToolResultOptInRelease, ServicePort: ToolResultOptInPort,
	}
	session, err := fixtures.OpenServiceSession(ctx, client, routerOpts)
	if err != nil {
		return err
	}
	defer session.Close()
	for _, backend := range []struct{ model, format string }{
		{"mock/opt-in-chat", "openai.chat.v1"},
		{"mock/opt-in-responses", "openai.responses.v1"},
	} {
		provider, providerErr := openProtocolCodecProviderSession(ctx, client, opts, backend.format)
		if providerErr != nil {
			return providerErr
		}
		runErr := runToolResultErrorOptIn(ctx, session, provider, backend.model, backend.format)
		provider.Close()
		if runErr != nil {
			return runErr
		}
	}
	return nil
}

// runToolResultErrorOptIn sends the failed command the way a coding agent
// does: the command's output is the tool-result content and is_error marks the
// failure. The backend must receive the output, never the flag.
func runToolResultErrorOptIn(ctx context.Context, session, provider *fixtures.ServiceSession, model, backendFormat string) error {
	const output = "Exit code 2: ls: cannot access /nope: No such file or directory"
	for _, stream := range []bool{false, true} {
		body := map[string]any{
			"model": model, "max_tokens": 64, "stream": stream,
			"messages": []any{
				map[string]any{"role": "user", "content": "List /nope"},
				map[string]any{"role": "assistant", "content": []any{
					map[string]any{"type": "tool_use", "id": "call_1", "name": "Bash", "input": map[string]any{"command": "ls /nope"}},
				}},
				map[string]any{"role": "user", "content": []any{
					map[string]any{"type": "tool_result", "tool_use_id": "call_1", "content": output, "is_error": true},
				}},
			},
		}
		sessionID := "tool-result-error-opt-in-" + uuid.NewString()
		result, err := sendProtocolMatrixRaw(ctx, session, "/v1/messages", body, stream,
			map[string]string{"x-vsr-test-session-id": sessionID})
		if err != nil {
			return fmt.Errorf("opt-in tool-result request (backend=%s stream=%t): %w", backendFormat, stream, err)
		}
		if result.StatusCode != http.StatusOK {
			return fmt.Errorf("opted-in failed tool result was not dispatched (backend=%s stream=%t): status=%d body=%s",
				backendFormat, stream, result.StatusCode, truncateString(string(result.Body), 500))
		}
		if warnings := result.Headers.Get("x-vsr-protocol-warnings"); !strings.Contains(warnings, toolResultLossWarning) {
			return fmt.Errorf("x-vsr-protocol-warnings = %q, want %q (backend=%s stream=%t)", warnings, toolResultLossWarning, backendFormat, stream)
		}
		raw, captureErr := lastProviderSimulatorRequest(ctx, provider, sessionID)
		if captureErr != nil {
			return captureErr
		}
		if !bytes.Contains(raw, []byte("No such file or directory")) || bytes.Contains(raw, []byte("is_error")) {
			return fmt.Errorf("backend must receive the command output without is_error (backend=%s stream=%t): %s",
				backendFormat, stream, truncateString(string(raw), 800))
		}
		if stream {
			err = assertAnthropicTextStream(result.Body, "tool result accepted")
		} else {
			err = assertAnthropicText(result.Body, "tool result accepted")
		}
		if err != nil {
			return fmt.Errorf("opt-in client response (backend=%s stream=%t): %w", backendFormat, stream, err)
		}
	}
	return nil
}
