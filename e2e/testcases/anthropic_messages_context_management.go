package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("anthropic-messages-context-management", pkgtestcases.TestCase{
		Description: "context_management on /v1/messages survives buffered and streaming dispatch to the Anthropic backend",
		Tags:        []string{"anthropic", "protocol-codec", "routing", "streaming"},
		Fn:          testAnthropicMessagesContextManagement,
	})
}

// contextManagementDirective is the prompt-trimming directive Claude Code
// sends on every /v1/messages request. It changes what the upstream bills for
// the turn, so the router must forward it to an Anthropic-format backend
// byte-for-byte instead of dropping it during routing.
var contextManagementDirective = map[string]any{
	"edits": []any{map[string]any{"type": "clear_thinking_20251015", "keep": "all"}},
}

// testAnthropicMessagesContextManagement drives the real Envoy + ExtProc
// dispatch path: a /v1/messages request carrying context_management must
// return 200 and the anthropic-shim backend must have received the directive
// unchanged, on both the buffered and the streaming route.
func testAnthropicMessagesContextManagement(
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

	sessionID := fmt.Sprintf("context-management-%d", time.Now().UnixNano())
	request := map[string]any{
		"model":              "MoM",
		"max_tokens":         32,
		"messages":           []any{map[string]any{"role": "user", "content": "Say hello."}},
		"context_management": contextManagementDirective,
	}

	body, err := sendProtocolMatrixRequestWithHeaders(
		ctx, session, "/v1/messages", request, false,
		map[string]string{"x-vsr-test-session-id": sessionID},
	)
	if err != nil {
		return fmt.Errorf("buffered /v1/messages with context_management failed: %w", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(body, &parsed); err != nil {
		return fmt.Errorf("buffered response is not valid JSON: %w (body=%s)", err, truncateString(string(body), 200))
	}
	if err := validateForwardedContextManagement(ctx, backendSession, sessionID); err != nil {
		return fmt.Errorf("buffered dispatch lost context_management: %w", err)
	}

	streamRequest := cloneMap(request)
	streamRequest["stream"] = true
	stream, err := sendProtocolMatrixRequestWithHeaders(
		ctx, session, "/v1/messages", streamRequest, true,
		map[string]string{"x-vsr-test-session-id": sessionID},
	)
	if err != nil {
		return fmt.Errorf("streaming /v1/messages with context_management failed: %w", err)
	}
	if !bytes.Contains(stream, []byte("message_start")) || !bytes.Contains(stream, []byte("message_stop")) {
		return fmt.Errorf("context-managed Messages stream is invalid: %s", truncateString(string(stream), 600))
	}
	if err := validateForwardedContextManagement(ctx, backendSession, sessionID); err != nil {
		return fmt.Errorf("streaming dispatch lost context_management: %w", err)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"buffered_directive_forwarded":  true,
			"streaming_directive_forwarded": true,
		})
	}
	return nil
}

// validateForwardedContextManagement reads the shim's capture of the most
// recent backend request for the session and requires the forwarded body to
// carry the directive exactly as the client sent it.
func validateForwardedContextManagement(
	ctx context.Context,
	backendSession *fixtures.ServiceSession,
	sessionID string,
) error {
	forwarded, err := lastProviderSimulatorRequest(ctx, backendSession, sessionID)
	if err != nil {
		return err
	}
	var debug struct {
		Body struct {
			ContextManagement any `json:"context_management"`
		} `json:"body"`
	}
	if err := json.Unmarshal(forwarded, &debug); err != nil {
		return fmt.Errorf("decode provider request: %w", err)
	}
	var expected any
	canonical, err := json.Marshal(contextManagementDirective)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(canonical, &expected); err != nil {
		return err
	}
	if !reflect.DeepEqual(debug.Body.ContextManagement, expected) {
		return fmt.Errorf("forwarded body changed or dropped context_management: %s", truncateString(string(forwarded), 800))
	}
	return nil
}
