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

// Claude Code's Artifact tool description is about 20 KiB. The response-api
// profile's opt-in Router raises global.router.protocol_limits; its main Router
// keeps the 16 KiB default.
const artifactToolDescriptionBytes = 20_225

func init() {
	pkgtestcases.Register("protocol-codec-tool-description-limit", pkgtestcases.TestCase{
		Description: "A tool description over the default limit is rejected by default and reaches the backend when the operator raises protocol_limits",
		Tags:        []string{"protocol-codec", "anthropic", "agents"},
		Fn:          testProtocolCodecToolDescriptionLimit,
	})
}

func testProtocolCodecToolDescriptionLimit(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	description := "artifact-" + strings.Repeat("d", artifactToolDescriptionBytes-len("artifact-"))

	defaultSession, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	rejected, err := sendProtocolMatrixRaw(ctx, defaultSession, "/v1/messages",
		toolDescriptionLimitBody(chatBackendModel, description), false, nil)
	defaultSession.Close()
	if err != nil {
		return fmt.Errorf("default-limit request: %w", err)
	}
	want := fmt.Sprintf("tools[0].description is %d bytes and exceeds the configured limit of 16384 bytes", artifactToolDescriptionBytes)
	if rejected.StatusCode != http.StatusBadRequest || !bytes.Contains(rejected.Body, []byte(want)) {
		return fmt.Errorf("default Router must reject the tool with %q: status=%d body=%s",
			want, rejected.StatusCode, truncateString(string(rejected.Body), 500))
	}
	if bytes.Contains(rejected.Body, []byte("artifact-")) {
		return fmt.Errorf("tool_text_limit error echoed the tool description")
	}

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
		sessionID := "tool-description-limit-" + uuid.NewString()
		body := toolDescriptionLimitBody("mock/opt-in-chat", description)
		body["stream"] = stream
		result, sendErr := sendProtocolMatrixRaw(ctx, session, "/v1/messages", body, stream,
			map[string]string{"x-vsr-test-session-id": sessionID})
		if sendErr != nil {
			return fmt.Errorf("raised-limit request (stream=%t): %w", stream, sendErr)
		}
		if result.StatusCode != http.StatusOK {
			return fmt.Errorf("raised limit did not admit the tool (stream=%t): status=%d body=%s",
				stream, result.StatusCode, truncateString(string(result.Body), 500))
		}
		raw, captureErr := lastProviderSimulatorRequest(ctx, provider, sessionID)
		if captureErr != nil {
			return captureErr
		}
		if !bytes.Contains(raw, []byte(description)) {
			return fmt.Errorf("backend did not receive the whole tool description (stream=%t)", stream)
		}
	}
	return nil
}

func toolDescriptionLimitBody(model, description string) map[string]any {
	return map[string]any{
		"model": model, "max_tokens": 64,
		"messages": []any{map[string]any{"role": "user", "content": "Draft a short note."}},
		"tools": []any{map[string]any{
			"name": "Artifact", "description": description,
			"input_schema": map[string]any{"type": "object"},
		}},
	}
}
