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

const unreadableFileWarning = "dropped;1 files the selected model cannot read sent as a note;content.file"

func init() {
	pkgtestcases.Register("protocol-codec-unreadable-file-opt-in", pkgtestcases.TestCase{
		Description: "A PDF a coding agent read reaches a model without file input as a note when the operator accepts that loss",
		Tags:        []string{"protocol-codec", "anthropic", "agents"},
		Fn:          testProtocolCodecUnreadableFileOptIn,
	})
}

// testProtocolCodecUnreadableFileOptIn sends the turn after an agent read a PDF:
// the file is a document block in the tool result. The opt-in Router's chat
// model declares no file input, so the backend must receive a note in place of
// the file instead of the session failing on every later call.
func testProtocolCodecUnreadableFileOptIn(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
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
				map[string]any{"role": "user", "content": "Summarize report.pdf"},
				map[string]any{"role": "assistant", "content": []any{
					map[string]any{"type": "tool_use", "id": "call_1", "name": "Read", "input": map[string]any{"file_path": "report.pdf"}},
				}},
				map[string]any{"role": "user", "content": []any{
					map[string]any{"type": "tool_result", "tool_use_id": "call_1", "content": []any{
						map[string]any{"type": "text", "text": "PDF file read: report.pdf"},
						map[string]any{"type": "document", "source": map[string]any{"type": "base64", "media_type": "application/pdf", "data": "JVBERi0xLjMK"}},
					}},
				}},
			},
		}
		sessionID := "unreadable-file-opt-in-" + uuid.NewString()
		result, err := sendProtocolMatrixRaw(ctx, session, "/v1/messages", body, stream,
			map[string]string{"x-vsr-test-session-id": sessionID})
		if err != nil {
			return fmt.Errorf("opt-in PDF request (stream=%t): %w", stream, err)
		}
		if result.StatusCode != http.StatusOK {
			return fmt.Errorf("opted-in PDF was not dispatched (stream=%t): status=%d body=%s",
				stream, result.StatusCode, truncateString(string(result.Body), 500))
		}
		if warnings := result.Headers.Get("x-vsr-protocol-warnings"); !strings.Contains(warnings, unreadableFileWarning) {
			return fmt.Errorf("x-vsr-protocol-warnings = %q, want %q (stream=%t)", warnings, unreadableFileWarning, stream)
		}
		raw, err := lastProviderSimulatorRequest(ctx, provider, sessionID)
		if err != nil {
			return err
		}
		if bytes.Contains(raw, []byte("JVBERi0xLjMK")) || !bytes.Contains(raw, []byte("was not sent because the selected model cannot read files")) {
			return fmt.Errorf("backend must receive a note in place of the PDF (stream=%t): %s", stream, truncateString(string(raw), 800))
		}
	}
	return nil
}
