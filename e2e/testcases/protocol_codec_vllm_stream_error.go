package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const (
	vllmStreamErrorMarker    = "__mock_vllm_stream_error__"
	vllmMidstreamErrorMarker = "__mock_vllm_midstream_error__"
	vllmGrammarErrorPrefix   = "Grammar error: regex parse error"
)

func init() {
	pkgtestcases.Register("protocol-codec-vllm-stream-error", pkgtestcases.TestCase{
		Description: "vLLM's in-stream error event, whose code is an integer HTTP status, reaches every client with vLLM's message",
		Tags:        []string{"protocol-codec", "vllm", "anthropic", "responses", "errors", "streaming"},
		Fn:          testProtocolCodecVLLMStreamError,
	})
}

// vllmStreamErrorWant is the client's last frame: its SSE event name, error
// type, code as raw JSON (a string, or absent for Messages), and the start of
// vLLM's message, plus the text delta that must reach the client first when
// content has started.
type vllmStreamErrorWant struct{ event, errorType, code, message, partial string }

var vllmStreamErrorCases = []struct {
	marker string
	want   map[string]vllmStreamErrorWant
}{
	{vllmStreamErrorMarker, map[string]vllmStreamErrorWant{
		"/v1/chat/completions": {"", "invalid_request_error", `"400"`, vllmGrammarErrorPrefix, ""},
		"/v1/messages":         {"error", "invalid_request_error", "", vllmGrammarErrorPrefix, ""},
		"/v1/responses":        {"error", "", `"400"`, vllmGrammarErrorPrefix, ""},
	}},
	{vllmMidstreamErrorMarker, map[string]vllmStreamErrorWant{
		"/v1/chat/completions": {"", "server_error", `"500"`, "Internal server error", `"content":"partial"`},
		"/v1/messages":         {"error", "api_error", "", "Internal server error", `"text":"partial"`},
		"/v1/responses":        {"error", "", `"500"`, "Internal server error", `"delta":"partial"`},
	}},
}

func testProtocolCodecVLLMStreamError(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	for _, tc := range vllmStreamErrorCases {
		for _, protocol := range protocolCodecE2EClients {
			result, err := sendProtocolMatrixRaw(ctx, session, protocol.path, protocol.request(chatBackendModel, tc.marker, true), true, nil)
			if err != nil {
				return fmt.Errorf("%s %s: %w", protocol.name, tc.marker, err)
			}
			if err := assertVLLMStreamErrorReply(result, tc.want[protocol.path]); err != nil {
				return fmt.Errorf("%s %s: %w", protocol.name, tc.marker, err)
			}
		}
	}
	return nil
}

// A streamed reply is SSE whose last frame is vLLM's error. After content has
// started, the partial output reaches the client first.
func assertVLLMStreamErrorReply(result protocolMatrixHTTPResult, want vllmStreamErrorWant) error {
	body := string(result.Body)
	if result.StatusCode != http.StatusOK || !strings.HasPrefix(result.Headers.Get("Content-Type"), "text/event-stream") {
		return fmt.Errorf("want HTTP 200 SSE, got HTTP %d %q: %s",
			result.StatusCode, result.Headers.Get("Content-Type"), truncateString(body, 900))
	}
	frames := strings.Split(strings.TrimSpace(body), "\n\n")
	event, data := "", ""
	for _, line := range strings.Split(frames[len(frames)-1], "\n") {
		if name, ok := strings.CutPrefix(line, "event: "); ok {
			event = name
		}
		if payload, ok := strings.CutPrefix(line, "data: "); ok {
			data = payload
		}
	}
	var frame struct {
		Code    json.RawMessage `json:"code"`
		Message string          `json:"message"`
		Error   struct {
			Type    string          `json:"type"`
			Code    json.RawMessage `json:"code"`
			Message string          `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal([]byte(data), &frame); err != nil {
		return fmt.Errorf("last frame is not JSON: %w: %s", err, truncateString(body, 900))
	}
	code, message := frame.Code, frame.Message
	if frame.Error.Message != "" {
		code, message = frame.Error.Code, frame.Error.Message
	}
	if event != want.event || frame.Error.Type != want.errorType || string(code) != want.code || !strings.HasPrefix(message, want.message) {
		return fmt.Errorf("last frame is not vLLM's error (want event %q, type %q, code %q, message %q): %s",
			want.event, want.errorType, want.code, want.message, truncateString(body, 900))
	}
	if want.partial != "" && !strings.Contains(strings.Join(frames[:len(frames)-1], "\n\n"), want.partial) {
		return fmt.Errorf("partial output %s did not reach the client before the error: %s", want.partial, truncateString(body, 900))
	}
	return nil
}
