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

const vllmStopSequenceMarker = "__mock_vllm_stop_sequence__"

func init() {
	pkgtestcases.Register("protocol-codec-vllm-stop-sequence", pkgtestcases.TestCase{
		Description: "vLLM's matched stop string in choices[].stop_reason reaches Anthropic clients as stop_sequence",
		Tags:        []string{"protocol-codec", "vllm", "anthropic", "stop-reason", "streaming"},
		Fn:          testProtocolCodecVLLMStopSequence,
	})
}

func testProtocolCodecVLLMStopSequence(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	provider, err := openProtocolCodecProviderSession(ctx, client, opts, "openai.chat.v1")
	if err != nil {
		return err
	}
	defer provider.Close()

	// The fixture answers only when the Chat request carries a stop list, so a
	// pass also shows the Router forwarded stop_sequences as Chat stop.
	for _, stream := range []bool{false, true} {
		sessionID := fmt.Sprintf("vllm-stop-sequence-%t-%s", stream, uuid.NewString())
		result, err := sendProtocolMatrixRaw(ctx, session, "/v1/messages", map[string]any{
			"model": chatBackendModel, "max_tokens": 64, "stream": stream,
			"stop_sequences": []string{"CHARLIE"},
			"messages":       []map[string]string{{"role": "user", "content": vllmStopSequenceMarker}},
		}, stream, map[string]string{"x-vsr-test-session-id": sessionID})
		if err != nil {
			return fmt.Errorf("vLLM stop sequence Messages request (stream=%t): %w", stream, err)
		}
		if result.StatusCode != http.StatusOK {
			return fmt.Errorf("vLLM stop sequence Messages (stream=%t) returned HTTP %d: %s",
				stream, result.StatusCode, truncateString(string(result.Body), 600))
		}
		if replyErr := assertVLLMStopSequenceReply(result, stream); replyErr != nil {
			return replyErr
		}
		if verificationErr := verifyProviderSimulatorRequest(ctx, provider, sessionID, "openai.chat.v1", vllmStopSequenceMarker); verificationErr != nil {
			return fmt.Errorf("vLLM stop sequence provider dispatch (stream=%t): %w", stream, verificationErr)
		}
		if modeErr := assertProviderStreamMode(ctx, provider, sessionID, stream); modeErr != nil {
			return modeErr
		}
	}
	return nil
}

// A streamed reply must be real SSE with the match on message_delta, not a
// buffered body that happens to contain it.
func assertVLLMStopSequenceReply(result protocolMatrixHTTPResult, stream bool) error {
	var terminal struct {
		Type         string  `json:"type"`
		StopReason   string  `json:"stop_reason"`
		StopSequence *string `json:"stop_sequence"`
		Delta        *struct {
			StopReason   string  `json:"stop_reason"`
			StopSequence *string `json:"stop_sequence"`
		} `json:"delta"`
	}
	body := string(result.Body)
	payload := body
	if stream {
		contentType := result.Headers.Get("Content-Type")
		delta := strings.Index(body, "event: message_delta\ndata: ")
		stop := strings.Index(body, "event: message_stop")
		if !strings.HasPrefix(contentType, "text/event-stream") || delta < 0 || stop < delta {
			return fmt.Errorf("streamed Messages reply is not SSE with message_delta then message_stop (content-type %q): %s",
				contentType, truncateString(body, 900))
		}
		payload, _, _ = strings.Cut(body[delta+len("event: message_delta\ndata: "):], "\n")
	}
	if err := json.Unmarshal([]byte(payload), &terminal); err != nil {
		return fmt.Errorf("decode Messages terminal (stream=%t): %w: %s", stream, err, truncateString(payload, 600))
	}
	if terminal.Delta != nil {
		terminal.StopReason, terminal.StopSequence = terminal.Delta.StopReason, terminal.Delta.StopSequence
	}
	if terminal.StopReason != "stop_sequence" || terminal.StopSequence == nil || *terminal.StopSequence != "CHARLIE" {
		return fmt.Errorf("vLLM matched stop sequence did not reach the Messages client (stream=%t): %s",
			stream, truncateString(payload, 900))
	}
	return nil
}

func assertProviderStreamMode(ctx context.Context, provider *fixtures.ServiceSession, sessionID string, stream bool) error {
	recorded, err := lastProviderSimulatorRequest(ctx, provider, sessionID)
	if err != nil {
		return err
	}
	var debug struct {
		Body struct {
			Stream bool `json:"stream"`
		} `json:"body"`
	}
	if err := json.Unmarshal(recorded, &debug); err != nil {
		return fmt.Errorf("decode provider simulator request: %w", err)
	}
	if debug.Body.Stream != stream {
		return fmt.Errorf("provider request stream=%t, want %t", debug.Body.Stream, stream)
	}
	return nil
}
