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

func init() {
	pkgtestcases.Register("protocol-codec-chat-stream-multiple-choices", pkgtestcases.TestCase{
		Description: "A streamed Chat request for several choices gets a typed 400 before any provider is called, while one streamed choice and several buffered choices still reach the Chat provider (issue #5574)",
		Tags:        []string{"protocol-codec", "streaming", "failure"},
		Fn:          testProtocolCodecChatStreamMultipleChoices,
	})
}

// The Chat stream decoder reads choice index 0 only. Before #5574 the Router
// forwarded n>1 with stream:true, answered 200, and reported the decoder's
// error only at the provider's end of stream.
func testProtocolCodecChatStreamMultipleChoices(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
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

	// The controls run first: they prove the provider observation path is
	// live, so the rejected request's missing observation means no dispatch.
	for _, control := range []struct {
		name   string
		n      int
		stream bool
	}{{"one streamed choice", 1, true}, {"several buffered choices", 2, false}} {
		if controlErr := assertChatCandidatesDispatched(ctx, session, provider, control.n, control.stream); controlErr != nil {
			return fmt.Errorf("%s: %w", control.name, controlErr)
		}
	}

	sessionID := "chat-stream-choices-" + uuid.NewString()
	result, err := sendProtocolMatrixRaw(ctx, session, "/v1/chat/completions",
		chatCandidatesRequest(2, true), true, map[string]string{"x-vsr-test-session-id": sessionID})
	if err != nil {
		return fmt.Errorf("streamed n=2 request: %w", err)
	}
	if result.StatusCode != http.StatusBadRequest {
		return fmt.Errorf("streamed n=2 returned HTTP %d, want 400: %s", result.StatusCode,
			truncateString(string(result.Body), 500))
	}
	var response struct {
		Error struct {
			Type string `json:"type"`
			Code string `json:"code"`
		} `json:"error"`
	}
	if decodeErr := json.Unmarshal(result.Body, &response); decodeErr != nil {
		return fmt.Errorf("decode streamed n=2 error: %w: %s", decodeErr, truncateString(string(result.Body), 500))
	}
	if response.Error.Type != "invalid_request_error" || response.Error.Code != "stream_multiple_choices" {
		return fmt.Errorf("streamed n=2 returned the wrong error: %s", truncateString(string(result.Body), 500))
	}
	dispatched, model, err := lookupShortCircuitDispatch(ctx, provider, sessionID)
	if err != nil {
		return fmt.Errorf("streamed n=2 provider observation: %w", err)
	}
	if dispatched {
		return fmt.Errorf("streamed n=2 request reached provider model %q", model)
	}
	return nil
}

func chatCandidatesRequest(n int, stream bool) map[string]any {
	return map[string]any{
		"model": chatBackendModel, "n": n, "stream": stream, "max_tokens": 16,
		"messages": []map[string]string{{"role": "user", "content": "Chat candidate count probe"}},
	}
}

// assertChatCandidatesDispatched sends a Chat request for n choices and
// requires a successful reply and a provider request that kept n and stream.
func assertChatCandidatesDispatched(
	ctx context.Context,
	session, provider *fixtures.ServiceSession,
	n int,
	stream bool,
) error {
	sessionID := "chat-candidates-control-" + uuid.NewString()
	result, err := sendProtocolMatrixRaw(ctx, session, "/v1/chat/completions",
		chatCandidatesRequest(n, stream), stream, map[string]string{"x-vsr-test-session-id": sessionID})
	if err != nil {
		return err
	}
	if result.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d, want 200: %s", result.StatusCode, truncateString(string(result.Body), 500))
	}
	if stream && !strings.Contains(string(result.Body), "data: [DONE]") {
		return fmt.Errorf("stream did not end with [DONE]: %s", truncateString(string(result.Body), 900))
	}
	observed, err := lastProviderSimulatorRequest(ctx, provider, sessionID)
	if err != nil {
		return fmt.Errorf("provider observation: %w", err)
	}
	var receipt struct {
		Body struct {
			N      int  `json:"n"`
			Stream bool `json:"stream"`
		} `json:"body"`
	}
	if decodeErr := json.Unmarshal(observed, &receipt); decodeErr != nil {
		return fmt.Errorf("decode provider observation: %w", decodeErr)
	}
	if receipt.Body.N != n || receipt.Body.Stream != stream {
		return fmt.Errorf("provider received n=%d stream=%t, want n=%d stream=%t: %s",
			receipt.Body.N, receipt.Body.Stream, n, stream, truncateString(string(observed), 500))
	}
	return nil
}
