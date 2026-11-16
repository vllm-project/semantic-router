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

// provider-mocker answers this marker with a reasoning item that holds only
// an encrypted blob. Its endpoint is not OpenAI-hosted, so the Router gives
// the blob to the Responses client but never sends one to the mocker.
const (
	encryptedReasoningMarker = "__mock_responses_encrypted_reasoning__"
	encryptedReasoningBlob   = "mock-encrypted-reasoning"
)

func init() {
	pkgtestcases.Register("protocol-codec-responses-encrypted-reasoning", pkgtestcases.TestCase{
		Description: "A Responses client gets a reasoning item's encrypted_content back, buffered and streamed, and a replayed blob is dropped and reported for a backend that is not OpenAI-hosted",
		Tags:        []string{"protocol-codec", "response-api", "agents"},
		Fn:          testProtocolCodecResponsesEncryptedReasoning,
	})
}

func testProtocolCodecResponsesEncryptedReasoning(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, sessionErr := fixtures.OpenServiceSession(ctx, client, opts)
	if sessionErr != nil {
		return sessionErr
	}
	defer session.Close()
	provider, providerErr := openProtocolCodecProviderSession(ctx, client, opts, "openai.responses.v1")
	if providerErr != nil {
		return providerErr
	}
	defer provider.Close()

	for _, stream := range []bool{false, true} {
		result, err := sendProtocolMatrixRaw(ctx, session, "/v1/responses", map[string]any{
			"model": nativeResponsesBackendModel, "input": encryptedReasoningMarker, "store": false, "stream": stream,
			"include": []string{"reasoning.encrypted_content"},
		}, stream, nil)
		if err != nil {
			return fmt.Errorf("stream=%t request: %w", stream, err)
		}
		if result.StatusCode != http.StatusOK {
			return fmt.Errorf("stream=%t returned HTTP %d: %s", stream, result.StatusCode, truncateString(string(result.Body), 500))
		}
		if err := assertReplyCarriesEncryptedReasoning(result.Body, stream); err != nil {
			return fmt.Errorf("stream=%t: %w", stream, err)
		}
	}

	const replay = "Encrypted reasoning replay probe"
	const sessionID = "responses-encrypted-reasoning-replay"
	result, err := sendProtocolMatrixRaw(ctx, session, "/v1/responses", map[string]any{
		"model": nativeResponsesBackendModel, "store": false,
		"include": []string{"reasoning.encrypted_content"},
		"input": []map[string]any{
			{"type": "reasoning", "id": "rs_replayed", "summary": []any{}, "encrypted_content": encryptedReasoningBlob},
			{"type": "message", "role": "user", "content": []map[string]string{{"type": "input_text", "text": replay}}},
		},
	}, false, map[string]string{"x-vsr-test-session-id": sessionID})
	if err != nil {
		return fmt.Errorf("replay request: %w", err)
	}
	if result.StatusCode != http.StatusOK {
		return fmt.Errorf("replay returned HTTP %d: %s", result.StatusCode, truncateString(string(result.Body), 500))
	}
	if warnings := result.Headers.Get(warningsHeader); !hasProtocolFieldDiagnostic(warnings, "dropped", "input[].encrypted_content") {
		return fmt.Errorf("replay did not report the dropped blob: %q", warnings)
	}
	raw, err := lastProviderSimulatorRequest(ctx, provider, sessionID)
	if err != nil {
		return fmt.Errorf("replay provider observation: %w", err)
	}
	if !strings.Contains(string(raw), replay) {
		return fmt.Errorf("replay provider observation does not match request: %s", truncateString(string(raw), 500))
	}
	if strings.Contains(string(raw), encryptedReasoningBlob) || strings.Contains(string(raw), "reasoning.encrypted_content") {
		return fmt.Errorf("the blob or its include reached a backend that is not OpenAI-hosted: %s", truncateString(string(raw), 500))
	}
	return nil
}

type encryptedReasoningItem struct {
	Type             string `json:"type"`
	EncryptedContent string `json:"encrypted_content"`
}

// assertReplyCarriesEncryptedReasoning reads the reply's reasoning items: the
// buffered output, or a stream's item-done event and its terminal output.
func assertReplyCarriesEncryptedReasoning(body []byte, stream bool) error {
	if !stream {
		var response struct {
			Output []encryptedReasoningItem `json:"output"`
		}
		if err := json.Unmarshal(body, &response); err != nil {
			return fmt.Errorf("decode reply: %w", err)
		}
		return firstReasoningCarriesBlob(response.Output, "output")
	}
	var itemDone, terminal []encryptedReasoningItem
	for _, line := range strings.Split(string(body), "\n") {
		data, found := strings.CutPrefix(line, "data: ")
		if !found {
			continue
		}
		var event struct {
			Type     string                 `json:"type"`
			Item     encryptedReasoningItem `json:"item"`
			Response struct {
				Output []encryptedReasoningItem `json:"output"`
			} `json:"response"`
		}
		if json.Unmarshal([]byte(data), &event) != nil {
			continue
		}
		switch event.Type {
		case "response.output_item.done":
			itemDone = append(itemDone, event.Item)
		case "response.completed":
			terminal = event.Response.Output
		}
	}
	if err := firstReasoningCarriesBlob(itemDone, "response.output_item.done"); err != nil {
		return err
	}
	return firstReasoningCarriesBlob(terminal, "response.completed")
}

func firstReasoningCarriesBlob(items []encryptedReasoningItem, where string) error {
	for _, item := range items {
		if item.Type != "reasoning" {
			continue
		}
		if item.EncryptedContent != encryptedReasoningBlob {
			return fmt.Errorf("%s reasoning item has encrypted_content %q, want %q", where, item.EncryptedContent, encryptedReasoningBlob)
		}
		return nil
	}
	return fmt.Errorf("%s holds no reasoning item: %+v", where, items)
}
