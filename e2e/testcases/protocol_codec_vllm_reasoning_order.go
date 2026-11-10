package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"strings"

	"github.com/google/uuid"
	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const vllmReasoningOrderMarker = "__mock_vllm_reasoning_order__"

func init() {
	pkgtestcases.Register("protocol-codec-vllm-reasoning-order", pkgtestcases.TestCase{
		Description: "vLLM's delta that ends the reasoning and starts the answer reaches Messages and Responses clients reasoning first and Chat clients unchanged",
		Tags:        []string{"protocol-codec", "vllm", "anthropic", "response-api", "reasoning", "streaming"},
		Fn:          testProtocolCodecVLLMReasoningOrder,
	})
}

// vllmReasoningOrderCase is one client's wants: parts in order, the last data
// frame, and the Messages block or Responses item types the reply opens, in
// order. The fixture's mixed delta is {"content":"\n\nHello","reasoning":".\n"}.
type vllmReasoningOrderCase struct {
	path   string
	stream bool
	order  []string
	last   string
	types  []string
}

var vllmReasoningOrderCases = []vllmReasoningOrderCase{
	{"/v1/messages", true, []string{`"thinking":".\n"`, `"text":"\n\nHello"`}, `"type":"message_stop"`, []string{"thinking", "text"}},
	{"/v1/responses", true, []string{`"delta":".\n"`, `"delta":"\n\nHello"`}, `"type":"response.completed"`, []string{"reasoning", "message"}},
	// A Chat client gets vLLM's own frames.
	{"/v1/chat/completions", true, []string{`"content":"\n\nHello","reasoning":".\n"`}, "[DONE]", nil},
	{"/v1/messages", false, nil, "", []string{"thinking", "text"}},
	{"/v1/responses", false, nil, "", []string{"reasoning", "message"}},
}

func testProtocolCodecVLLMReasoningOrder(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
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

	for _, tc := range vllmReasoningOrderCases {
		name := fmt.Sprintf("vLLM reasoning order %s (stream=%t)", tc.path, tc.stream)
		sessionID := fmt.Sprintf("vllm-reasoning-order-%t-%s", tc.stream, uuid.NewString())
		request := protocolCodecE2EClient{path: tc.path}.request(chatBackendModel, vllmReasoningOrderMarker, tc.stream)
		result, err := sendProtocolMatrixRaw(ctx, session, tc.path, request, tc.stream,
			map[string]string{"x-vsr-test-session-id": sessionID})
		if err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		if err := assertVLLMReasoningOrderReply(result, tc); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		if err := verifyProviderSimulatorRequest(ctx, provider, sessionID, "openai.chat.v1", vllmReasoningOrderMarker); err != nil {
			return fmt.Errorf("%s provider dispatch: %w", name, err)
		}
		if err := assertProviderStreamMode(ctx, provider, sessionID, tc.stream); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
	}
	return nil
}

func assertVLLMReasoningOrderReply(result protocolMatrixHTTPResult, tc vllmReasoningOrderCase) error {
	body := string(result.Body)
	if result.StatusCode != http.StatusOK || strings.Contains(body, "event: error") {
		return fmt.Errorf("want HTTP 200 with no error event, got HTTP %d: %s", result.StatusCode, truncateString(body, 900))
	}
	if err := validateOrderedStreamMarkers(body, tc.order); err != nil {
		return err
	}
	types, last, err := vllmReasoningOrderTypes(result.Body, tc.stream)
	if err != nil {
		return fmt.Errorf("%w: %s", err, truncateString(body, 900))
	}
	if !slices.Equal(types, tc.types) || !strings.Contains(last, tc.last) {
		return fmt.Errorf("opened %v and ended %q, want %v ending %q: %s",
			types, truncateString(last, 200), tc.types, tc.last, truncateString(body, 900))
	}
	return nil
}

// vllmReasoningOrderTypes returns the Messages content block or Responses
// output item types a reply opens, in order, and its last SSE data frame.
func vllmReasoningOrderTypes(body []byte, stream bool) ([]string, string, error) {
	type typed struct {
		Type string `json:"type"`
	}
	var types []string
	if !stream {
		var reply struct{ Content, Output []typed }
		if err := json.Unmarshal(body, &reply); err != nil {
			return nil, "", err
		}
		for _, part := range append(reply.Content, reply.Output...) {
			types = append(types, part.Type)
		}
		return types, "", nil
	}
	frames := protocolSSEDataFrames(body)
	if len(frames) == 0 {
		return nil, "", fmt.Errorf("reply has no SSE data frames")
	}
	for _, data := range frames {
		var frame struct {
			Type         string `json:"type"`
			ContentBlock typed  `json:"content_block"`
			Item         typed  `json:"item"`
		}
		if json.Unmarshal([]byte(data), &frame) != nil {
			continue
		}
		switch frame.Type {
		case "content_block_start":
			types = append(types, frame.ContentBlock.Type)
		case "response.output_item.added":
			types = append(types, frame.Item.Type)
		}
	}
	return types, frames[len(frames)-1], nil
}
