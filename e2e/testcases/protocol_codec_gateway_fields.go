package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("protocol-codec-anthropic-gateway-fields", pkgtestcases.TestCase{
		Description: "Gateway fields the Messages contract does not declare need no setting: buffered and streamed replies succeed for Messages, Chat, and Responses clients",
		Tags:        []string{"protocol-codec", "anthropic", "response-api", "streaming"},
		Fn:          testAnthropicGatewayFields,
	})
}

// The provider simulator adds usage.total_cost and a top-level
// gateway_trace_id, the shape an enterprise gateway in front of Claude sends.
func testAnthropicGatewayFields(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	return runResponseDecorationMatrix(ctx, client, opts, "vllm-sr/auto", "anthropic.messages.v1",
		"__mock_gateway_fields__ __mock_protocol_matrix__", protocolCodecAnthropicReply,
		assertAnthropicGatewayFields)
}

func assertAnthropicGatewayFields(client protocolCodecE2EClient, streamed bool, result protocolMatrixHTTPResult) error {
	body := string(result.Body)
	if client.path == "/v1/messages" {
		// A same-format client receives the gateway's fields unchanged.
		for _, field := range []string{`"total_cost":`, `"gateway_trace_id":`} {
			if !strings.Contains(body, field) {
				return fmt.Errorf("native Messages reply lost %s: %s", field, truncateString(body, 600))
			}
		}
		return nil
	}
	for _, field := range []string{`"total_cost"`, `"gateway_trace_id"`} {
		if strings.Contains(body, field) {
			return fmt.Errorf("%s reply leaked the gateway's %s: %s", client.name, field, truncateString(body, 600))
		}
	}
	if streamed {
		return nil
	}
	warnings := result.Headers.Get("x-vsr-protocol-warnings")
	for _, field := range []string{"usage.total_cost", "gateway_trace_id"} {
		if !hasProtocolFieldDiagnostic(warnings, "dropped", field) {
			return fmt.Errorf("gateway field %s omitted without a warning: %q", field, warnings)
		}
	}
	if client.path == "/v1/chat/completions" {
		var reply struct {
			Usage struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}
		if err := json.Unmarshal(result.Body, &reply); err != nil {
			return err
		}
		if reply.Usage.PromptTokens != 6 || reply.Usage.CompletionTokens != 3 {
			return fmt.Errorf("chat usage = %d/%d, want the provider's 6/3", reply.Usage.PromptTokens, reply.Usage.CompletionTokens)
		}
	}
	return nil
}
