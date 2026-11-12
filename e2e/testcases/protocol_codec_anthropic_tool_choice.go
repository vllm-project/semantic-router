package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/google/uuid"
	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func runAnthropicNoneParallelControl(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	router, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer router.Close()
	provider, err := openProtocolCodecProviderSession(ctx, client, opts, "anthropic.messages.v1")
	if err != nil {
		return err
	}
	defer provider.Close()
	for _, source := range protocolCodecE2EClients {
		if source.path == "/v1/messages" {
			continue
		}
		for _, stream := range []bool{false, true} {
			for _, parallel := range []string{"omitted", "false", "true"} {
				body := source.request("vllm-sr/auto", protocolCodecAnthropicProbe, stream)
				delete(body, "store") // Keep conversation-state policy outside this regression.
				body["tool_choice"] = "none"
				tool := protocolLookupTool()
				if source.path == "/v1/chat/completions" {
					delete(tool, "type")
					body["tools"] = []any{map[string]any{"type": "function", "function": tool}}
				} else {
					body["tools"] = []any{tool}
				}
				if parallel != "omitted" {
					body["parallel_tool_calls"] = parallel == "true"
				}
				sessionID := "none-parallel-" + uuid.NewString()
				result, err := sendProtocolMatrixRaw(ctx, router, source.path, body, stream,
					map[string]string{"x-vsr-test-session-id": sessionID})
				if err != nil {
					return err
				}
				if result.StatusCode != http.StatusOK {
					return fmt.Errorf("none tool choice (%s stream=%t parallel=%s): HTTP %d: %s", source.name, stream, parallel, result.StatusCode, truncateString(string(result.Body), 500))
				}
				if stream {
					err = source.validateStream(result.Body, protocolCodecAnthropicReply)
				} else {
					err = source.validateBuffered(result.Body, protocolCodecAnthropicReply)
				}
				if err != nil {
					return err
				}
				raw, err := lastProviderSimulatorRequest(ctx, provider, sessionID)
				if err != nil {
					return err
				}
				// A permissive mock's HTTP 200 cannot prove Anthropic wire validity.
				if err := assertAnthropicNoneToolChoice(raw); err != nil {
					return fmt.Errorf("none tool choice (%s stream=%t parallel=%s): %w", source.name, stream, parallel, err)
				}
			}
		}
	}
	return nil
}

func assertAnthropicNoneToolChoice(raw []byte) error {
	var observation struct {
		Body map[string]json.RawMessage `json:"body"`
	}
	if err := json.Unmarshal(raw, &observation); err != nil {
		return err
	}
	var choice map[string]any
	if err := json.Unmarshal(observation.Body["tool_choice"], &choice); err != nil {
		return err
	}
	if len(choice) != 1 || choice["type"] != "none" {
		return fmt.Errorf("provider tool_choice must contain only type=none, got %s", observation.Body["tool_choice"])
	}
	if _, found := observation.Body["parallel_tool_calls"]; found {
		return fmt.Errorf("provider request leaked OpenAI parallel_tool_calls into Anthropic request")
	}
	return nil
}
