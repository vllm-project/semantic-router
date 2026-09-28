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
	pkgtestcases.Register("protocol-codec-responses-input-compat", pkgtestcases.TestCase{
		Description: "Chat and Copilot tool turns reach Responses without invented IDs, echoed status, or zero-penalty rejection",
		Tags:        []string{"protocol-codec", "response-api", "agents", "tools"},
		Fn:          testProtocolCodecResponsesInputCompat,
	})
}

func testProtocolCodecResponsesInputCompat(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	provider, err := openProtocolCodecProviderSession(ctx, client, opts, "openai.responses.v1")
	if err != nil {
		return err
	}
	defer provider.Close()

	for _, check := range []struct {
		name, path string
		body       map[string]any
		wantCall   bool
	}{
		{
			name: "chat-zero-penalties", path: "/v1/chat/completions",
			body: map[string]any{
				"model": nativeResponsesBackendModel, "frequency_penalty": 0, "presence_penalty": 0,
				"messages": []map[string]string{
					{"role": "system", "content": "Responses input ID probe"},
					{"role": "user", "content": "Replies should work without fabricated IDs"},
				},
			},
		},
		{
			name: "copilot-completed-tool-turn", path: "/v1/responses", wantCall: true,
			body: map[string]any{
				"model": nativeResponsesBackendModel, "store": false,
				"input": []map[string]any{
					{"type": "function_call", "id": "fc_1", "name": "bash", "arguments": `{}`, "call_id": "call_1", "status": "completed"},
					{"type": "function_call_output", "call_id": "call_1", "output": "ok", "status": "completed"},
					{"type": "message", "role": "user", "content": []map[string]string{{"type": "input_text", "text": "Copilot completed tool turn probe"}}, "status": "completed"},
				},
			},
		},
	} {
		sessionID := "responses-input-compat-" + uuid.NewString()
		result, requestErr := sendProtocolMatrixRaw(ctx, session, check.path, check.body, false,
			map[string]string{"x-vsr-test-session-id": sessionID})
		if requestErr != nil {
			return fmt.Errorf("%s request: %w", check.name, requestErr)
		}
		if result.StatusCode != http.StatusOK {
			return fmt.Errorf("%s returned HTTP %d: %s", check.name, result.StatusCode, truncateString(string(result.Body), 500))
		}
		raw, observationErr := lastProviderSimulatorRequest(ctx, provider, sessionID)
		if observationErr != nil {
			return fmt.Errorf("%s provider observation: %w", check.name, observationErr)
		}
		var observation struct {
			Body struct {
				Input []map[string]json.RawMessage `json:"input"`
			} `json:"body"`
		}
		if decodeErr := json.Unmarshal(raw, &observation); decodeErr != nil {
			return fmt.Errorf("%s provider observation decode: %w", check.name, decodeErr)
		}
		wantItems := 2
		if check.wantCall {
			wantItems = 3
		}
		if len(observation.Body.Input) != wantItems {
			return fmt.Errorf("%s provider received %d items, want %d: %s", check.name,
				len(observation.Body.Input), wantItems, truncateString(string(raw), 500))
		}
		for index, item := range observation.Body.Input {
			if _, found := item["status"]; found {
				return fmt.Errorf("%s input[%d] forwarded historical status: %s", check.name, index, truncateString(string(raw), 500))
			}
			if _, found := item["id"]; found && (!check.wantCall || index != 0) {
				return fmt.Errorf("%s input[%d] invented an item ID: %s", check.name, index, truncateString(string(raw), 500))
			}
		}
		if check.wantCall && (string(observation.Body.Input[0]["id"]) != `"fc_1"` ||
			string(observation.Body.Input[0]["call_id"]) != `"call_1"` ||
			string(observation.Body.Input[1]["call_id"]) != `"call_1"`) {
			return fmt.Errorf("%s lost the echoed tool call: %s", check.name, truncateString(string(raw), 500))
		}
	}

	anthropicID := "responses-anthropic-tool-result-" + uuid.NewString()
	anthropic, requestErr := sendProtocolMatrixRaw(ctx, session, "/v1/messages", map[string]any{
		"model": nativeResponsesBackendModel, "max_tokens": 64,
		"tools": []map[string]any{{"name": "bash", "input_schema": map[string]any{"type": "object", "properties": map[string]any{}}}},
		"messages": []map[string]any{
			{"role": "user", "content": "Call bash"},
			{"role": "assistant", "content": []map[string]any{{"type": "tool_use", "id": "call_2", "name": "bash", "input": map[string]any{}}}},
			{"role": "user", "content": []map[string]any{{"type": "tool_result", "tool_use_id": "call_2", "content": "ok"}}},
			{"role": "user", "content": "Summarize"},
		},
	}, false, map[string]string{"x-vsr-test-session-id": anthropicID})
	if requestErr != nil {
		return fmt.Errorf("anthropic tool-result request: %w", requestErr)
	}
	if anthropic.StatusCode != http.StatusOK {
		return fmt.Errorf("anthropic tool-result returned HTTP %d: %s", anthropic.StatusCode,
			truncateString(string(anthropic.Body), 500))
	}
	raw, observationErr := lastProviderSimulatorRequest(ctx, provider, anthropicID)
	if observationErr != nil {
		return fmt.Errorf("anthropic tool-result provider observation: %w", observationErr)
	}
	var observation struct {
		Body struct {
			Input []map[string]json.RawMessage `json:"input"`
		} `json:"body"`
	}
	if decodeErr := json.Unmarshal(raw, &observation); decodeErr != nil {
		return fmt.Errorf("anthropic tool-result provider observation decode: %w", decodeErr)
	}
	if len(observation.Body.Input) != 4 {
		return fmt.Errorf("anthropic tool-result produced %d input items, want 4: %s",
			len(observation.Body.Input), truncateString(string(raw), 500))
	}
	for index, item := range observation.Body.Input {
		if _, found := item["id"]; found {
			return fmt.Errorf("anthropic tool-result input[%d] invented an item ID: %s", index, truncateString(string(raw), 500))
		}
	}
	if string(observation.Body.Input[1]["call_id"]) != `"call_2"` ||
		string(observation.Body.Input[2]["call_id"]) != `"call_2"` {
		return fmt.Errorf("anthropic tool-result lost its call ID: %s", truncateString(string(raw), 500))
	}

	blockedID := "responses-nonzero-penalty-" + uuid.NewString()
	blocked, requestErr := sendProtocolMatrixRaw(ctx, session, "/v1/chat/completions", map[string]any{
		"model": nativeResponsesBackendModel, "frequency_penalty": 0.25,
		"messages": []map[string]string{{"role": "user", "content": "Nonzero penalty must still fail"}},
	}, false, map[string]string{"x-vsr-test-session-id": blockedID})
	if requestErr != nil {
		return fmt.Errorf("nonzero penalty request: %w", requestErr)
	}
	if blocked.StatusCode != http.StatusBadRequest || !strings.Contains(string(blocked.Body), "unsupported_capability") {
		return fmt.Errorf("nonzero penalty returned HTTP %d, want typed 400: %s", blocked.StatusCode,
			truncateString(string(blocked.Body), 500))
	}
	dispatched, model, err := lookupShortCircuitDispatch(ctx, provider, blockedID)
	if err != nil {
		return fmt.Errorf("nonzero penalty provider observation: %w", err)
	}
	if dispatched {
		return fmt.Errorf("nonzero penalty reached provider model %q", model)
	}
	return nil
}
