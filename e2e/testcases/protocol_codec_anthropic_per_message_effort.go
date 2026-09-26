package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/google/uuid"
	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("protocol-codec-anthropic-per-message-effort", pkgtestcases.TestCase{
		Description: "Claude per-message effort is explicitly dropped for Chat and Responses backends",
		Tags:        []string{"protocol-codec", "anthropic", "response-api", "agents"},
		Fn:          testProtocolCodecAnthropicPerMessageEffort,
	})
	pkgtestcases.Register("protocol-codec-anthropic-per-message-effort-backend", pkgtestcases.TestCase{
		Description: "Claude per-message effort survives Anthropic Messages backend dispatch",
		Tags:        []string{"protocol-codec", "anthropic", "agents"},
		Fn:          testProtocolCodecAnthropicPerMessageEffortBackend,
	})
}

type perMessageEffortBackend struct {
	model, format, reply string
}

func testProtocolCodecAnthropicPerMessageEffort(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	return runProtocolCodecAnthropicPerMessageEffort(ctx, client, opts, []perMessageEffortBackend{
		{model: chatBackendModel, format: "openai.chat.v1", reply: protocolCodecChatReply},
		{model: nativeResponsesBackendModel, format: "openai.responses.v1", reply: protocolCodecResponsesReply},
	})
}

func testProtocolCodecAnthropicPerMessageEffortBackend(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	return runProtocolCodecAnthropicPerMessageEffort(ctx, client, opts, []perMessageEffortBackend{
		{model: "MoM", format: "anthropic.messages.v1", reply: protocolCodecAnthropicReply},
	})
}

func runProtocolCodecAnthropicPerMessageEffort(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions, backends []perMessageEffortBackend) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	for _, check := range backends {
		provider, providerErr := openProtocolCodecProviderSession(ctx, client, opts, check.format)
		if providerErr != nil {
			return providerErr
		}
		sessionID := "anthropic-per-message-effort-" + uuid.NewString()
		result, requestErr := sendProtocolMatrixRaw(ctx, session, "/v1/messages", map[string]any{
			"model": check.model, "max_tokens": 64,
			"messages": []map[string]any{
				{"role": "user", "content": "First turn"},
				{"role": "system", "content": []map[string]string{{"type": "text", "text": "Current date is 2026-01-01."}}, "output_config": map[string]string{"effort": "medium"}},
				{"role": "system", "content": []any{}, "output_config": map[string]string{"effort": "low"}},
				{"role": "user", "content": "Per message effort probe"},
			},
		}, false, map[string]string{
			"x-vsr-test-session-id": sessionID,
			"anthropic-beta":        "per-turn-control-2026-07-01",
		})
		if requestErr != nil {
			provider.Close()
			return fmt.Errorf("%s request: %w", check.format, requestErr)
		}
		if result.StatusCode != http.StatusOK {
			provider.Close()
			return fmt.Errorf("%s returned HTTP %d: %s", check.format, result.StatusCode,
				truncateString(string(result.Body), 500))
		}
		if responseErr := assertAnthropicBody(result.Body, check.reply); responseErr != nil {
			provider.Close()
			return fmt.Errorf("%s client response: %w", check.format, responseErr)
		}
		upstream, observationErr := lastProviderSimulatorRequest(ctx, provider, sessionID)
		provider.Close()
		if observationErr != nil {
			return fmt.Errorf("%s provider observation: %w", check.format, observationErr)
		}
		if inspectErr := assertPerMessageEffortProjection(upstream, check.format); inspectErr != nil {
			return fmt.Errorf("%s: %w", check.format, inspectErr)
		}
		warned := hasProtocolFieldDiagnostic(result.Headers.Get("x-vsr-protocol-warnings"), "dropped", "messages[].output_config.effort")
		if warned != (check.format != "anthropic.messages.v1") {
			return fmt.Errorf("%s per-message effort warning = %t", check.format, warned)
		}
	}
	return nil
}

func assertPerMessageEffortProjection(raw []byte, format string) error {
	var observation struct {
		Body    map[string]json.RawMessage `json:"body"`
		Headers map[string]string          `json:"headers"`
	}
	if err := json.Unmarshal(raw, &observation); err != nil {
		return err
	}
	key := "messages"
	if format == "openai.responses.v1" {
		key = "input"
	}
	var messages []map[string]json.RawMessage
	if err := json.Unmarshal(observation.Body[key], &messages); err != nil {
		return fmt.Errorf("decode provider %s: %w", key, err)
	}
	if format == "anthropic.messages.v1" {
		if observation.Headers["anthropic-beta"] != "per-turn-control-2026-07-01" {
			return fmt.Errorf("messages backend lost the per-turn-control beta header")
		}
		if len(messages) != 4 {
			return fmt.Errorf("messages backend lost the effort-only system turn: %s", truncateString(string(raw), 500))
		}
		for index, effort := range map[int]string{1: "medium", 2: "low"} {
			var config struct {
				Effort string `json:"effort"`
			}
			if err := json.Unmarshal(messages[index]["output_config"], &config); err != nil || config.Effort != effort {
				return fmt.Errorf("messages backend lost effort %q at %d: %s", effort, index, truncateString(string(raw), 500))
			}
		}
		if string(messages[2]["content"]) != "[]" {
			return fmt.Errorf("messages backend changed effort-only content: %s", truncateString(string(raw), 500))
		}
		return nil
	}
	if len(messages) != 3 {
		return fmt.Errorf("%s forwarded the empty effort-only system turn: %s", format, truncateString(string(raw), 500))
	}
	if !bytes.Contains(messages[1]["content"], []byte("Current date is 2026-01-01.")) {
		return fmt.Errorf("%s lost the nonempty system turn: %s", format, truncateString(string(raw), 500))
	}
	for _, message := range messages {
		if _, leaked := message["output_config"]; leaked {
			return fmt.Errorf("%s forwarded per-message output_config: %s", format, truncateString(string(raw), 500))
		}
	}
	return nil
}
