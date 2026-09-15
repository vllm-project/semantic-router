package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const pluginMutationSystemPrompt = "E2E plugin contract: system prompt applied."

func init() {
	pkgtestcases.Register("plugin-request-mutations", pkgtestcases.TestCase{
		Description: "Verify system_prompt, request_params, and header_mutation effects at the provider boundary",
		Tags:        []string{"plugin", "request-mutation", "provider-boundary"},
		Fn:          testPluginRequestMutations,
	})
}

func testPluginRequestMutations(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	backendOpts := opts
	backendOpts.ServiceConfig = pkgtestcases.ServiceConfig{
		Namespace:   "default",
		Name:        "vllm-llama3-8b-instruct",
		ServicePort: "8000",
	}
	backendSession, err := fixtures.OpenServiceSession(ctx, client, backendOpts)
	if err != nil {
		return err
	}
	defer backendSession.Close()

	sessionID := fmt.Sprintf("plugin-request-mutations-%d", time.Now().UnixNano())
	responseBody, err := sendProtocolMatrixRequestWithHeaders(
		ctx,
		session,
		"/v1/chat/completions",
		map[string]any{
			"model": "MoM",
			"messages": []any{map[string]any{
				"role":    "user",
				"content": "__PLUGIN_REQUEST_MUTATIONS__ preserve this user message",
			}},
			"max_tokens":        512,
			"n":                 4,
			"frequency_penalty": 0.75,
			"presence_penalty":  0.5,
		},
		false,
		map[string]string{
			"x-vsr-e2e-deleted":     "remove-me",
			"x-vsr-e2e-updated":     "client-value",
			"x-vsr-test-session-id": sessionID,
		},
	)
	if err != nil {
		return err
	}
	if validationErr := validatePluginMutationResponse(responseBody); validationErr != nil {
		return validationErr
	}

	observedBody, err := lastProviderSimulatorRequest(ctx, backendSession, sessionID)
	if err != nil {
		return err
	}
	if validationErr := validatePluginMutationProviderRequest(observedBody); validationErr != nil {
		return validationErr
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"plugins_verified":           []string{"header_mutation", "request_params", "system_prompt"},
			"provider_boundary_verified": true,
			"protocol_preserved":         true,
		})
	}
	return nil
}

func validatePluginMutationResponse(body []byte) error {
	var response struct {
		Object  string `json:"object"`
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return fmt.Errorf("decode chat-completions response: %w", err)
	}
	if response.Object != "chat.completion" || len(response.Choices) != 1 ||
		response.Choices[0].Message.Content == "" {
		return fmt.Errorf("request mutations changed the Chat Completions response contract: %s", truncateString(string(body), 600))
	}
	return nil
}

func validatePluginMutationProviderRequest(observed []byte) error {
	var request struct {
		Body    map[string]any    `json:"body"`
		Headers map[string]string `json:"headers"`
	}
	if err := json.Unmarshal(observed, &request); err != nil {
		return fmt.Errorf("decode provider-bound request: %w", err)
	}

	if err := validatePluginMutationMessages(request.Body); err != nil {
		return err
	}
	if got := request.Body["max_completion_tokens"]; got != float64(64) {
		return fmt.Errorf("request_params did not cap max_completion_tokens: got %#v, want 64", got)
	}
	if _, exists := request.Body["max_tokens"]; exists {
		return fmt.Errorf("provider request retained legacy max_tokens after mutation: %s", truncateString(string(observed), 800))
	}
	if got := request.Body["n"]; got != float64(1) {
		return fmt.Errorf("request_params did not cap n: got %#v, want 1", got)
	}
	for _, field := range []string{"frequency_penalty", "presence_penalty"} {
		if _, exists := request.Body[field]; exists {
			return fmt.Errorf("request_params did not remove %q from provider request: %s", field, truncateString(string(observed), 800))
		}
	}
	if request.Headers["x-vsr-e2e-added"] != "added-by-router" {
		return fmt.Errorf("header_mutation add was not observed: %#v", request.Headers)
	}
	if request.Headers["x-vsr-e2e-updated"] != "updated-by-router" {
		return fmt.Errorf("header_mutation update was not observed: %#v", request.Headers)
	}
	if _, exists := request.Headers["x-vsr-e2e-deleted"]; exists {
		return fmt.Errorf("header_mutation delete was not observed: %#v", request.Headers)
	}
	return nil
}

func validatePluginMutationMessages(body map[string]any) error {
	messages, ok := body["messages"].([]any)
	if !ok || len(messages) != 2 {
		return fmt.Errorf("system_prompt did not produce the expected two-message request: %#v", body["messages"])
	}
	system, systemOK := messages[0].(map[string]any)
	user, userOK := messages[1].(map[string]any)
	if !systemOK || system["role"] != "system" || system["content"] != pluginMutationSystemPrompt {
		return fmt.Errorf("system_prompt mutation mismatch: %#v", messages[0])
	}
	if !userOK || user["role"] != "user" ||
		user["content"] != "__PLUGIN_REQUEST_MUTATIONS__ preserve this user message" {
		return fmt.Errorf("request mutation changed the user message: %#v", messages[1])
	}
	return nil
}
