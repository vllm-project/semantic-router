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

const ollamaCodecModel = "mock/ollama-chat"

func init() {
	pkgtestcases.Register("protocol-codec-ollama-output-limit", pkgtestcases.TestCase{
		Description: "Chat, Messages and Responses output limits reach Ollama as max_tokens without changing ordinary Chat providers",
		Tags:        []string{"protocol-codec", "ollama", "response-api", "agents"},
		Fn:          testProtocolCodecOllamaOutputLimit,
	})
	pkgtestcases.Register("protocol-codec-auto-unsupported-capability", pkgtestcases.TestCase{
		Description: "Auto routing reports a typed 400 when disabled thinking has no Chat backend off control, without dispatching a provider",
		Tags:        []string{"protocol-codec", "anthropic", "response-api", "agents", "failure"},
		Fn:          testProtocolCodecAutoUnsupportedCapability,
	})
}

func testProtocolCodecOllamaOutputLimit(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
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

	cases := []struct {
		name, path string
		body       map[string]any
		wantLimit  string
		assert     func([]byte, string) error
	}{
		{
			name: "chat", path: "/v1/chat/completions",
			body: map[string]any{
				"model": ollamaCodecModel, "max_tokens": 8,
				"messages": []map[string]string{{"role": "user", "content": "Ollama Chat limit probe"}},
			},
			wantLimit: "8", assert: assertChatCompletionBody,
		},
		{
			name: "messages", path: "/v1/messages",
			body: map[string]any{
				"model": ollamaCodecModel, "max_tokens": 8,
				"messages": []map[string]string{{"role": "user", "content": "Ollama Messages limit probe"}},
			},
			wantLimit: "8", assert: assertAnthropicBody,
		},
		{
			name: "responses", path: "/v1/responses",
			body: map[string]any{
				"model": ollamaCodecModel, "max_output_tokens": 16, "store": false,
				"input": "Ollama Responses limit probe",
			},
			wantLimit: "16", assert: assertResponsesBody,
		},
	}
	for _, check := range cases {
		sessionID := "ollama-limit-" + uuid.NewString()
		result, requestErr := sendProtocolMatrixRaw(ctx, session, check.path, check.body, false,
			map[string]string{"x-vsr-test-session-id": sessionID})
		if requestErr != nil {
			return fmt.Errorf("%s output limit request: %w", check.name, requestErr)
		}
		if result.StatusCode != http.StatusOK {
			return fmt.Errorf("%s output limit returned HTTP %d: %s", check.name, result.StatusCode,
				truncateString(string(result.Body), 500))
		}
		if assertionErr := check.assert(result.Body, `"protocol":"chat_completions"`); assertionErr != nil {
			return fmt.Errorf("%s output limit response: %w", check.name, assertionErr)
		}
		observed, observationErr := lastProviderSimulatorRequest(ctx, provider, sessionID)
		if observationErr != nil {
			return fmt.Errorf("%s Ollama provider observation: %w", check.name, observationErr)
		}
		var receipt struct {
			Body map[string]json.RawMessage `json:"body"`
		}
		if decodeErr := json.Unmarshal(observed, &receipt); decodeErr != nil {
			return fmt.Errorf("decode %s Ollama provider observation: %w", check.name, decodeErr)
		}
		if string(receipt.Body["max_tokens"]) != check.wantLimit || string(receipt.Body["model"]) != `"`+ollamaCodecModel+`"` {
			return fmt.Errorf("%s Ollama provider lost the model or output limit: %s", check.name,
				truncateString(string(observed), 500))
		}
		if _, exists := receipt.Body["max_completion_tokens"]; exists {
			return fmt.Errorf("%s Ollama provider received unsupported max_completion_tokens: %s", check.name,
				truncateString(string(observed), 500))
		}
	}

	// The adapter must be scoped to Ollama: the same client limit continues to
	// use the Chat codec's standard field on the ordinary provider.
	const controlPrompt = "Ordinary Chat limit control"
	controlID := "ollama-limit-control-" + uuid.NewString()
	control, err := sendProtocolMatrixRaw(ctx, session, "/v1/chat/completions", map[string]any{
		"model": "astra-chat", "max_tokens": 8,
		"messages": []map[string]string{{"role": "user", "content": controlPrompt}},
	}, false, map[string]string{"x-vsr-test-session-id": controlID})
	if err != nil {
		return fmt.Errorf("ordinary Chat output limit request: %w", err)
	}
	if control.StatusCode != http.StatusOK {
		return fmt.Errorf("ordinary Chat output limit returned HTTP %d: %s", control.StatusCode,
			truncateString(string(control.Body), 500))
	}
	observed, err := lastProviderSimulatorRequest(ctx, provider, controlID)
	if err != nil {
		return fmt.Errorf("ordinary Chat provider observation: %w", err)
	}
	var receipt struct {
		Body map[string]json.RawMessage `json:"body"`
	}
	if decodeErr := json.Unmarshal(observed, &receipt); decodeErr != nil {
		return fmt.Errorf("decode ordinary Chat provider observation: %w", decodeErr)
	}
	if string(receipt.Body["max_completion_tokens"]) != "8" ||
		!strings.Contains(string(observed), controlPrompt) {
		return fmt.Errorf("ordinary Chat provider lost its output limit or marker: %s", truncateString(string(observed), 500))
	}
	if _, exists := receipt.Body["max_tokens"]; exists {
		return fmt.Errorf("ordinary Chat provider unexpectedly received Ollama's max_tokens field: %s",
			truncateString(string(observed), 500))
	}
	return nil
}

func testProtocolCodecAutoUnsupportedCapability(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
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

	// First prove the same auto decision has a reachable backend and the
	// provider observation path is active; a bare 400 could hide a broken stack.
	const prompt = "Auto capability control"
	controlID := "auto-capability-control-" + uuid.NewString()
	control, err := sendProtocolMatrixRaw(ctx, session, "/v1/messages", map[string]any{
		"model": "auto", "max_tokens": 16,
		"messages": []map[string]string{{"role": "user", "content": prompt}},
	}, false, map[string]string{"x-vsr-test-session-id": controlID})
	if err != nil {
		return fmt.Errorf("auto capability control request: %w", err)
	}
	if control.StatusCode != http.StatusOK {
		return fmt.Errorf("auto capability control returned HTTP %d: %s", control.StatusCode,
			truncateString(string(control.Body), 500))
	}
	if assertionErr := assertAnthropicBody(control.Body, `"protocol":"chat_completions"`); assertionErr != nil {
		return fmt.Errorf("auto capability control response: %w", assertionErr)
	}
	if verificationErr := verifyProviderSimulatorRequest(ctx, provider, controlID, "openai.chat.v1", prompt); verificationErr != nil {
		return fmt.Errorf("auto capability control provider dispatch: %w", verificationErr)
	}

	for _, check := range []struct {
		name, model string
	}{{"auto", "auto"}, {"named", chatBackendModel}} {
		sessionID := "disabled-capability-" + uuid.NewString()
		result, requestErr := sendProtocolMatrixRaw(ctx, session, "/v1/messages", map[string]any{
			"model": check.model, "max_tokens": 16,
			"messages": []map[string]string{{"role": "user", "content": "Disabled thinking needs a backend off control"}},
			"thinking": map[string]string{"type": "disabled"},
		}, false, map[string]string{"x-vsr-test-session-id": sessionID})
		if requestErr != nil {
			return fmt.Errorf("%s disabled-thinking request: %w", check.name, requestErr)
		}
		if result.StatusCode != http.StatusBadRequest {
			return fmt.Errorf("%s disabled-thinking request returned HTTP %d, want 400: %s", check.name,
				result.StatusCode, truncateString(string(result.Body), 500))
		}
		var response struct {
			Type  string `json:"type"`
			Error struct {
				Type, Message string
			} `json:"error"`
		}
		if decodeErr := json.Unmarshal(result.Body, &response); decodeErr != nil {
			return fmt.Errorf("decode %s disabled-thinking error: %w", check.name, decodeErr)
		}
		if response.Type != "error" || response.Error.Type != "invalid_request_error" ||
			!strings.Contains(response.Error.Message, "reasoning-off control") {
			return fmt.Errorf("%s disabled-thinking request returned the wrong capability error: %s", check.name,
				truncateString(string(result.Body), 500))
		}
		dispatched, model, err := lookupShortCircuitDispatch(ctx, provider, sessionID)
		if err != nil {
			return fmt.Errorf("%s disabled-thinking provider observation: %w", check.name, err)
		}
		if dispatched {
			return fmt.Errorf("%s unsupported disabled-thinking request reached provider model %q", check.name, model)
		}
	}
	return nil
}
