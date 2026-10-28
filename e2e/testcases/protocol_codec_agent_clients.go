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

func init() {
	pkgtestcases.Register("protocol-codec-agent-client-fields", pkgtestcases.TestCase{
		Description: "Codex Responses metadata and Copilot custom tools reach a Chat backend without losing supported fields",
		Tags:        []string{"protocol-codec", "response-api", "agents", "tools"},
		Fn:          testProtocolCodecAgentClientFields,
	})
	pkgtestcases.Register("protocol-codec-azure-ingress", pkgtestcases.TestCase{
		Description: "Azure Chat and Responses paths preserve wire format, route the model and strip the client API key",
		Tags:        []string{"protocol-codec", "azure", "agents", "security"},
		Fn:          testProtocolCodecAzureIngress,
	})
	pkgtestcases.Register("protocol-codec-reasoning-summary-responses-backend", pkgtestcases.TestCase{
		Description: "Responses reasoning summaries reach a native Responses backend and invalid values fail at ingress",
		Tags:        []string{"protocol-codec", "response-api", "agents"},
		Fn:          testProtocolCodecReasoningSummaryResponsesBackend,
	})
}

func hasProtocolFieldDiagnostic(header, action, field string) bool {
	for _, entry := range strings.Split(header, ",") {
		if strings.HasPrefix(entry, action+";") && strings.HasSuffix(entry, ";"+field) {
			return true
		}
	}
	return false
}

func testProtocolCodecAzureIngress(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, sessionErr := fixtures.OpenServiceSession(ctx, client, opts)
	if sessionErr != nil {
		return sessionErr
	}
	defer session.Close()
	provider, providerErr := openProtocolCodecProviderSession(ctx, client, opts, "openai.chat.v1")
	if providerErr != nil {
		return providerErr
	}
	defer provider.Close()

	cases := []struct {
		name      string
		path      string
		marker    string
		body      map[string]any
		responses bool
	}{
		{
			name: "deployment-chat", path: "/openai/deployments/" + chatBackendModel + "/chat/completions?api-version=2024-10-21",
			marker: "Azure deployment Chat probe",
			body:   map[string]any{"messages": []map[string]string{{"role": "user", "content": "Azure deployment Chat probe"}}},
		},
		{
			name: "dated-responses", path: "/openai/responses?api-version=2025-04-01-preview", responses: true,
			marker: "Azure dated Responses probe",
			body: map[string]any{
				"model": chatBackendModel, "input": "Azure dated Responses probe", "store": false,
				"reasoning": map[string]string{"summary": "auto"},
			},
		},
		{
			name: "v1-responses", path: "/openai/v1/responses", responses: true,
			marker: "Azure v1 Responses probe",
			body: map[string]any{
				"model": chatBackendModel, "input": "Azure v1 Responses probe", "store": false,
				"reasoning": map[string]string{"summary": "auto"},
			},
		},
		{
			name: "v1-chat", path: "/openai/v1/chat/completions",
			marker: "Azure v1 Chat probe",
			body:   map[string]any{"model": chatBackendModel, "messages": []map[string]string{{"role": "user", "content": "Azure v1 Chat probe"}}},
		},
	}
	for _, check := range cases {
		sessionID := "azure-ingress-codec-e2e-" + check.name
		result, requestErr := sendProtocolMatrixRaw(ctx, session, check.path, check.body, false,
			map[string]string{"api-key": "azure-client-test-key", "x-vsr-test-session-id": sessionID})
		if requestErr != nil {
			return fmt.Errorf("%s request: %w", check.name, requestErr)
		}
		if result.StatusCode != http.StatusOK {
			return fmt.Errorf("%s returned HTTP %d: %s", check.name, result.StatusCode, truncateString(string(result.Body), 500))
		}
		if check.responses {
			if err := assertResponsesBody(result.Body, `"protocol":"chat_completions"`); err != nil {
				return fmt.Errorf("%s response: %w", check.name, err)
			}
			if warnings := result.Headers.Get("x-vsr-protocol-warnings"); !hasProtocolFieldDiagnostic(warnings, "dropped", "reasoning.summary") {
				return fmt.Errorf("%s did not report the dropped reasoning summary: %q", check.name, warnings)
			}
		} else if err := assertChatCompletionBody(result.Body, `"protocol":"chat_completions"`); err != nil {
			return fmt.Errorf("%s response: %w", check.name, err)
		}
		raw, observationErr := lastProviderSimulatorRequest(ctx, provider, sessionID)
		if observationErr != nil {
			return fmt.Errorf("%s provider observation: %w", check.name, observationErr)
		}
		var observed struct {
			Body          map[string]json.RawMessage `json:"body"`
			APIKeyPresent bool                       `json:"api_key_present"`
		}
		if err := json.Unmarshal(raw, &observed); err != nil {
			return fmt.Errorf("%s provider observation decode: %w", check.name, err)
		}
		if observed.APIKeyPresent {
			return fmt.Errorf("%s leaked the Azure client api-key to the provider", check.name)
		}
		if check.responses {
			if _, forwarded := observed.Body["reasoning"]; forwarded {
				return fmt.Errorf("%s forwarded reasoning.summary to a Chat backend: %s", check.name, observed.Body["reasoning"])
			}
		}
		if len(observed.Body["model"]) == 0 || len(observed.Body["messages"]) == 0 ||
			!strings.Contains(string(raw), check.marker) {
			return fmt.Errorf("%s did not select and dispatch the model: %s", check.name, truncateString(string(raw), 500))
		}
	}

	for _, path := range []string{
		"/openai/deployments/" + chatBackendModel + "/embeddings?api-version=2024-10-21",
		"/openai/v1/embeddings", "/openai/v1/responses/resp_123",
	} {
		unsupported, unsupportedErr := sendProtocolMatrixRaw(ctx, session, path,
			map[string]any{"input": "Azure unsupported operation"}, false, nil)
		if unsupportedErr != nil {
			return fmt.Errorf("unsupported Azure request %s: %w", path, unsupportedErr)
		}
		if unsupported.StatusCode != http.StatusNotFound {
			return fmt.Errorf("unsupported Azure operation %s returned HTTP %d, want 404", path, unsupported.StatusCode)
		}
	}
	return nil
}

func testProtocolCodecAgentClientFields(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, sessionErr := fixtures.OpenServiceSession(ctx, client, opts)
	if sessionErr != nil {
		return sessionErr
	}
	defer session.Close()
	provider, providerErr := openProtocolCodecProviderSession(ctx, client, opts, "openai.chat.v1")
	if providerErr != nil {
		return providerErr
	}
	defer provider.Close()

	const cacheKey = "agent-client-codec-e2e"
	cases := []struct {
		name    string
		marker  string
		path    string
		body    map[string]any
		inspect func(map[string]json.RawMessage) error
	}{
		{
			name: "codex-responses", marker: "Codex agent field probe", path: "/v1/responses",
			body: map[string]any{
				"model": chatBackendModel, "input": "Codex agent field probe", "store": false,
				"include":          []string{"reasoning.encrypted_content"},
				"client_metadata":  map[string]any{"x-codex-turn-metadata": `{"request_kind":"turn"}`},
				"prompt_cache_key": cacheKey,
				"reasoning":        map[string]string{"summary": "auto"},
			},
			inspect: func(body map[string]json.RawMessage) error {
				if string(body["prompt_cache_key"]) != `"`+cacheKey+`"` {
					return fmt.Errorf("codex cache key was lost in Chat dispatch: %s", body["prompt_cache_key"])
				}
				for _, field := range []string{"include", "client_metadata", "store", "reasoning"} {
					if _, found := body[field]; found {
						return fmt.Errorf("unsupported Codex field %q leaked to Chat provider", field)
					}
				}
				return nil
			},
		},
		{
			name: "copilot-custom-tool", marker: "Copilot custom tool probe", path: "/v1/chat/completions",
			body: map[string]any{
				"model":    chatBackendModel,
				"messages": []map[string]any{{"role": "user", "content": "Copilot custom tool probe"}},
				"tools": []map[string]any{{
					"type": "custom", "custom": map[string]any{
						"name": "bash", "description": "Run a shell command",
						"format": map[string]any{"type": "text"},
					},
				}},
			},
			inspect: func(body map[string]json.RawMessage) error {
				var tools []struct {
					Type   string `json:"type"`
					Custom struct {
						Name string `json:"name"`
					} `json:"custom"`
				}
				if err := json.Unmarshal(body["tools"], &tools); err != nil {
					return fmt.Errorf("decode Copilot provider tools: %w", err)
				}
				if len(tools) != 1 || tools[0].Type != "custom" || tools[0].Custom.Name != "bash" {
					return fmt.Errorf("copilot custom tool was lost in Chat dispatch: %s", body["tools"])
				}
				return nil
			},
		},
	}

	for _, check := range cases {
		sessionID := "agent-client-" + check.name
		result, requestErr := sendProtocolMatrixRaw(ctx, session, check.path, check.body, false,
			map[string]string{"x-vsr-test-session-id": sessionID})
		if requestErr != nil {
			return fmt.Errorf("%s request: %w", check.name, requestErr)
		}
		if result.StatusCode != http.StatusOK {
			return fmt.Errorf("%s returned HTTP %d: %s", check.name, result.StatusCode, truncateString(string(result.Body), 500))
		}
		if check.path == "/v1/responses" {
			if err := assertResponsesBody(result.Body, `"protocol":"chat_completions"`); err != nil {
				return fmt.Errorf("%s response: %w", check.name, err)
			}
			if warnings := result.Headers.Get("x-vsr-protocol-warnings"); !hasProtocolFieldDiagnostic(warnings, "dropped", "reasoning.summary") {
				return fmt.Errorf("%s did not report the dropped reasoning summary: %q", check.name, warnings)
			}
		} else if err := assertChatCompletionBody(result.Body, `"protocol":"chat_completions"`); err != nil {
			return fmt.Errorf("%s response: %w", check.name, err)
		}
		raw, observationErr := lastProviderSimulatorRequest(ctx, provider, sessionID)
		if observationErr != nil {
			return fmt.Errorf("%s provider observation: %w", check.name, observationErr)
		}
		var observation struct {
			Body map[string]json.RawMessage `json:"body"`
		}
		if err := json.Unmarshal(raw, &observation); err != nil {
			return fmt.Errorf("%s provider observation decode: %w", check.name, err)
		}
		if len(observation.Body) == 0 || !strings.Contains(string(raw), check.marker) {
			// The unique marker proves this observation belongs to this request.
			return fmt.Errorf("%s provider observation does not match request: %s", check.name, truncateString(string(raw), 500))
		}
		if err := check.inspect(observation.Body); err != nil {
			return err
		}
	}
	return nil
}

func testProtocolCodecReasoningSummaryResponsesBackend(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
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

	const marker = "Native Responses reasoning summary probe"
	const sessionID = "reasoning-summary-native-responses"
	result, requestErr := sendProtocolMatrixRaw(ctx, session, "/v1/responses", map[string]any{
		"model": nativeResponsesBackendModel, "input": marker, "store": false,
		"reasoning": map[string]string{"summary": "auto"},
	}, false, map[string]string{"x-vsr-test-session-id": sessionID})
	if requestErr != nil {
		return fmt.Errorf("native Responses summary request: %w", requestErr)
	}
	if result.StatusCode != http.StatusOK {
		return fmt.Errorf("native Responses summary returned HTTP %d: %s", result.StatusCode, truncateString(string(result.Body), 500))
	}
	if err := assertResponsesBody(result.Body, protocolCodecResponsesReply); err != nil {
		return fmt.Errorf("native Responses summary response: %w", err)
	}
	if warnings := result.Headers.Get("x-vsr-protocol-warnings"); hasProtocolFieldDiagnostic(warnings, "dropped", "reasoning.summary") {
		return fmt.Errorf("native Responses backend dropped reasoning.summary: %q", warnings)
	}
	raw, observationErr := lastProviderSimulatorRequest(ctx, provider, sessionID)
	if observationErr != nil {
		return fmt.Errorf("native Responses provider observation: %w", observationErr)
	}
	var observed struct {
		Body map[string]json.RawMessage `json:"body"`
	}
	if err := json.Unmarshal(raw, &observed); err != nil {
		return fmt.Errorf("decode native Responses provider observation: %w", err)
	}
	if !strings.Contains(string(raw), marker) {
		return fmt.Errorf("native Responses provider observation does not match request: %s", truncateString(string(raw), 500))
	}
	var reasoning struct {
		Summary string `json:"summary"`
	}
	if err := json.Unmarshal(observed.Body["reasoning"], &reasoning); err != nil {
		return fmt.Errorf("decode native Responses reasoning summary: %w", err)
	}
	if reasoning.Summary != "auto" {
		return fmt.Errorf("native Responses provider lost reasoning.summary: %s", observed.Body["reasoning"])
	}

	invalid, invalidErr := sendProtocolMatrixRaw(ctx, session, "/v1/responses", map[string]any{
		"model": nativeResponsesBackendModel, "input": marker,
		"reasoning": map[string]string{"summary": ""},
	}, false, nil)
	if invalidErr != nil {
		return fmt.Errorf("invalid Responses summary request: %w", invalidErr)
	}
	if invalid.StatusCode != http.StatusBadRequest || !strings.Contains(string(invalid.Body), "invalid_reasoning_summary") {
		return fmt.Errorf("empty reasoning.summary returned HTTP %d: %s", invalid.StatusCode, truncateString(string(invalid.Body), 500))
	}
	return nil
}
