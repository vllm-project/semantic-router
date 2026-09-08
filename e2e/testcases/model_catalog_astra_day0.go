package testcases

import (
	"context"
	"encoding/json"
	"fmt"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const (
	astraChatAlias      = "astra-chat-day0"
	astraResponsesAlias = "astra-responses-day0"
	astraProviderModel  = "gpt-6-astra"
)

func init() {
	pkgtestcases.Register("model-catalog-astra-day0", pkgtestcases.TestCase{
		Description: "A catalog-only Day-0 model materializes and emits exact OpenAI Chat and Responses reasoning controls",
		Tags:        []string{"model-catalog", "day0", "openai", "reasoning", "response-api"},
		Fn:          testModelCatalogAstraDay0,
	})
}

func testModelCatalogAstraDay0(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	routerSession, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer routerSession.Close()

	providerSession, err := openProtocolCodecProviderSession(ctx, client, opts, "openai.responses.v1")
	if err != nil {
		return err
	}
	defer providerSession.Close()

	chatSessionID := "astra-day0-chat"
	if _, err := sendProtocolMatrixRequestWithHeaders(
		ctx,
		routerSession,
		"/v1/chat/completions",
		map[string]any{
			"model": astraChatAlias,
			"messages": []map[string]string{{
				"role": "user", "content": "Astra Day-0 Chat contract",
			}},
			"reasoning_effort": "xhigh",
		},
		false,
		map[string]string{"x-vsr-test-session-id": chatSessionID},
	); err != nil {
		return fmt.Errorf("astra Chat request: %w", err)
	}
	if err := verifyAstraProviderRequest(ctx, providerSession, chatSessionID, false, "xhigh", false); err != nil {
		return err
	}

	responsesSessionID := "astra-day0-responses"
	if _, err := sendProtocolMatrixRequestWithHeaders(
		ctx,
		routerSession,
		"/v1/responses",
		map[string]any{
			"model":     astraResponsesAlias,
			"input":     "Astra Day-0 Responses contract",
			"reasoning": map[string]any{"effort": "max"},
			"store":     false,
		},
		false,
		map[string]string{"x-vsr-test-session-id": responsesSessionID},
	); err != nil {
		return fmt.Errorf("astra Responses request: %w", err)
	}
	if err := verifyAstraProviderRequest(ctx, providerSession, responsesSessionID, true, "max", false); err != nil {
		return err
	}

	toolSessionID := "astra-day0-responses-tools"
	if _, err := sendProtocolMatrixRequestWithHeaders(
		ctx,
		routerSession,
		"/v1/responses",
		map[string]any{
			"model":     astraResponsesAlias,
			"input":     "__mock_tool_call__",
			"reasoning": map[string]any{"effort": "high"},
			"store":     false,
			"tools":     []any{protocolLookupTool()},
			"tool_choice": map[string]any{
				"type": "function",
				"name": "lookup",
			},
		},
		false,
		map[string]string{"x-vsr-test-session-id": toolSessionID},
	); err != nil {
		return fmt.Errorf("astra Responses tool request: %w", err)
	}
	if err := verifyAstraProviderRequest(ctx, providerSession, toolSessionID, true, "high", true); err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"catalog":            "openai/gpt-6-astra",
			"provider_model_id":  astraProviderModel,
			"protocols_verified": []string{"openai.chat.v1", "openai.responses.v1"},
			"responses_tools":    true,
		})
	}
	return nil
}

func verifyAstraProviderRequest(
	ctx context.Context,
	providerSession *fixtures.ServiceSession,
	sessionID string,
	responses bool,
	wantEffort string,
	wantTools bool,
) error {
	observed, err := lastProviderSimulatorRequest(ctx, providerSession, sessionID)
	if err != nil {
		return fmt.Errorf("read Astra provider request: %w", err)
	}
	var debug struct {
		Body map[string]json.RawMessage `json:"body"`
	}
	if err := json.Unmarshal(observed, &debug); err != nil {
		return fmt.Errorf("decode Astra provider request: %w", err)
	}

	var model string
	if err := json.Unmarshal(debug.Body["model"], &model); err != nil || model != astraProviderModel {
		return fmt.Errorf("astra provider model = %q, want %q: %s", model, astraProviderModel, truncateString(string(observed), 600))
	}
	if responses {
		if _, found := debug.Body["input"]; !found {
			return fmt.Errorf("astra Responses request lost input: %s", truncateString(string(observed), 600))
		}
		if _, found := debug.Body["messages"]; found {
			return fmt.Errorf("astra Responses request leaked messages: %s", truncateString(string(observed), 600))
		}
		if _, found := debug.Body["reasoning_effort"]; found {
			return fmt.Errorf("astra Responses request leaked top-level reasoning_effort: %s", truncateString(string(observed), 600))
		}
		var reasoning struct {
			Effort string `json:"effort"`
		}
		if err := json.Unmarshal(debug.Body["reasoning"], &reasoning); err != nil || reasoning.Effort != wantEffort {
			return fmt.Errorf("astra Responses effort = %q, want %q: %s", reasoning.Effort, wantEffort, truncateString(string(observed), 600))
		}
		_, hasTools := debug.Body["tools"]
		if hasTools != wantTools {
			return fmt.Errorf("astra Responses tools present = %t, want %t: %s", hasTools, wantTools, truncateString(string(observed), 600))
		}
		if wantTools {
			var tools []struct {
				Type        string `json:"type"`
				Name        string `json:"name"`
				Description string `json:"description"`
				Parameters  struct {
					Type       string `json:"type"`
					Properties map[string]struct {
						Type string `json:"type"`
					} `json:"properties"`
					Required []string `json:"required"`
				} `json:"parameters"`
			}
			if err := json.Unmarshal(debug.Body["tools"], &tools); err != nil || len(tools) != 1 ||
				tools[0].Type != "function" || tools[0].Name != "lookup" ||
				tools[0].Description != "Look up a value" || tools[0].Parameters.Type != "object" ||
				tools[0].Parameters.Properties["query"].Type != "string" ||
				len(tools[0].Parameters.Required) != 1 || tools[0].Parameters.Required[0] != "query" {
				return fmt.Errorf("astra Responses tool schema was not preserved: %s", truncateString(string(observed), 600))
			}
			var toolChoice struct {
				Type string `json:"type"`
				Name string `json:"name"`
			}
			if err := json.Unmarshal(debug.Body["tool_choice"], &toolChoice); err != nil ||
				toolChoice.Type != "function" || toolChoice.Name != "lookup" {
				return fmt.Errorf("astra Responses tool choice was not preserved: %s", truncateString(string(observed), 600))
			}
		}
		return nil
	}

	if _, found := debug.Body["messages"]; !found {
		return fmt.Errorf("astra Chat request lost messages: %s", truncateString(string(observed), 600))
	}
	if _, found := debug.Body["input"]; found {
		return fmt.Errorf("astra Chat request leaked input: %s", truncateString(string(observed), 600))
	}
	if _, found := debug.Body["reasoning"]; found {
		return fmt.Errorf("astra Chat request leaked a reasoning object: %s", truncateString(string(observed), 600))
	}
	var effort string
	if err := json.Unmarshal(debug.Body["reasoning_effort"], &effort); err != nil || effort != wantEffort {
		return fmt.Errorf("astra Chat effort = %q, want %q: %s", effort, wantEffort, truncateString(string(observed), 600))
	}
	return nil
}
