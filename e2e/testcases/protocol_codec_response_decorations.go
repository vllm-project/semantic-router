package testcases

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("protocol-codec-anthropic-response-diagnostics", pkgtestcases.TestCase{
		Description: "Anthropic cache diagnostics allow buffered and streamed replies for Messages, Chat, and Responses clients",
		Tags:        []string{"protocol-codec", "anthropic", "response-api", "streaming"},
		Fn:          testAnthropicResponseDiagnostics,
	})
	pkgtestcases.Register("protocol-codec-responses-provider-decorations", pkgtestcases.TestCase{
		Description: "OpenAI Responses provider decorations allow buffered and streamed replies for Responses, Chat, and Messages clients",
		Tags:        []string{"protocol-codec", "response-api", "streaming"},
		Fn:          testResponsesProviderDecorations,
	})
}

func testAnthropicResponseDiagnostics(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	return runResponseDecorationMatrix(ctx, client, opts, "MoM", "anthropic.messages.v1",
		"__mock_anthropic_diagnostics__ __mock_protocol_matrix__", protocolCodecAnthropicReply,
		assertAnthropicResponseDiagnostics)
}

func testResponsesProviderDecorations(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	return runResponseDecorationMatrix(ctx, client, opts, nativeResponsesBackendModel, "openai.responses.v1",
		"__mock_responses_decorations__", protocolCodecResponsesReply,
		assertResponsesProviderDecorations)
}

func runResponseDecorationMatrix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	model, backendFormat, prompt, expectedText string,
	assertDecoration func(protocolCodecE2EClient, bool, protocolMatrixHTTPResult) error,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	provider, err := openProtocolCodecProviderSession(ctx, client, opts, backendFormat)
	if err != nil {
		return err
	}
	defer provider.Close()

	var failures []error
	results := make(map[string]string, 2*len(protocolCodecE2EClients))
	for _, clientContract := range protocolCodecE2EClients {
		for _, streamed := range []bool{false, true} {
			mode := "buffered"
			if streamed {
				mode = "streamed"
			}
			cell := clientContract.name + "/" + mode
			sessionID := "response-decorations-" + protocolMatrixSessionID(backendFormat, clientContract.name, mode)
			result, requestErr := sendProtocolMatrixRaw(ctx, session, clientContract.path,
				clientContract.request(model, prompt, streamed), streamed,
				map[string]string{"x-vsr-test-session-id": sessionID})
			if requestErr == nil && result.StatusCode != http.StatusOK {
				requestErr = fmt.Errorf("HTTP %d: %s", result.StatusCode, truncateString(string(result.Body), 500))
			}
			if requestErr == nil {
				if streamed {
					requestErr = clientContract.validateStream(result.Body, expectedText)
				} else {
					requestErr = clientContract.validateBuffered(result.Body, expectedText)
				}
			}
			if requestErr == nil {
				requestErr = assertDecoration(clientContract, streamed, result)
			}
			if requestErr == nil {
				requestErr = verifyProviderSimulatorRequest(ctx, provider, sessionID, backendFormat, prompt)
			}
			if requestErr != nil {
				results[cell] = "failed"
				failures = append(failures, fmt.Errorf("%s: %w", cell, requestErr))
				continue
			}
			results[cell] = "passed"
		}
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"backend_format": backendFormat, "matrix_cells": results})
	}
	return errors.Join(failures...)
}

func assertAnthropicResponseDiagnostics(client protocolCodecE2EClient, streamed bool, result protocolMatrixHTTPResult) error {
	if streamed {
		if client.path == "/v1/messages" {
			if !strings.Contains(string(result.Body), `"diagnostics":`) ||
				!strings.Contains(string(result.Body), `"tools_changed"`) {
				return fmt.Errorf("native Messages stream lost cache diagnostics: %s", truncateString(string(result.Body), 600))
			}
			return nil
		}
		return rejectDecoratedStreamFields(result.Body, "diagnostics")
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(result.Body, &body); err != nil {
		return err
	}
	if client.path == "/v1/messages" {
		var diagnostics struct {
			CacheMissReason struct {
				Type                   string `json:"type"`
				CacheMissedInputTokens int    `json:"cache_missed_input_tokens"`
			} `json:"cache_miss_reason"`
		}
		if err := json.Unmarshal(body["diagnostics"], &diagnostics); err != nil {
			return fmt.Errorf("native Messages response lost cache diagnostics: %w", err)
		}
		if diagnostics.CacheMissReason.Type != "tools_changed" || diagnostics.CacheMissReason.CacheMissedInputTokens != 4 {
			return fmt.Errorf("native Messages response changed cache diagnostics: %s", truncateString(string(result.Body), 600))
		}
	} else if _, leaked := body["diagnostics"]; leaked {
		return fmt.Errorf("%s response leaked Anthropic-only diagnostics: %s", client.name, truncateString(string(result.Body), 600))
	}
	if warnings := result.Headers.Get("x-vsr-protocol-warnings"); !hasProtocolFieldDiagnostic(warnings, "dropped", "diagnostics") {
		return fmt.Errorf("anthropic cache diagnostics omitted without a warning: %q", warnings)
	}
	return nil
}

func assertResponsesProviderDecorations(_ protocolCodecE2EClient, streamed bool, result protocolMatrixHTTPResult) error {
	fields := []string{"access_programs", "billing", "frequency_penalty", "presence_penalty", "tool_usage"}
	if streamed {
		return rejectDecoratedStreamFields(result.Body, fields...)
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(result.Body, &body); err != nil {
		return err
	}
	// The Responses API assigns its own public response ID, so even a native
	// Responses backend reply is rewritten before it reaches the client.
	for _, field := range fields {
		if _, leaked := body[field]; leaked {
			return fmt.Errorf("response leaked provider-only %s: %s", field, truncateString(string(result.Body), 600))
		}
	}
	warnings := result.Headers.Get("x-vsr-protocol-warnings")
	for _, field := range []string{"billing", "frequency_penalty", "presence_penalty", "tool_usage"} {
		if !hasProtocolFieldDiagnostic(warnings, "dropped", field) {
			return fmt.Errorf("responses provider field %s omitted without a warning: %q", field, warnings)
		}
	}
	if hasProtocolFieldDiagnostic(warnings, "dropped", "access_programs") {
		return fmt.Errorf("null access_programs incorrectly reported as a loss: %q", warnings)
	}
	return nil
}

func rejectDecoratedStreamFields(body []byte, fields ...string) error {
	for _, field := range fields {
		if strings.Contains(string(body), `"`+field+`":`) {
			return fmt.Errorf("stream leaked provider-only %s: %s", field, truncateString(string(body), 600))
		}
	}
	return nil
}
