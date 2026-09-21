package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const (
	ragProviderBoundaryDecision = "rag_provider_boundary_decision"
	ragProviderBoundaryMarker   = "__RAG_PROVIDER_BOUNDARY__"
	ragProviderBoundaryContext  = "E2E RAG fact: Project Zephyr rotates signing keys every 17 days."
)

func init() {
	pkgtestcases.Register("rag-provider-boundary", pkgtestcases.TestCase{
		Description: "Verify RAG retrieval injects correlated tool context into the provider request",
		Tags:        []string{"plugin", "rag", "provider-boundary"},
		Fn:          testRAGProviderBoundary,
	})
}

func testRAGProviderBoundary(
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

	sessionID := fmt.Sprintf("rag-provider-boundary-%d", time.Now().UnixNano())
	response, err := sendProtocolMatrixRaw(
		ctx,
		session,
		"/v1/chat/completions",
		map[string]any{
			"model": "e2e-plugins",
			"messages": []any{map[string]any{
				"role": "user",
				"content": ragProviderBoundaryMarker +
					" What is Project Zephyr's signing-key rotation interval?",
			}},
		},
		false,
		map[string]string{
			"x-vsr-debug":           "true",
			"x-vsr-test-session-id": sessionID,
		},
	)
	if err != nil {
		return err
	}
	if response.StatusCode != http.StatusOK ||
		response.Headers.Get("x-vsr-response-path") != "upstream" ||
		response.Headers.Get("x-vsr-selected-recipe") != "e2e-plugins" ||
		response.Headers.Get("x-vsr-selected-decision") != ragProviderBoundaryDecision {
		return fmt.Errorf("RAG probe did not dispatch through its retrieval policy: status=%d headers=%v",
			response.StatusCode, response.Headers)
	}
	if responseErr := validateRAGProviderBoundaryResponse(response.Body); responseErr != nil {
		return responseErr
	}

	observed, err := lastProviderSimulatorRequest(ctx, backendSession, sessionID)
	if err != nil {
		return err
	}
	if providerErr := validateRAGProviderBoundaryRequest(observed); providerErr != nil {
		return providerErr
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"plugins_verified":           []string{"rag"},
			"provider_boundary_verified": true,
			"retrieval_verified":         true,
			"protocol_preserved":         true,
		})
	}
	return nil
}

func validateRAGProviderBoundaryResponse(body []byte) error {
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
		return fmt.Errorf("RAG changed the Chat Completions response contract: %s",
			truncateString(string(body), 600))
	}
	return nil
}

func validateRAGProviderBoundaryRequest(observed []byte) error {
	var request struct {
		Body struct {
			Messages []ragProviderBoundaryMessage `json:"messages"`
		} `json:"body"`
	}
	if err := json.Unmarshal(observed, &request); err != nil {
		return fmt.Errorf("decode provider-bound RAG request: %w", err)
	}
	messages := request.Body.Messages
	if len(messages) != 3 {
		return fmt.Errorf("provider messages = %d, want user plus RAG tool exchange: %s",
			len(messages), truncateString(string(observed), 800))
	}
	wantUser := ragProviderBoundaryMarker +
		" What is Project Zephyr's signing-key rotation interval?"
	if messages[0].Role != "user" || messages[0].Content != wantUser {
		return fmt.Errorf("RAG changed the user request: %#v", messages[0])
	}
	if err := validateRAGToolCall(messages[1]); err != nil {
		return err
	}
	callID := messages[1].ToolCalls[0].ID
	if messages[2].Role != "tool" ||
		messages[2].ToolCallID != callID ||
		messages[2].Content != ragProviderBoundaryContext {
		return fmt.Errorf("RAG tool result mismatch: %#v", messages[2])
	}
	return nil
}

type ragProviderBoundaryMessage struct {
	Role       string `json:"role"`
	Content    string `json:"content"`
	ToolCallID string `json:"tool_call_id"`
	ToolCalls  []struct {
		ID       string `json:"id"`
		Type     string `json:"type"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	} `json:"tool_calls"`
}

func validateRAGToolCall(message ragProviderBoundaryMessage) error {
	if message.Role != "assistant" || len(message.ToolCalls) != 1 {
		return fmt.Errorf("RAG assistant tool call missing: %#v", message)
	}
	call := message.ToolCalls[0]
	if !strings.HasPrefix(call.ID, "rag_") ||
		call.Type != "function" ||
		call.Function.Name != "vsr_rag_context" ||
		call.Function.Arguments != "{}" {
		return fmt.Errorf("RAG tool call mismatch: %#v", call)
	}
	return nil
}
