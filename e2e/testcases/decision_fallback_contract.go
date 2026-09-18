package testcases

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
)

func buildFallbackRequest(tc DecisionFallbackCase) fixtures.ChatCompletionsRequest {
	content := tc.Query
	if !tc.ShouldFallback {
		content = fallbackMatchMarker + "\n" + content
	}
	return fixtures.ChatCompletionsRequest{Model: fallbackEntrypoint, Messages: []fixtures.ChatMessage{{Role: "user", Content: content}}}
}

func expectedFallbackModel(tc DecisionFallbackCase) string {
	if tc.ShouldFallback {
		return fallbackDefaultModel
	}
	return fallbackMatchedModel
}

func validateFallbackResponse(tc DecisionFallbackCase, header http.Header, body []byte) error {
	if (tc.ExpectedDecision == "") != tc.ShouldFallback || (!tc.ShouldFallback && tc.ExpectedDecision != fallbackMatchDecision) {
		return fmt.Errorf("inconsistent fallback fixture oracle: decision=%q fallback=%t", tc.ExpectedDecision, tc.ShouldFallback)
	}
	for name, expected := range map[string]string{
		"x-vsr-schema-version": "2", "x-vsr-response-path": "upstream",
		"x-vsr-selected-recipe": fallbackEntrypoint, "x-vsr-selected-decision": tc.ExpectedDecision,
		"x-vsr-selected-model": expectedFallbackModel(tc),
	} {
		if got := header.Get(name); got != expected {
			return fmt.Errorf("fallback contract %s=%q, want %q", name, got, expected)
		}
	}
	var response struct {
		Object  string `json:"object"`
		Model   string `json:"model"`
		Choices []struct {
			Message fixtures.ChatMessage `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return fmt.Errorf("decode fallback completion: %w", err)
	}
	if response.Object != "chat.completion" || response.Model != expectedFallbackModel(tc) ||
		len(response.Choices) != 1 || response.Choices[0].Message.Role != "assistant" || response.Choices[0].Message.Content == "" {
		return fmt.Errorf("fallback did not return the selected provider's chat completion")
	}
	return nil
}

func validateFallbackProviderRequest(tc DecisionFallbackCase, observed []byte) error {
	var request struct {
		Body fixtures.ChatCompletionsRequest `json:"body"`
	}
	if err := json.Unmarshal(observed, &request); err != nil {
		return fmt.Errorf("decode fallback provider request: %w", err)
	}
	want := buildFallbackRequest(tc)
	if request.Body.Model != expectedFallbackModel(tc) || len(request.Body.Messages) != 1 || request.Body.Messages[0] != want.Messages[0] {
		return fmt.Errorf("provider did not receive the selected model and original fallback fixture message")
	}
	return nil
}
