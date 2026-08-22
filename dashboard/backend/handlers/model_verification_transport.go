package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

var errModelVerificationResponseTooLarge = errors.New("provider response exceeded the size limit")

const modelVerificationReasoningSummary = "Provider returned a valid reasoning response."

func (service *modelVerificationService) call(ctx context.Context, target modelVerificationTarget) (string, error) {
	payload, err := buildModelVerificationPayload(target)
	if err != nil {
		return "", modelVerificationFailure(http.StatusInternalServerError, "verification_failed", "Model verification request could not be prepared.")
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, target.endpointURL, bytes.NewReader(payload))
	if err != nil {
		return "", modelVerificationFailure(http.StatusUnprocessableEntity, "backend_invalid", "The configured provider backend URL is invalid.")
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json")
	for key, value := range target.extraHeaders {
		if modelVerificationHeaderAllowed(key) {
			request.Header.Set(key, value)
		}
	}
	if target.apiKey != "" && modelVerificationHeaderAllowed(target.authHeader) {
		value := target.apiKey
		if strings.TrimSpace(target.authPrefix) != "" {
			value = strings.TrimSpace(target.authPrefix) + " " + value
		}
		request.Header.Set(target.authHeader, value)
	}
	if target.dialect == "anthropic" && request.Header.Get("anthropic-version") == "" {
		request.Header.Set("anthropic-version", "2023-06-01")
	}

	response, err := service.client.Do(request)
	if err != nil {
		if errors.Is(ctx.Err(), context.DeadlineExceeded) || errors.Is(err, context.DeadlineExceeded) {
			return "", modelVerificationFailure(
				http.StatusGatewayTimeout,
				"verification_timeout",
				"Model inference timed out before a generation response was received.",
			)
		}
		return "", modelVerificationFailure(
			http.StatusBadGateway,
			"provider_unreachable",
			"The provider inference request failed before a generation response was received.",
		)
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return "", modelVerificationFailure(
			http.StatusBadGateway,
			"provider_rejected",
			fmt.Sprintf("Provider inference returned HTTP %d.", response.StatusCode),
		)
	}
	body, err := readBoundedModelVerificationResponse(response.Body)
	if err != nil {
		if errors.Is(err, errModelVerificationResponseTooLarge) {
			return "", modelVerificationFailure(http.StatusBadGateway, "response_too_large", "The provider returned an oversized verification response.")
		}
		return "", modelVerificationFailure(http.StatusBadGateway, "invalid_provider_response", "The provider verification response could not be read.")
	}
	content, err := decodeModelVerificationContent(body, target.dialect)
	if err != nil || strings.TrimSpace(content) == "" {
		return "", modelVerificationFailure(
			http.StatusBadGateway,
			"invalid_provider_response",
			"The provider returned no generated text or reasoning for the verification query.",
		)
	}
	return redactModelVerificationSecrets(content, target.secrets...), nil
}

func buildModelVerificationPayload(target modelVerificationTarget) ([]byte, error) {
	if target.dialect == "anthropic" {
		return json.Marshal(struct {
			Model     string `json:"model"`
			MaxTokens int    `json:"max_tokens"`
			Stream    bool   `json:"stream"`
			Messages  []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}{
			Model:     target.providerModel,
			MaxTokens: modelVerificationMaxTokens,
			Stream:    false,
			Messages: []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			}{{Role: "user", Content: modelVerificationPrompt}},
		})
	}
	payload := struct {
		Model               string              `json:"model"`
		Messages            []openAIChatMessage `json:"messages"`
		Stream              bool                `json:"stream"`
		MaxTokens           int                 `json:"max_tokens,omitempty"`
		MaxCompletionTokens int                 `json:"max_completion_tokens,omitempty"`
	}{
		Model:    target.providerModel,
		Messages: []openAIChatMessage{{Role: "user", Content: modelVerificationPrompt}},
		Stream:   false,
	}
	if target.provider == "openai" || target.provider == "azure-openai" {
		payload.MaxCompletionTokens = modelVerificationMaxTokens
	} else {
		payload.MaxTokens = modelVerificationMaxTokens
	}
	return json.Marshal(payload)
}

func readBoundedModelVerificationResponse(reader io.Reader) ([]byte, error) {
	body, err := io.ReadAll(io.LimitReader(reader, modelVerificationMaxResponseBytes+1))
	if err != nil {
		return nil, err
	}
	if len(body) > modelVerificationMaxResponseBytes {
		return nil, errModelVerificationResponseTooLarge
	}
	return body, nil
}

func decodeModelVerificationContent(body []byte, dialect string) (string, error) {
	if dialect == "anthropic" {
		var response struct {
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		}
		if err := json.Unmarshal(body, &response); err != nil {
			return "", err
		}
		parts := make([]string, 0, len(response.Content))
		for _, part := range response.Content {
			if part.Type == "text" && strings.TrimSpace(part.Text) != "" {
				parts = append(parts, strings.TrimSpace(part.Text))
			}
		}
		return strings.Join(parts, " "), nil
	}
	var response struct {
		Choices []modelVerificationOpenAIChoice `json:"choices"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return "", err
	}
	if len(response.Choices) == 0 {
		return "", errors.New("response has no choices")
	}
	for _, choice := range response.Choices {
		if content := extractOpenAIChatContent(choice.Message.Content); content != "" {
			return content, nil
		}
		if content := strings.TrimSpace(choice.Text); content != "" {
			return content, nil
		}
	}
	for _, choice := range response.Choices {
		for _, reasoning := range []json.RawMessage{
			choice.Message.ReasoningContent,
			choice.Message.Reasoning,
			choice.ReasoningContent,
			choice.Reasoning,
		} {
			if extractOpenAIChatContent(reasoning) != "" {
				return modelVerificationReasoningSummary, nil
			}
		}
	}
	return "", nil
}

// vLLM and compatible gateways expose reasoning output in either the message
// or choice envelope. A short verification token budget can legitimately end
// before a reasoning model emits final content, so reasoning is valid evidence
// that the configured model executed. The reasoning itself is never returned
// to the dashboard.
type modelVerificationOpenAIChoice struct {
	Message struct {
		Content          json.RawMessage `json:"content"`
		ReasoningContent json.RawMessage `json:"reasoning_content"`
		Reasoning        json.RawMessage `json:"reasoning"`
	} `json:"message"`
	Text             string          `json:"text"`
	ReasoningContent json.RawMessage `json:"reasoning_content"`
	Reasoning        json.RawMessage `json:"reasoning"`
}
