package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	pathpkg "path"
	"strings"
	"sync/atomic"

	sharednlgen "github.com/vllm-project/semantic-router/src/semantic-router/pkg/nlgen"
)

type builderNLLLMClient struct {
	envoyURL       string
	req            BuilderNLGenerateRequest
	runtimeOptions builderNLRuntimeOptions
	reporter       builderNLProgressReporter
	currentAttempt atomic.Int64
}

func newBuilderNLLLMClient(
	envoyURL string,
	req BuilderNLGenerateRequest,
	runtimeOptions builderNLRuntimeOptions,
	reporter builderNLProgressReporter,
) *builderNLLLMClient {
	return &builderNLLLMClient{
		envoyURL:       envoyURL,
		req:            req,
		runtimeOptions: runtimeOptions,
		reporter:       reporter,
	}
}

func (c *builderNLLLMClient) setAttempt(attempt int) {
	if attempt > 0 {
		c.currentAttempt.Store(int64(attempt))
	}
}

func (c *builderNLLLMClient) ChatCompletion(ctx context.Context, req sharednlgen.ChatCompletionRequest) (string, error) {
	attempt := safeBuilderNLAttempt(c.currentAttempt.Load())
	if attempt <= 0 {
		attempt = 1
	}
	return callBuilderNLMessages(ctx, c.envoyURL, c.req, c.runtimeOptions, req, c.reporter, "model_call", attempt)
}

func safeBuilderNLAttempt(value int64) int {
	if value <= 0 {
		return 0
	}
	if value > int64(^uint(0)>>1) {
		return int(^uint(0) >> 1)
	}
	return int(value)
}

func callBuilderNLMessages(
	ctx context.Context,
	envoyURL string,
	req BuilderNLGenerateRequest,
	runtimeOptions builderNLRuntimeOptions,
	chatReq sharednlgen.ChatCompletionRequest,
	reporter builderNLProgressReporter,
	phase string,
	attempt int,
) (string, error) {
	if chatReq.MaxTokens <= 0 {
		chatReq.MaxTokens = builderNLDefaultMaxTokens
	}
	callCtx, cancel := context.WithTimeout(ctx, runtimeOptions.Timeout)
	defer cancel()

	return runBuilderNLModelCallWithProgress(callCtx, reporter, phase, attempt, runtimeOptions.Timeout, func() (string, error) {
		connectionMode, err := builderNLConnectionModeOrDefault(req.ConnectionMode)
		if err != nil {
			return "", err
		}

		if connectionMode == builderNLConnectionModeCustom {
			if req.CustomConnection == nil {
				return "", fmt.Errorf("custom connection details are missing")
			}
			reportBuilderNLProgress(
				reporter,
				phase,
				builderNLProgressInfo,
				fmt.Sprintf(
					"Calling custom %s generation model %q.",
					req.CustomConnection.ProviderKind,
					strings.TrimSpace(req.CustomConnection.ModelName),
				),
				attempt,
			)
			return callBuilderNLCustomConnectionMessages(callCtx, *req.CustomConnection, chatReq)
		}

		if strings.TrimSpace(envoyURL) == "" {
			return "", fmt.Errorf("default Builder AI connection is unavailable because Envoy is not configured")
		}
		reportBuilderNLProgress(
			reporter,
			phase,
			builderNLProgressInfo,
			fmt.Sprintf("Calling default Builder generator model %q through the runtime gateway.", builderNLFallbackModelAlias),
			attempt,
		)
		return callBuilderNLOpenAICompatibleMessages(
			callCtx,
			strings.TrimRight(envoyURL, "/")+"/v1/chat/completions",
			builderNLFallbackModelAlias,
			"",
			chatReq,
		)
	})
}

func callBuilderNLCustomConnectionMessages(
	ctx context.Context,
	conn builderNLConnection,
	chatReq sharednlgen.ChatCompletionRequest,
) (string, error) {
	modelName := strings.TrimSpace(conn.ModelName)
	if modelName == "" {
		return "", fmt.Errorf("custom connection modelName is required")
	}
	parsedBaseURL, err := normalizeBuilderNLBaseURL(conn.BaseURL, conn.ProviderKind)
	if err != nil {
		return "", err
	}

	switch conn.ProviderKind {
	case builderNLProviderVLLM, builderNLProviderOpenAICompatible:
		return callBuilderNLOpenAICompatibleMessages(
			ctx,
			resolveBuilderNLOpenAIURL(parsedBaseURL),
			modelName,
			strings.TrimSpace(conn.AccessKey),
			chatReq,
		)
	case builderNLProviderAnthropic:
		return callBuilderNLAnthropicMessages(
			ctx,
			resolveBuilderNLAnthropicURL(parsedBaseURL),
			modelName,
			strings.TrimSpace(conn.AccessKey),
			chatReq,
		)
	default:
		return "", fmt.Errorf("unsupported custom connection providerKind %q", conn.ProviderKind)
	}
}

func resolveBuilderNLOpenAIURL(baseURL *url.URL) string {
	candidate := *baseURL
	path := strings.TrimRight(candidate.Path, "/")
	switch {
	case strings.HasSuffix(path, "/chat/completions"):
		candidate.Path = path
	case strings.HasSuffix(path, "/v1"):
		candidate.Path = path + "/chat/completions"
	default:
		candidate.Path = pathpkg.Join(path, "/v1/chat/completions")
	}
	return candidate.String()
}

func resolveBuilderNLAnthropicURL(baseURL *url.URL) string {
	candidate := *baseURL
	path := strings.TrimRight(candidate.Path, "/")
	switch {
	case strings.HasSuffix(path, "/messages"):
		candidate.Path = path
	case strings.HasSuffix(path, "/v1"):
		candidate.Path = path + "/messages"
	default:
		candidate.Path = pathpkg.Join(path, "/v1/messages")
	}
	return candidate.String()
}

func callBuilderNLOpenAICompatibleMessages(
	ctx context.Context,
	endpoint string,
	modelName string,
	accessKey string,
	chatReq sharednlgen.ChatCompletionRequest,
) (string, error) {
	payload := openAIChatRequest{
		Model:     modelName,
		Messages:  buildOpenAIChatMessages(chatReq.Messages),
		Stream:    false,
		MaxTokens: chatReq.MaxTokens,
	}
	if chatReq.Temperature >= 0 {
		temperature := chatReq.Temperature
		payload.Temperature = &temperature
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to marshal builder ai request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(raw))
	if err != nil {
		return "", fmt.Errorf("failed to create builder ai request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")
	if accessKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+accessKey)
	}

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("builder ai request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, _ := io.ReadAll(resp.Body)
	trimmedBody := strings.TrimSpace(string(body))
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		if trimmedBody == "" {
			trimmedBody = resp.Status
		}
		return "", fmt.Errorf("builder ai request failed: %s", trimmedBody)
	}

	var parsed openAIChatResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return "", fmt.Errorf("failed to decode builder ai response: %w", err)
	}
	if parsed.Error != nil && strings.TrimSpace(parsed.Error.Message) != "" {
		return "", fmt.Errorf("builder ai request failed: %s", parsed.Error.Message)
	}
	if len(parsed.Choices) == 0 {
		return "", fmt.Errorf("builder ai request failed: empty response")
	}
	content := extractOpenAIChatChoiceContent(parsed.Choices[0])
	if content == "" {
		return "", fmt.Errorf("builder ai request failed: empty response content")
	}
	return content, nil
}

func buildOpenAIChatMessages(messages []sharednlgen.ChatMessage) []openAIChatMessage {
	result := make([]openAIChatMessage, 0, len(messages))
	for _, message := range messages {
		role := strings.TrimSpace(message.Role)
		if role == "" {
			role = "user"
		}
		result = append(result, openAIChatMessage{
			Role:    role,
			Content: message.Content,
		})
	}
	return result
}

func callBuilderNLAnthropicMessages(
	ctx context.Context,
	endpoint string,
	modelName string,
	accessKey string,
	chatReq sharednlgen.ChatCompletionRequest,
) (string, error) {
	if accessKey == "" {
		return "", fmt.Errorf("anthropic custom connection requires an accessKey")
	}

	var systemParts []string
	var anthropicMessages []anthropicMessageRequest
	for _, message := range chatReq.Messages {
		role := strings.TrimSpace(strings.ToLower(message.Role))
		content := strings.TrimSpace(message.Content)
		if content == "" {
			continue
		}
		if role == "system" {
			systemParts = append(systemParts, content)
			continue
		}
		if role != "assistant" {
			role = "user"
		}
		anthropicMessages = append(anthropicMessages, anthropicMessageRequest{
			Role:    role,
			Content: content,
		})
	}
	if len(anthropicMessages) == 0 {
		return "", fmt.Errorf("anthropic builder ai request failed: no user message content")
	}

	payload := anthropicRequest{
		Model:     modelName,
		MaxTokens: chatReq.MaxTokens,
		System:    strings.Join(systemParts, "\n\n"),
		Messages:  anthropicMessages,
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to marshal anthropic builder ai request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(raw))
	if err != nil {
		return "", fmt.Errorf("failed to create anthropic builder ai request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")
	httpReq.Header.Set("x-api-key", accessKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("anthropic builder ai request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, _ := io.ReadAll(resp.Body)
	trimmedBody := strings.TrimSpace(string(body))
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		if trimmedBody == "" {
			trimmedBody = resp.Status
		}
		return "", fmt.Errorf("anthropic builder ai request failed: %s", trimmedBody)
	}

	var parsed anthropicResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return "", fmt.Errorf("failed to decode anthropic builder ai response: %w", err)
	}
	if parsed.Error != nil && strings.TrimSpace(parsed.Error.Message) != "" {
		return "", fmt.Errorf("anthropic builder ai request failed: %s", parsed.Error.Message)
	}
	for _, part := range parsed.Content {
		if strings.TrimSpace(part.Text) != "" {
			return strings.TrimSpace(part.Text), nil
		}
	}
	return "", fmt.Errorf("anthropic builder ai request failed: empty response content")
}
