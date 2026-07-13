package handlers

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	pathpkg "path"
	"strings"
	"sync/atomic"
	"time"

	sharednlgen "github.com/vllm-project/semantic-router/src/semantic-router/pkg/nlgen"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/jsonunicode"
)

const (
	builderNLMaxResponseBodyBytes int64 = 4 << 20
	builderNLMaxModelNameBytes          = 256
	builderNLMaxEndpointNameBytes       = 256
	builderNLMaxAccessKeyBytes          = 8 << 10
)

var (
	errBuilderNLRequestFailed    = errors.New("builder ai request failed")
	errBuilderNLResponseTooLarge = errors.New("builder ai response exceeded the size limit")
	secureBuilderNLHTTPClient    = newBuilderNLHTTPClient()
)

type builderNLHTTPDoer interface {
	Do(*http.Request) (*http.Response, error)
}

// newBuilderNLHTTPClient permits explicitly configured private vLLM endpoints,
// but does not inherit ambient proxies and never follows redirects. This is
// essential for custom access keys: a redirect must never move a credential or
// generated routing prompt to another origin.
func newBuilderNLHTTPClient() *http.Client {
	dialer := &net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}
	transport := &http.Transport{
		Proxy:                  nil,
		DialContext:            dialer.DialContext,
		ForceAttemptHTTP2:      true,
		MaxIdleConns:           20,
		MaxIdleConnsPerHost:    10,
		IdleConnTimeout:        90 * time.Second,
		TLSHandshakeTimeout:    10 * time.Second,
		ResponseHeaderTimeout:  builderNLMaxTimeout,
		ExpectContinueTimeout:  time.Second,
		MaxResponseHeaderBytes: outboundMaxResponseHeaderBytes,
		TLSClientConfig:        &tls.Config{MinVersion: tls.VersionTLS12},
	}
	return &http.Client{
		Transport: transport,
		Timeout:   builderNLMaxTimeout,
		CheckRedirect: func(_ *http.Request, _ []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
}

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
	connectionMode, err := builderNLConnectionModeOrDefault(req.ConnectionMode)
	if err != nil {
		return "", err
	}
	var customConnection *builderNLConnection
	if connectionMode == builderNLConnectionModeCustom {
		if req.CustomConnection == nil {
			return "", errors.New("custom connection details are missing")
		}
		normalized, err := normalizeBuilderNLConnection(*req.CustomConnection)
		if err != nil {
			return "", err
		}
		customConnection = &normalized
	}
	callCtx, cancel := context.WithTimeout(ctx, runtimeOptions.Timeout)
	defer cancel()

	return runBuilderNLModelCallWithProgress(callCtx, reporter, phase, attempt, runtimeOptions.Timeout, func() (string, error) {
		if connectionMode == builderNLConnectionModeCustom {
			reportBuilderNLProgress(
				reporter,
				phase,
				builderNLProgressInfo,
				fmt.Sprintf("Calling custom %s generation model.", customConnection.ProviderKind),
				attempt,
			)
			return callBuilderNLCustomConnectionMessages(callCtx, *customConnection, chatReq)
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
	parsedBaseURL, err := validateBuilderNLConnection(conn)
	if err != nil {
		return "", err
	}
	modelName := strings.TrimSpace(conn.ModelName)

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
	return callBuilderNLOpenAICompatibleMessagesWithClient(
		secureBuilderNLHTTPClient,
		ctx,
		endpoint,
		modelName,
		accessKey,
		chatReq,
	)
}

func callBuilderNLOpenAICompatibleMessagesWithClient(
	client builderNLHTTPDoer,
	ctx context.Context,
	endpoint string,
	modelName string,
	accessKey string,
	chatReq sharednlgen.ChatCompletionRequest,
) (string, error) {
	if client == nil {
		return "", errBuilderNLRequestFailed
	}
	if _, err := validateBuilderNLEndpoint(endpoint, accessKey); err != nil {
		return "", err
	}
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
		return "", errBuilderNLRequestFailed
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(raw))
	if err != nil {
		return "", errBuilderNLRequestFailed
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")
	if accessKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+accessKey)
	}

	resp, err := client.Do(httpReq)
	if err != nil {
		return "", errBuilderNLRequestFailed
	}
	defer func() { _ = resp.Body.Close() }()
	body, err := readBoundedOutboundBody(resp.Body, builderNLMaxResponseBodyBytes)
	if err != nil {
		if errors.Is(err, errOutboundResponseTooLarge) {
			return "", errBuilderNLResponseTooLarge
		}
		return "", errBuilderNLRequestFailed
	}
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return "", fmt.Errorf("builder ai request failed with HTTP status %d", resp.StatusCode)
	}
	if !jsonunicode.Valid(body) {
		return "", errors.New("builder ai response was invalid")
	}

	var parsed openAIChatResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return "", errors.New("builder ai response was invalid")
	}
	if parsed.Error != nil && strings.TrimSpace(parsed.Error.Message) != "" {
		return "", errors.New("builder ai provider reported an error")
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
	return callBuilderNLAnthropicMessagesWithClient(
		secureBuilderNLHTTPClient,
		ctx,
		endpoint,
		modelName,
		accessKey,
		chatReq,
	)
}

func callBuilderNLAnthropicMessagesWithClient(
	client builderNLHTTPDoer,
	ctx context.Context,
	endpoint string,
	modelName string,
	accessKey string,
	chatReq sharednlgen.ChatCompletionRequest,
) (string, error) {
	if accessKey == "" {
		return "", fmt.Errorf("anthropic custom connection requires an accessKey")
	}
	if client == nil {
		return "", errBuilderNLRequestFailed
	}
	if _, err := validateBuilderNLEndpoint(endpoint, accessKey); err != nil {
		return "", err
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
		return "", errBuilderNLRequestFailed
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(raw))
	if err != nil {
		return "", errBuilderNLRequestFailed
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")
	httpReq.Header.Set("x-api-key", accessKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	resp, err := client.Do(httpReq)
	if err != nil {
		return "", errBuilderNLRequestFailed
	}
	defer func() { _ = resp.Body.Close() }()
	body, err := readBoundedOutboundBody(resp.Body, builderNLMaxResponseBodyBytes)
	if err != nil {
		if errors.Is(err, errOutboundResponseTooLarge) {
			return "", errBuilderNLResponseTooLarge
		}
		return "", errBuilderNLRequestFailed
	}
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return "", fmt.Errorf("anthropic builder ai request failed with HTTP status %d", resp.StatusCode)
	}
	if !jsonunicode.Valid(body) {
		return "", errors.New("anthropic builder ai response was invalid")
	}

	var parsed anthropicResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return "", errors.New("anthropic builder ai response was invalid")
	}
	if parsed.Error != nil && strings.TrimSpace(parsed.Error.Message) != "" {
		return "", errors.New("anthropic builder ai provider reported an error")
	}
	for _, part := range parsed.Content {
		if strings.TrimSpace(part.Text) != "" {
			return strings.TrimSpace(part.Text), nil
		}
	}
	return "", fmt.Errorf("anthropic builder ai request failed: empty response content")
}

func validateBuilderNLConnection(conn builderNLConnection) (*url.URL, error) {
	switch conn.ProviderKind {
	case builderNLProviderVLLM, builderNLProviderOpenAICompatible, builderNLProviderAnthropic:
	default:
		return nil, errors.New("unsupported custom connection providerKind")
	}
	modelName := strings.TrimSpace(conn.ModelName)
	if modelName == "" {
		return nil, errors.New("custom connection modelName is required")
	}
	if len([]byte(conn.ModelName)) > builderNLMaxModelNameBytes || containsUnicodeControl(conn.ModelName) {
		return nil, errors.New("custom connection modelName is too large")
	}
	if len([]byte(conn.EndpointName)) > builderNLMaxEndpointNameBytes || containsUnicodeControl(conn.EndpointName) {
		return nil, errors.New("custom connection endpointName is too large")
	}
	accessKey := strings.TrimSpace(conn.AccessKey)
	if len([]byte(conn.AccessKey)) > builderNLMaxAccessKeyBytes || containsUnicodeControl(conn.AccessKey) {
		return nil, errors.New("custom connection accessKey is invalid")
	}
	parsed, err := normalizeBuilderNLBaseURL(conn.BaseURL, conn.ProviderKind)
	if err != nil {
		return nil, err
	}
	if accessKey != "" && parsed.Scheme != "https" {
		return nil, errors.New("custom connections with an accessKey must use HTTPS")
	}
	return parsed, nil
}

// normalizeBuilderNLConnection validates all user-controlled custom connection
// metadata before it can reach progress events, logs, URLs, or request headers.
func normalizeBuilderNLConnection(conn builderNLConnection) (builderNLConnection, error) {
	parsed, err := validateBuilderNLConnection(conn)
	if err != nil {
		return builderNLConnection{}, err
	}
	conn.ModelName = strings.TrimSpace(conn.ModelName)
	conn.EndpointName = strings.TrimSpace(conn.EndpointName)
	conn.AccessKey = strings.TrimSpace(conn.AccessKey)
	conn.BaseURL = parsed.String()
	return conn, nil
}

func validateBuilderNLEndpoint(raw, accessKey string) (*url.URL, error) {
	if len([]byte(strings.TrimSpace(raw))) > outboundMaxURLBytes {
		return nil, errors.New("builder ai endpoint is invalid")
	}
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || parsed.Opaque != "" || parsed.Host == "" || parsed.Hostname() == "" || parsed.User != nil || parsed.Fragment != "" {
		return nil, errors.New("builder ai endpoint is invalid")
	}
	parsed.Scheme = strings.ToLower(parsed.Scheme)
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return nil, errors.New("builder ai endpoint is invalid")
	}
	if strings.TrimSpace(accessKey) != "" && parsed.Scheme != "https" {
		return nil, errors.New("custom connections with an accessKey must use HTTPS")
	}
	return parsed, nil
}
