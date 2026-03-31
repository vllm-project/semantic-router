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
	"regexp"
	"strings"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type builderNLConnectionMode string

type builderNLProviderKind string

const (
	builderNLConnectionModeDefault builderNLConnectionMode = "default"
	builderNLConnectionModeCustom  builderNLConnectionMode = "custom"

	builderNLProviderVLLM             builderNLProviderKind = "vllm"
	builderNLProviderOpenAICompatible builderNLProviderKind = "openai-compatible"
	builderNLProviderAnthropic        builderNLProviderKind = "anthropic"

	builderNLDefaultModelAlias = "MoM"
)

type BuilderNLGenerateRequest struct {
	Prompt           string                  `json:"prompt"`
	CurrentDSL       string                  `json:"currentDsl,omitempty"`
	ConnectionMode   builderNLConnectionMode `json:"connectionMode"`
	CustomConnection *builderNLConnection    `json:"customConnection,omitempty"`
}

type builderNLConnection struct {
	ProviderKind builderNLProviderKind `json:"providerKind"`
	ModelName    string                `json:"modelName"`
	BaseURL      string                `json:"baseUrl"`
	AccessKey    string                `json:"accessKey,omitempty"`
	EndpointName string                `json:"endpointName,omitempty"`
}

type BuilderNLReview struct {
	Ready    bool     `json:"ready"`
	Summary  string   `json:"summary"`
	Warnings []string `json:"warnings,omitempty"`
	Checks   []string `json:"checks,omitempty"`
}

type BuilderNLGenerateResponse struct {
	DSL                string          `json:"dsl"`
	BaseYAML           string          `json:"baseYaml"`
	Summary            string          `json:"summary"`
	SuggestedTestQuery string          `json:"suggestedTestQuery,omitempty"`
	Review             BuilderNLReview `json:"review"`
}

type builderNLLLMOutput struct {
	DSL                string
	Summary            string
	SuggestedTestQuery string
}

type anthropicMessageRequest struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type anthropicRequest struct {
	Model     string                    `json:"model"`
	MaxTokens int                       `json:"max_tokens"`
	System    string                    `json:"system,omitempty"`
	Messages  []anthropicMessageRequest `json:"messages"`
}

type anthropicResponse struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

func generateBuilderNLDraft(
	ctx context.Context,
	configPath string,
	envoyURL string,
	req BuilderNLGenerateRequest,
) (BuilderNLGenerateResponse, error) {
	prompt := strings.TrimSpace(req.Prompt)
	if prompt == "" {
		return BuilderNLGenerateResponse{}, fmt.Errorf("routing request prompt is required")
	}
	if len(prompt) > 12000 {
		return BuilderNLGenerateResponse{}, fmt.Errorf("routing request is too large")
	}

	connectionMode := req.ConnectionMode
	if connectionMode == "" {
		connectionMode = builderNLConnectionModeDefault
	}
	if connectionMode != builderNLConnectionModeDefault && connectionMode != builderNLConnectionModeCustom {
		return BuilderNLGenerateResponse{}, fmt.Errorf("unsupported connectionMode %q", connectionMode)
	}

	baseConfig, err := readBuilderNLBaseConfig(configPath)
	if err != nil {
		return BuilderNLGenerateResponse{}, err
	}
	ensureBuilderNLGlobalDefaults(baseConfig)
	if connectionMode == builderNLConnectionModeCustom {
		if req.CustomConnection == nil {
			return BuilderNLGenerateResponse{}, fmt.Errorf("customConnection is required when connectionMode is custom")
		}
		if applyErr := applyBuilderNLCustomConnection(baseConfig, *req.CustomConnection); applyErr != nil {
			return BuilderNLGenerateResponse{}, applyErr
		}
	}

	baseYAMLBytes, err := marshalYAMLBytes(baseConfig)
	if err != nil {
		return BuilderNLGenerateResponse{}, fmt.Errorf("failed to build deploy base yaml: %w", err)
	}

	targetModelName := builderNLTargetModelName(req)
	systemPrompt, userPrompt := buildBuilderNLGenerationPrompts(prompt, strings.TrimSpace(req.CurrentDSL), targetModelName, connectionMode)
	content, err := callBuilderNLModel(ctx, envoyURL, req, systemPrompt, userPrompt)
	if err != nil {
		return BuilderNLGenerateResponse{}, err
	}

	generated, err := parseBuilderNLGenerationOutput(content)
	if err != nil {
		return BuilderNLGenerateResponse{}, fmt.Errorf("failed to parse generated DSL draft: %w", err)
	}

	var review BuilderNLReview
	reviewPrompt := buildBuilderNLReviewPrompt(prompt, targetModelName, generated.DSL)
	reviewContent, reviewErr := callBuilderNLModel(ctx, envoyURL, req, builderNLReviewSystemPrompt(), reviewPrompt)
	if reviewErr != nil {
		review = BuilderNLReview{
			Ready:    false,
			Summary:  "AI double-check was unavailable.",
			Warnings: []string{reviewErr.Error()},
		}
	} else {
		parsedReview, parseErr := parseBuilderNLReviewOutput(reviewContent)
		if parseErr != nil {
			review = BuilderNLReview{
				Ready:    false,
				Summary:  "AI double-check returned an unreadable payload.",
				Warnings: []string{parseErr.Error()},
			}
		} else {
			review = parsedReview
		}
	}

	return BuilderNLGenerateResponse{
		DSL:                generated.DSL,
		BaseYAML:           string(baseYAMLBytes),
		Summary:            generated.Summary,
		SuggestedTestQuery: generated.SuggestedTestQuery,
		Review:             review,
	}, nil
}

func readBuilderNLBaseConfig(configPath string) (*routerconfig.CanonicalConfig, error) {
	cfg, err := readCanonicalConfigFile(configPath)
	if err == nil {
		return cfg, nil
	}
	if strings.Contains(strings.ToLower(err.Error()), "no such file") {
		return &routerconfig.CanonicalConfig{}, nil
	}
	return nil, fmt.Errorf("failed to read builder deploy base: %w", err)
}

func ensureBuilderNLGlobalDefaults(cfg *routerconfig.CanonicalConfig) {
	if cfg == nil {
		return
	}
	defaults := routerconfig.DefaultCanonicalGlobal()
	if cfg.Global == nil {
		cfg.Global = &defaults
		return
	}
	if strings.TrimSpace(cfg.Global.Router.AutoModelName) == "" {
		cfg.Global.Router.AutoModelName = defaults.Router.AutoModelName
	}
}

func applyBuilderNLCustomConnection(
	cfg *routerconfig.CanonicalConfig,
	conn builderNLConnection,
) error {
	modelName := strings.TrimSpace(conn.ModelName)
	if modelName == "" {
		return fmt.Errorf("custom connection modelName is required")
	}

	normalizedBaseURL, err := normalizeBuilderNLBaseURL(conn.BaseURL, conn.ProviderKind)
	if err != nil {
		return err
	}
	endpointName := strings.TrimSpace(conn.EndpointName)
	if endpointName == "" {
		endpointName = slugifyBuilderNLName(modelName)
		if endpointName == "" {
			endpointName = "nl-custom"
		}
	}

	providerModel := routerconfig.CanonicalProviderModel{
		Name:            modelName,
		ProviderModelID: modelName,
	}
	backendRef := routerconfig.CanonicalBackendRef{
		Name:   endpointName,
		Weight: 100,
	}
	if accessKey := strings.TrimSpace(conn.AccessKey); accessKey != "" {
		backendRef.APIKey = accessKey
	}

	switch conn.ProviderKind {
	case builderNLProviderVLLM:
		backendRef.Protocol = normalizedBaseURL.Scheme
		backendRef.Endpoint = builderNLEndpointFromURL(normalizedBaseURL)
	case builderNLProviderOpenAICompatible:
		backendRef.Protocol = normalizedBaseURL.Scheme
		backendRef.BaseURL = strings.TrimRight(normalizedBaseURL.String(), "/")
		backendRef.Provider = "openai"
	case builderNLProviderAnthropic:
		backendRef.Protocol = normalizedBaseURL.Scheme
		backendRef.BaseURL = strings.TrimRight(normalizedBaseURL.String(), "/")
		backendRef.Provider = "anthropic"
		providerModel.APIFormat = "anthropic"
	default:
		return fmt.Errorf("unsupported custom connection providerKind %q", conn.ProviderKind)
	}

	providerModel.BackendRefs = []routerconfig.CanonicalBackendRef{backendRef}
	cfg.Providers.Models = upsertBuilderNLProviderModel(cfg.Providers.Models, providerModel)
	if strings.TrimSpace(cfg.Providers.Defaults.DefaultModel) == "" {
		cfg.Providers.Defaults.DefaultModel = modelName
	}
	return nil
}

func upsertBuilderNLProviderModel(
	models []routerconfig.CanonicalProviderModel,
	candidate routerconfig.CanonicalProviderModel,
) []routerconfig.CanonicalProviderModel {
	updated := false
	for index := range models {
		if strings.EqualFold(strings.TrimSpace(models[index].Name), strings.TrimSpace(candidate.Name)) {
			models[index] = candidate
			updated = true
			break
		}
	}
	if updated {
		return models
	}
	return append(models, candidate)
}

func normalizeBuilderNLBaseURL(raw string, providerKind builderNLProviderKind) (*url.URL, error) {
	trimmed := strings.TrimSpace(strings.TrimRight(raw, "/"))
	if trimmed == "" {
		return nil, fmt.Errorf("custom connection baseUrl is required")
	}
	if !strings.Contains(trimmed, "://") {
		trimmed = inferBuilderNLProtocol(trimmed, providerKind) + "://" + trimmed
	}
	parsed, err := url.Parse(trimmed)
	if err != nil {
		return nil, fmt.Errorf("invalid custom connection baseUrl: %w", err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return nil, fmt.Errorf("custom connection baseUrl must use http or https")
	}
	if parsed.Host == "" {
		return nil, fmt.Errorf("custom connection baseUrl must include a host")
	}
	return parsed, nil
}

func inferBuilderNLProtocol(raw string, providerKind builderNLProviderKind) string {
	if providerKind == builderNLProviderAnthropic {
		return "https"
	}
	trimmed := strings.TrimSpace(raw)
	for _, prefix := range []string{"localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"} {
		if strings.HasPrefix(trimmed, prefix) {
			return "http"
		}
	}
	if strings.Contains(trimmed, ":80") {
		return "http"
	}
	return "https"
}

func builderNLEndpointFromURL(parsed *url.URL) string {
	endpoint := parsed.Host
	path := strings.TrimSpace(strings.TrimRight(parsed.Path, "/"))
	if path != "" && path != "/" {
		endpoint += path
	}
	return endpoint
}

func slugifyBuilderNLName(value string) string {
	value = strings.TrimSpace(strings.ToLower(value))
	value = regexp.MustCompile(`[^a-z0-9]+`).ReplaceAllString(value, "-")
	return strings.Trim(value, "-")
}

func builderNLTargetModelName(req BuilderNLGenerateRequest) string {
	if req.ConnectionMode == builderNLConnectionModeCustom && req.CustomConnection != nil {
		if name := strings.TrimSpace(req.CustomConnection.ModelName); name != "" {
			return name
		}
	}
	return builderNLDefaultModelAlias
}

func buildBuilderNLGenerationPrompts(
	request string,
	currentDSL string,
	targetModelName string,
	connectionMode builderNLConnectionMode,
) (string, string) {
	systemPrompt := strings.Join([]string{
		"You generate vLLM Semantic Router Builder DSL.",
		"Return ONLY valid JSON with keys dsl, summary, suggestedTestQuery.",
		"The dsl value must be plain DSL source without markdown fences.",
		"Use Builder-compatible syntax only: MODEL, SIGNAL, ROUTE, PLUGIN, PRIORITY, WHEN, MODEL refs, ALGORITHM, and PLUGIN refs.",
		"Connection settings live outside the DSL, so never emit YAML or providers/global blocks.",
		"Always include at least one MODEL declaration and one ROUTE declaration.",
		"Keep summaries concise and action-oriented.",
	}, "\n")

	contextBlock := "No current DSL is available. Build a fresh draft."
	if strings.TrimSpace(currentDSL) != "" {
		contextBlock = "Current DSL context (preserve unrelated valid declarations unless the request clearly replaces them):\n" + currentDSL
	}

	userPrompt := strings.Join([]string{
		fmt.Sprintf("Target model name for route references: %s", targetModelName),
		fmt.Sprintf("Connection mode: %s", connectionMode),
		"Default auto-model alias: MoM.",
		"Use short descriptions, add a sensible fallback route when the request implies one, and prefer deterministic priorities.",
		"Route request:",
		request,
		"",
		contextBlock,
		"",
		"JSON response shape:",
		`{"dsl":"...","summary":"...","suggestedTestQuery":"..."}`,
	}, "\n")

	return systemPrompt, userPrompt
}

func buildBuilderNLReviewPrompt(request string, targetModelName string, dsl string) string {
	return strings.Join([]string{
		"Review this vLLM Semantic Router Builder DSL against the user's request.",
		fmt.Sprintf("Expected primary model name or alias: %s", targetModelName),
		"Return ONLY valid JSON with keys ready, summary, warnings, checks.",
		"Warnings should describe mismatches, risky assumptions, or missing fallbacks.",
		"Checks should list what you verified successfully.",
		"Do not rewrite the DSL.",
		"User request:",
		request,
		"",
		"DSL:",
		dsl,
		"",
		`JSON response shape: {"ready":true,"summary":"...","warnings":["..."],"checks":["..."]}`,
	}, "\n")
}

func builderNLReviewSystemPrompt() string {
	return strings.Join([]string{
		"You review vLLM Semantic Router Builder DSL for correctness.",
		"Return ONLY valid JSON.",
		"Be strict about missing routes, contradictory conditions, and wrong model references.",
	}, "\n")
}

func callBuilderNLModel(
	ctx context.Context,
	envoyURL string,
	req BuilderNLGenerateRequest,
	systemPrompt string,
	userPrompt string,
) (string, error) {
	if req.ConnectionMode == builderNLConnectionModeCustom {
		if req.CustomConnection == nil {
			return "", fmt.Errorf("custom connection details are missing")
		}
		return callBuilderNLCustomConnection(ctx, *req.CustomConnection, systemPrompt, userPrompt)
	}
	if strings.TrimSpace(envoyURL) == "" {
		return "", fmt.Errorf("default Builder AI connection is unavailable because Envoy is not configured")
	}
	return callBuilderNLOpenAICompatible(
		ctx,
		strings.TrimRight(envoyURL, "/")+"/v1/chat/completions",
		builderNLDefaultModelAlias,
		"",
		systemPrompt,
		userPrompt,
	)
}

func callBuilderNLCustomConnection(
	ctx context.Context,
	conn builderNLConnection,
	systemPrompt string,
	userPrompt string,
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
		return callBuilderNLOpenAICompatible(
			ctx,
			resolveBuilderNLOpenAIURL(parsedBaseURL),
			modelName,
			strings.TrimSpace(conn.AccessKey),
			systemPrompt,
			userPrompt,
		)
	case builderNLProviderAnthropic:
		return callBuilderNLAnthropic(
			ctx,
			resolveBuilderNLAnthropicURL(parsedBaseURL),
			modelName,
			strings.TrimSpace(conn.AccessKey),
			systemPrompt,
			userPrompt,
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

func callBuilderNLOpenAICompatible(
	ctx context.Context,
	endpoint string,
	modelName string,
	accessKey string,
	systemPrompt string,
	userPrompt string,
) (string, error) {
	payload := openAIChatRequest{
		Model: modelName,
		Messages: []openAIChatMessage{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userPrompt},
		},
		Stream: false,
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
	content := strings.TrimSpace(parsed.Choices[0].Message.Content)
	if content == "" {
		return "", fmt.Errorf("builder ai request failed: empty response content")
	}
	return content, nil
}

func callBuilderNLAnthropic(
	ctx context.Context,
	endpoint string,
	modelName string,
	accessKey string,
	systemPrompt string,
	userPrompt string,
) (string, error) {
	if accessKey == "" {
		return "", fmt.Errorf("anthropic custom connection requires an accessKey")
	}
	payload := anthropicRequest{
		Model:     modelName,
		MaxTokens: 4096,
		System:    systemPrompt,
		Messages: []anthropicMessageRequest{{
			Role:    "user",
			Content: userPrompt,
		}},
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

func parseBuilderNLGenerationOutput(raw string) (builderNLLLMOutput, error) {
	payload, err := parseBuilderNLJSONObject(raw)
	if err != nil {
		dsl := sanitizeBuilderNLDsl(extractMarkdownFence(raw, "dsl"))
		if dsl == "" {
			dsl = sanitizeBuilderNLDsl(extractMarkdownFence(raw, ""))
		}
		if dsl == "" {
			return builderNLLLMOutput{}, err
		}
		return builderNLLLMOutput{
			DSL:     dsl,
			Summary: "Generated DSL draft from natural-language request.",
		}, nil
	}

	dsl := sanitizeBuilderNLDsl(stringFromPayload(payload, "dsl", "dsl_source", "code"))
	if dsl == "" {
		dsl = sanitizeBuilderNLDsl(extractMarkdownFence(raw, "dsl"))
	}
	if dsl == "" {
		return builderNLLLMOutput{}, fmt.Errorf("generated payload did not include a DSL draft")
	}

	summary := stringFromPayload(payload, "summary", "notes")
	if strings.TrimSpace(summary) == "" {
		summary = "Generated DSL draft from natural-language request."
	}

	return builderNLLLMOutput{
		DSL:                dsl,
		Summary:            strings.TrimSpace(summary),
		SuggestedTestQuery: strings.TrimSpace(stringFromPayload(payload, "suggestedTestQuery", "suggested_test_query", "exampleQuery")),
	}, nil
}

func parseBuilderNLReviewOutput(raw string) (BuilderNLReview, error) {
	payload, err := parseBuilderNLJSONObject(raw)
	if err != nil {
		return BuilderNLReview{}, err
	}
	summary := strings.TrimSpace(stringFromPayload(payload, "summary"))
	if summary == "" {
		summary = "AI review completed."
	}
	warnings := stringSliceFromPayload(payload, "warnings")
	checks := stringSliceFromPayload(payload, "checks")
	ready := boolFromPayload(payload, "ready")
	if !ready && len(warnings) == 0 {
		ready = true
	}
	return BuilderNLReview{
		Ready:    ready,
		Summary:  summary,
		Warnings: warnings,
		Checks:   checks,
	}, nil
}

func parseBuilderNLJSONObject(raw string) (map[string]any, error) {
	candidates := []string{strings.TrimSpace(raw)}
	if fencedJSON := extractMarkdownFence(raw, "json"); fencedJSON != "" {
		candidates = append(candidates, fencedJSON)
	}
	if objectJSON := extractFirstJSONObject(raw); objectJSON != "" {
		candidates = append(candidates, objectJSON)
	}

	for _, candidate := range candidates {
		if strings.TrimSpace(candidate) == "" {
			continue
		}
		var parsed map[string]any
		if err := json.Unmarshal([]byte(candidate), &parsed); err == nil {
			return parsed, nil
		}
	}
	return nil, fmt.Errorf("no valid JSON object found in model response")
}

func extractMarkdownFence(raw string, language string) string {
	pattern := "```"
	if language != "" {
		pattern += language
	}
	re := regexp.MustCompile("(?s)" + regexp.QuoteMeta(pattern) + "\\s*(.*?)```")
	matches := re.FindStringSubmatch(raw)
	if len(matches) == 2 {
		return strings.TrimSpace(matches[1])
	}
	if language == "" {
		fallback := regexp.MustCompile("(?s)```[a-zA-Z-]*\\s*(.*?)```")
		matches = fallback.FindStringSubmatch(raw)
		if len(matches) == 2 {
			return strings.TrimSpace(matches[1])
		}
	}
	return ""
}

func extractFirstJSONObject(raw string) string {
	start := -1
	depth := 0
	inString := false
	escaped := false
	for index, r := range raw {
		switch {
		case escaped:
			escaped = false
		case r == '\\':
			escaped = true
		case r == '"':
			inString = !inString
		case inString:
			continue
		case r == '{':
			if depth == 0 {
				start = index
			}
			depth++
		case r == '}':
			if depth == 0 {
				continue
			}
			depth--
			if depth == 0 && start >= 0 {
				return strings.TrimSpace(raw[start : index+1])
			}
		}
	}
	return ""
}

func sanitizeBuilderNLDsl(raw string) string {
	trimmed := strings.TrimSpace(raw)
	trimmed = strings.TrimPrefix(trimmed, "```dsl")
	trimmed = strings.TrimPrefix(trimmed, "```")
	trimmed = strings.TrimSuffix(trimmed, "```")
	return strings.TrimSpace(trimmed)
}

func stringFromPayload(payload map[string]any, keys ...string) string {
	for _, key := range keys {
		if value, ok := payload[key]; ok {
			if text, ok := value.(string); ok {
				return text
			}
		}
	}
	return ""
}

func stringSliceFromPayload(payload map[string]any, key string) []string {
	value, ok := payload[key]
	if !ok {
		return nil
	}
	entries, ok := value.([]any)
	if !ok {
		return nil
	}
	result := make([]string, 0, len(entries))
	for _, entry := range entries {
		if text, ok := entry.(string); ok {
			text = strings.TrimSpace(text)
			if text != "" {
				result = append(result, text)
			}
		}
	}
	return result
}

func boolFromPayload(payload map[string]any, key string) bool {
	value, ok := payload[key]
	if !ok {
		return false
	}
	flag, ok := value.(bool)
	return ok && flag
}
