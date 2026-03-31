package handlers

import (
	"context"
	"fmt"
	"net/url"
	"strings"

	sharednlgen "github.com/vllm-project/semantic-router/src/semantic-router/pkg/nlgen"
)

type builderNLConnectionMode string

type builderNLProviderKind string

const (
	builderNLConnectionModeDefault builderNLConnectionMode = "default"
	builderNLConnectionModeCustom  builderNLConnectionMode = "custom"

	builderNLProviderVLLM             builderNLProviderKind = "vllm"
	builderNLProviderOpenAICompatible builderNLProviderKind = "openai-compatible"
	builderNLProviderAnthropic        builderNLProviderKind = "anthropic"

	builderNLFallbackModelAlias = "MoM"
)

type BuilderNLGenerateRequest struct {
	Prompt           string                  `json:"prompt"`
	CurrentDSL       string                  `json:"currentDsl,omitempty"`
	ConnectionMode   builderNLConnectionMode `json:"connectionMode"`
	CustomConnection *builderNLConnection    `json:"customConnection,omitempty"`
	Temperature      *float64                `json:"temperature,omitempty"`
	MaxRetries       *int                    `json:"maxRetries,omitempty"`
	TimeoutSeconds   *int                    `json:"timeoutSeconds,omitempty"`
}

type BuilderNLVerifyRequest struct {
	ConnectionMode   builderNLConnectionMode `json:"connectionMode"`
	CustomConnection *builderNLConnection    `json:"customConnection,omitempty"`
	TimeoutSeconds   *int                    `json:"timeoutSeconds,omitempty"`
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

type builderNLQuickFix struct {
	Description string `json:"description"`
	NewText     string `json:"newText"`
}

type BuilderNLDiagnostic struct {
	Level   string              `json:"level"`
	Message string              `json:"message"`
	Line    int                 `json:"line"`
	Column  int                 `json:"column"`
	Fixes   []builderNLQuickFix `json:"fixes,omitempty"`
}

type BuilderNLValidation struct {
	Ready        bool                  `json:"ready"`
	Diagnostics  []BuilderNLDiagnostic `json:"diagnostics,omitempty"`
	ErrorCount   int                   `json:"errorCount"`
	CompileError string                `json:"compileError,omitempty"`
	draftDSL     string
}

type BuilderNLVerifyResponse struct {
	Ready           bool                    `json:"ready"`
	Summary         string                  `json:"summary"`
	ConnectionMode  builderNLConnectionMode `json:"connectionMode"`
	ProviderKind    builderNLProviderKind   `json:"providerKind,omitempty"`
	ModelName       string                  `json:"modelName,omitempty"`
	TargetModelName string                  `json:"targetModelName,omitempty"`
	Endpoint        string                  `json:"endpoint,omitempty"`
}

type BuilderNLGenerateResponse struct {
	DSL                string              `json:"dsl"`
	BaseYAML           string              `json:"baseYaml"`
	Summary            string              `json:"summary"`
	SuggestedTestQuery string              `json:"suggestedTestQuery,omitempty"`
	Review             BuilderNLReview     `json:"review"`
	Validation         BuilderNLValidation `json:"validation"`
}

type BuilderNLProgressEvent struct {
	Phase          string `json:"phase"`
	Level          string `json:"level"`
	Message        string `json:"message"`
	Attempt        int    `json:"attempt,omitempty"`
	Kind           string `json:"kind,omitempty"`
	ElapsedSeconds int    `json:"elapsedSeconds,omitempty"`
	Timestamp      int64  `json:"timestamp"`
}

type builderNLProgressReporter func(BuilderNLProgressEvent)

type builderNLLLMOutput struct {
	DSL                string
	Summary            string
	SuggestedTestQuery string
	Warnings           []string
}

const (
	builderNLProgressInfo    = "info"
	builderNLProgressSuccess = "success"
	builderNLProgressWarning = "warning"
	builderNLProgressError   = "error"
)

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
	return generateBuilderNLDraftWithProgress(ctx, configPath, envoyURL, req, nil)
}

func generateBuilderNLDraftWithProgress(
	ctx context.Context,
	configPath string,
	envoyURL string,
	req BuilderNLGenerateRequest,
	reporter builderNLProgressReporter,
) (BuilderNLGenerateResponse, error) {
	runtimeOptions := resolveBuilderNLRuntimeOptions(req)
	prompt := strings.TrimSpace(req.Prompt)
	if prompt == "" {
		return BuilderNLGenerateResponse{}, fmt.Errorf("routing request prompt is required")
	}
	if len(prompt) > 12000 {
		return BuilderNLGenerateResponse{}, fmt.Errorf("routing request is too large")
	}
	reportBuilderNLProgress(reporter, "request", builderNLProgressInfo, "Accepted Builder NL request and prepared staged draft generation.", 0)

	connectionMode := req.ConnectionMode
	if connectionMode == "" {
		connectionMode = builderNLConnectionModeDefault
	}
	if connectionMode != builderNLConnectionModeDefault && connectionMode != builderNLConnectionModeCustom {
		return BuilderNLGenerateResponse{}, fmt.Errorf("unsupported connectionMode %q", connectionMode)
	}
	reportBuilderNLProgress(reporter, "context", builderNLProgressInfo, fmt.Sprintf("Using %s generation connection.", connectionMode), 0)

	generationContext, err := prepareBuilderNLGenerationContext(
		configPath,
		connectionMode,
		runtimeOptions,
		reporter,
	)
	if err != nil {
		return BuilderNLGenerateResponse{}, err
	}

	generated, validation, err := generateValidatedBuilderNLDraft(
		ctx,
		envoyURL,
		req,
		prompt,
		strings.TrimSpace(req.CurrentDSL),
		generationContext.targetModelName,
		generationContext.knownModelNames,
		runtimeOptions,
		reporter,
	)
	if err != nil {
		return BuilderNLGenerateResponse{}, err
	}

	review := reviewValidatedBuilderNLDraft(generationContext.targetModelName, validation, generated.Warnings, reporter)
	if validation.Ready {
		reportBuilderNLProgress(reporter, "complete", builderNLProgressSuccess, "Staged draft is ready for Builder review and apply.", 0)
	} else {
		reportBuilderNLProgress(reporter, "complete", builderNLProgressWarning, "Generated a staged draft, but repository validation still found issues to repair manually.", 0)
	}

	return BuilderNLGenerateResponse{
		DSL:                generated.DSL,
		BaseYAML:           generationContext.baseYAML,
		Summary:            generated.Summary,
		SuggestedTestQuery: generated.SuggestedTestQuery,
		Review:             review,
		Validation:         validation,
	}, nil
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

func verifyBuilderNLConnection(
	ctx context.Context,
	configPath string,
	envoyURL string,
	req BuilderNLVerifyRequest,
) (BuilderNLVerifyResponse, error) {
	connectionMode, err := builderNLConnectionModeOrDefault(req.ConnectionMode)
	if err != nil {
		return BuilderNLVerifyResponse{}, err
	}

	generateReq := BuilderNLGenerateRequest{
		ConnectionMode:   connectionMode,
		CustomConnection: req.CustomConnection,
		TimeoutSeconds:   req.TimeoutSeconds,
	}
	targetModelName := ""
	if strings.TrimSpace(configPath) != "" {
		baseConfig, configErr := readBuilderNLBaseConfig(configPath)
		if configErr != nil {
			return BuilderNLVerifyResponse{}, configErr
		}
		targetModelName = builderNLDraftTargetModelName(baseConfig)
	}

	content, err := callBuilderNLMessages(
		ctx,
		envoyURL,
		generateReq,
		resolveBuilderNLVerifyRuntimeOptions(req),
		sharednlgen.ChatCompletionRequest{
			Messages: []sharednlgen.ChatMessage{
				{Role: "system", Content: "You verify that a Builder AI connection is reachable."},
				{Role: "user", Content: "Reply with a short confirmation that the connection is working."},
			},
			MaxTokens: builderNLDefaultMaxTokens,
		},
		nil,
		"",
		0,
	)
	if err != nil {
		return BuilderNLVerifyResponse{}, err
	}

	response := BuilderNLVerifyResponse{
		Ready:          true,
		Summary:        strings.TrimSpace(content),
		ConnectionMode: connectionMode,
	}
	if response.Summary == "" {
		response.Summary = "Connection verified."
	}

	if connectionMode == builderNLConnectionModeCustom && req.CustomConnection != nil {
		response.ProviderKind = req.CustomConnection.ProviderKind
		response.ModelName = strings.TrimSpace(req.CustomConnection.ModelName)
		response.TargetModelName = strings.TrimSpace(targetModelName)
		response.Endpoint = builderNLConnectionEndpoint(envoyURL, generateReq)
		return response, nil
	}

	response.ProviderKind = builderNLProviderOpenAICompatible
	response.ModelName = builderNLFallbackModelAlias
	response.TargetModelName = strings.TrimSpace(targetModelName)
	if response.TargetModelName == "" {
		response.TargetModelName = builderNLFallbackModelAlias
	}
	response.Endpoint = builderNLConnectionEndpoint(envoyURL, generateReq)
	return response, nil
}

func builderNLConnectionEndpoint(envoyURL string, req BuilderNLGenerateRequest) string {
	if req.ConnectionMode == builderNLConnectionModeCustom && req.CustomConnection != nil {
		parsedBaseURL, err := normalizeBuilderNLBaseURL(req.CustomConnection.BaseURL, req.CustomConnection.ProviderKind)
		if err != nil {
			return strings.TrimSpace(req.CustomConnection.BaseURL)
		}
		switch req.CustomConnection.ProviderKind {
		case builderNLProviderAnthropic:
			return resolveBuilderNLAnthropicURL(parsedBaseURL)
		default:
			return resolveBuilderNLOpenAIURL(parsedBaseURL)
		}
	}
	if strings.TrimSpace(envoyURL) == "" {
		return ""
	}
	return strings.TrimRight(envoyURL, "/") + "/v1/chat/completions"
}
