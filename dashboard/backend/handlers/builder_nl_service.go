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
	"time"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl"
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
	builderNLRepairMaxAttempts = 2
)

type BuilderNLGenerateRequest struct {
	Prompt           string                  `json:"prompt"`
	CurrentDSL       string                  `json:"currentDsl,omitempty"`
	ConnectionMode   builderNLConnectionMode `json:"connectionMode"`
	CustomConnection *builderNLConnection    `json:"customConnection,omitempty"`
}

type BuilderNLVerifyRequest struct {
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
	Ready          bool                    `json:"ready"`
	Summary        string                  `json:"summary"`
	ConnectionMode builderNLConnectionMode `json:"connectionMode"`
	ProviderKind   builderNLProviderKind   `json:"providerKind,omitempty"`
	ModelName      string                  `json:"modelName,omitempty"`
	Endpoint       string                  `json:"endpoint,omitempty"`
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
	Phase     string `json:"phase"`
	Level     string `json:"level"`
	Message   string `json:"message"`
	Attempt   int    `json:"attempt,omitempty"`
	Timestamp int64  `json:"timestamp"`
}

type builderNLProgressReporter func(BuilderNLProgressEvent)

type builderNLLLMOutput struct {
	DSL                string
	Summary            string
	SuggestedTestQuery string
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

	baseConfig, err := readBuilderNLBaseConfig(configPath)
	if err != nil {
		return BuilderNLGenerateResponse{}, err
	}

	baseYAMLBytes, err := marshalYAMLBytes(baseConfig)
	if err != nil {
		return BuilderNLGenerateResponse{}, fmt.Errorf("failed to build deploy base yaml: %w", err)
	}
	reportBuilderNLProgress(reporter, "context", builderNLProgressSuccess, "Loaded deploy base YAML without mutating runtime provider settings.", 0)

	targetModelName := builderNLDraftTargetModelName()
	generated, validation, err := generateValidatedBuilderNLDraft(
		ctx,
		envoyURL,
		req,
		prompt,
		strings.TrimSpace(req.CurrentDSL),
		targetModelName,
		reporter,
	)
	if err != nil {
		return BuilderNLGenerateResponse{}, err
	}

	review := reviewValidatedBuilderNLDraft(ctx, envoyURL, req, prompt, targetModelName, generated.DSL, validation, reporter)

	if validation.Ready {
		reportBuilderNLProgress(reporter, "complete", builderNLProgressSuccess, "Staged draft is ready for Builder review and apply.", 0)
	} else {
		reportBuilderNLProgress(reporter, "complete", builderNLProgressWarning, "Generated a staged draft, but repository validation still found issues to repair manually.", 0)
	}

	return BuilderNLGenerateResponse{
		DSL:                generated.DSL,
		BaseYAML:           string(baseYAMLBytes),
		Summary:            generated.Summary,
		SuggestedTestQuery: generated.SuggestedTestQuery,
		Review:             review,
		Validation:         validation,
	}, nil
}

func reportBuilderNLProgress(
	reporter builderNLProgressReporter,
	phase string,
	level string,
	message string,
	attempt int,
) {
	if reporter == nil {
		return
	}

	reporter(BuilderNLProgressEvent{
		Phase:     phase,
		Level:     level,
		Message:   strings.TrimSpace(message),
		Attempt:   attempt,
		Timestamp: time.Now().UnixMilli(),
	})
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

func builderNLDraftTargetModelName() string {
	return builderNLDefaultModelAlias
}

func builderNLConnectionModeOrDefault(mode builderNLConnectionMode) (builderNLConnectionMode, error) {
	if mode == "" {
		return builderNLConnectionModeDefault, nil
	}
	if mode != builderNLConnectionModeDefault && mode != builderNLConnectionModeCustom {
		return "", fmt.Errorf("unsupported connectionMode %q", mode)
	}
	return mode, nil
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
		fmt.Sprintf("Prefer the Builder auto-model alias %q for new fallback or default routes unless the current DSL context already defines a better existing model reference.", builderNLDefaultModelAlias),
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
		"Do not invent provider wiring or deploy-time endpoint config inside the DSL.",
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

func buildBuilderNLRepairPrompts(
	request string,
	currentDSL string,
	targetModelName string,
	connectionMode builderNLConnectionMode,
	badDSL string,
	validation BuilderNLValidation,
) (string, string) {
	systemPrompt := strings.Join([]string{
		"You repair vLLM Semantic Router Builder DSL using repository validation errors.",
		"Return ONLY valid JSON with keys dsl, summary, suggestedTestQuery.",
		"The dsl value must be plain DSL source without markdown fences.",
		"Fix the validation and compile problems without adding unrelated features.",
	}, "\n")

	userPrompt := strings.Join([]string{
		fmt.Sprintf("Target model name for route references: %s", targetModelName),
		fmt.Sprintf("Connection mode: %s", connectionMode),
		"Original route request:",
		request,
		"",
		"Current Builder DSL context:",
		emptyBuilderNLContext(currentDSL),
		"",
		"Previous invalid DSL draft:",
		badDSL,
		"",
		"Repository validation findings:",
		builderNLValidationSummary(validation),
		"",
		`JSON response shape: {"dsl":"...","summary":"...","suggestedTestQuery":"..."}`,
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

func emptyBuilderNLContext(currentDSL string) string {
	if strings.TrimSpace(currentDSL) == "" {
		return "No current DSL is available."
	}
	return currentDSL
}

func generateValidatedBuilderNLDraft(
	ctx context.Context,
	envoyURL string,
	req BuilderNLGenerateRequest,
	prompt string,
	currentDSL string,
	targetModelName string,
	reporter builderNLProgressReporter,
) (builderNLLLMOutput, BuilderNLValidation, error) {
	connectionMode, err := builderNLConnectionModeOrDefault(req.ConnectionMode)
	if err != nil {
		return builderNLLLMOutput{}, BuilderNLValidation{}, err
	}

	systemPrompt, userPrompt := buildBuilderNLGenerationPrompts(prompt, currentDSL, targetModelName, connectionMode)
	var lastGenerated builderNLLLMOutput
	var lastValidation BuilderNLValidation

	for attempt := 0; attempt <= builderNLRepairMaxAttempts; attempt++ {
		attemptNumber := attempt + 1
		if attempt == 0 {
			reportBuilderNLProgress(reporter, "generate", builderNLProgressInfo, fmt.Sprintf("Generation attempt %d/%d: requesting a draft from the model.", attemptNumber, builderNLRepairMaxAttempts+1), attemptNumber)
		} else {
			reportBuilderNLProgress(reporter, "repair", builderNLProgressInfo, fmt.Sprintf("Repair attempt %d/%d: sending validation findings back to the model.", attemptNumber, builderNLRepairMaxAttempts+1), attemptNumber)
		}

		content, callErr := callBuilderNLModel(ctx, envoyURL, req, systemPrompt, userPrompt, reporter, "model_call", attemptNumber)
		if callErr != nil {
			return builderNLLLMOutput{}, BuilderNLValidation{}, callErr
		}

		generated, parseErr := parseBuilderNLGenerationOutput(content)
		if parseErr != nil {
			reportBuilderNLProgress(reporter, "parse", builderNLProgressError, fmt.Sprintf("Model output could not be parsed as a Builder draft: %s", parseErr), attemptNumber)
			return builderNLLLMOutput{}, BuilderNLValidation{}, fmt.Errorf("failed to parse generated DSL draft: %w", parseErr)
		}
		reportBuilderNLProgress(reporter, "parse", builderNLProgressSuccess, "Parsed model output into a candidate DSL draft.", attemptNumber)

		validation := validateBuilderNLDraft(generated.DSL)
		lastGenerated = generated
		lastValidation = validation
		if validation.Ready {
			reportBuilderNLProgress(reporter, "validate", builderNLProgressSuccess, "Repository validation passed for the staged draft.", attemptNumber)
			if formatted := strings.TrimSpace(validation.formattedDSL()); formatted != "" {
				lastGenerated.DSL = formatted
			}
			return lastGenerated, lastValidation, nil
		}

		reportBuilderNLProgress(
			reporter,
			"validate",
			builderNLProgressWarning,
			builderNLValidationProgressMessage(validation, attemptNumber),
			attemptNumber,
		)

		if attempt == builderNLRepairMaxAttempts {
			reportBuilderNLProgress(reporter, "repair", builderNLProgressWarning, "Repair budget exhausted; preserving the latest staged draft for manual repair.", attemptNumber)
			return lastGenerated, lastValidation, nil
		}

		systemPrompt, userPrompt = buildBuilderNLRepairPrompts(
			prompt,
			currentDSL,
			targetModelName,
			connectionMode,
			generated.DSL,
			validation,
		)
	}

	return lastGenerated, lastValidation, nil
}

func reviewValidatedBuilderNLDraft(
	ctx context.Context,
	envoyURL string,
	req BuilderNLGenerateRequest,
	prompt string,
	targetModelName string,
	dslSource string,
	validation BuilderNLValidation,
	reporter builderNLProgressReporter,
) BuilderNLReview {
	if !validation.Ready {
		warnings := builderNLValidationWarnings(validation)
		if len(warnings) == 0 {
			warnings = []string{"Builder validation still needs manual attention before this draft should be applied."}
		}
		return BuilderNLReview{
			Ready:    false,
			Summary:  "Builder validation found issues that still need repair.",
			Warnings: warnings,
		}
	}
	reportBuilderNLProgress(reporter, "review", builderNLProgressInfo, "Running an AI review pass against the validated staged draft.", 0)

	reviewPrompt := buildBuilderNLReviewPrompt(prompt, targetModelName, dslSource)
	reviewContent, reviewErr := callBuilderNLModel(ctx, envoyURL, req, builderNLReviewSystemPrompt(), reviewPrompt, reporter, "review_call", 0)
	if reviewErr != nil {
		reportBuilderNLProgress(reporter, "review", builderNLProgressWarning, fmt.Sprintf("AI review was unavailable: %s", reviewErr), 0)
		return BuilderNLReview{
			Ready:    false,
			Summary:  "AI double-check was unavailable.",
			Warnings: []string{reviewErr.Error()},
		}
	}

	parsedReview, parseErr := parseBuilderNLReviewOutput(reviewContent)
	if parseErr != nil {
		reportBuilderNLProgress(reporter, "review", builderNLProgressWarning, fmt.Sprintf("AI review returned an unreadable payload: %s", parseErr), 0)
		return BuilderNLReview{
			Ready:    false,
			Summary:  "AI double-check returned an unreadable payload.",
			Warnings: []string{parseErr.Error()},
		}
	}
	if parsedReview.Ready {
		reportBuilderNLProgress(reporter, "review", builderNLProgressSuccess, "AI review completed and marked the staged draft ready.", 0)
	} else {
		reportBuilderNLProgress(reporter, "review", builderNLProgressWarning, "AI review completed and recommends manual inspection before apply.", 0)
	}
	return parsedReview
}

func validateBuilderNLDraft(source string) BuilderNLValidation {
	diags, _, valErrs := dsl.ValidateWithSymbols(source)
	diagnostics := convertBuilderNLDiagnostics(diags)
	errorCount := 0
	for _, diag := range diags {
		if diag.Level == dsl.DiagError {
			errorCount++
		}
	}

	compileError := ""
	if _, compileErrs := dsl.Compile(source); len(compileErrs) > 0 {
		compileError = joinBuilderNLErrors(compileErrs)
	}
	if compileError == "" && len(valErrs) > 0 {
		compileError = joinBuilderNLErrors(valErrs)
	}

	validation := BuilderNLValidation{
		Ready:        errorCount == 0 && compileError == "",
		Diagnostics:  diagnostics,
		ErrorCount:   errorCount,
		CompileError: strings.TrimSpace(compileError),
	}
	if validation.Ready {
		if formatted, err := dsl.Format(source); err == nil {
			validation.Diagnostics = diagnostics
			validation.CompileError = ""
			validation.ErrorCount = errorCount
			validation.Ready = true
			validation.draftDSL = formatted
		}
	}
	return validation
}

func convertBuilderNLDiagnostics(diags []dsl.Diagnostic) []BuilderNLDiagnostic {
	result := make([]BuilderNLDiagnostic, len(diags))
	for i, diag := range diags {
		var fixes []builderNLQuickFix
		if diag.Fix != nil {
			fixes = []builderNLQuickFix{{
				Description: diag.Fix.Description,
				NewText:     diag.Fix.NewText,
			}}
		}
		result[i] = BuilderNLDiagnostic{
			Level:   diag.Level.String(),
			Message: diag.Message,
			Line:    diag.Pos.Line,
			Column:  diag.Pos.Column,
			Fixes:   fixes,
		}
	}
	return result
}

func joinBuilderNLErrors(errs []error) string {
	parts := make([]string, 0, len(errs))
	for _, err := range errs {
		if err == nil {
			continue
		}
		parts = append(parts, strings.TrimSpace(err.Error()))
	}
	return strings.Join(parts, "\n")
}

func builderNLValidationSummary(validation BuilderNLValidation) string {
	var lines []string
	if strings.TrimSpace(validation.CompileError) != "" {
		lines = append(lines, "Compile error:")
		lines = append(lines, validation.CompileError)
	}
	for _, diag := range validation.Diagnostics {
		lines = append(lines, fmt.Sprintf("- [%s] %s (line %d, column %d)", diag.Level, diag.Message, diag.Line, diag.Column))
		if len(lines) >= 10 {
			break
		}
	}
	if len(lines) == 0 {
		return "No validation findings were captured."
	}
	return strings.Join(lines, "\n")
}

func builderNLValidationWarnings(validation BuilderNLValidation) []string {
	warnings := make([]string, 0, 6)
	if strings.TrimSpace(validation.CompileError) != "" {
		warnings = append(warnings, validation.CompileError)
	}
	for _, diag := range validation.Diagnostics {
		warnings = append(warnings, diag.Message)
		if len(warnings) == 6 {
			break
		}
	}
	return warnings
}

func builderNLValidationProgressMessage(validation BuilderNLValidation, attempt int) string {
	message := fmt.Sprintf("Validation attempt %d found %d error(s).", attempt, validation.ErrorCount)
	if strings.TrimSpace(validation.CompileError) != "" {
		message += " Compile step also reported an error."
	}
	return message
}

func (v BuilderNLValidation) formattedDSL() string {
	return strings.TrimSpace(v.draftDSL)
}

func callBuilderNLModel(
	ctx context.Context,
	envoyURL string,
	req BuilderNLGenerateRequest,
	systemPrompt string,
	userPrompt string,
	reporter builderNLProgressReporter,
	phase string,
	attempt int,
) (string, error) {
	return runBuilderNLModelCallWithProgress(ctx, reporter, phase, attempt, func() (string, error) {
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
	})
}

func runBuilderNLModelCallWithProgress(
	ctx context.Context,
	reporter builderNLProgressReporter,
	phase string,
	attempt int,
	call func() (string, error),
) (string, error) {
	if reporter == nil {
		return call()
	}

	start := time.Now()
	done := make(chan struct{})
	reportBuilderNLProgress(reporter, phase, builderNLProgressInfo, "Waiting for model response.", attempt)

	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-done:
				return
			case <-ticker.C:
				elapsed := int(time.Since(start).Seconds())
				reportBuilderNLProgress(reporter, phase, builderNLProgressInfo, fmt.Sprintf("Still waiting for model response (%ds elapsed).", elapsed), attempt)
			}
		}
	}()

	content, err := call()
	close(done)
	if err != nil {
		reportBuilderNLProgress(reporter, phase, builderNLProgressError, fmt.Sprintf("Model call failed: %s", err), attempt)
		return "", err
	}

	reportBuilderNLProgress(reporter, phase, builderNLProgressSuccess, fmt.Sprintf("Model response received after %s.", time.Since(start).Round(time.Second)), attempt)
	return content, nil
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

func verifyBuilderNLConnection(
	ctx context.Context,
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
	}
	content, err := callBuilderNLModel(
		ctx,
		envoyURL,
		generateReq,
		"You verify that a Builder AI connection is reachable.",
		"Reply with a short confirmation that the connection is working.",
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
		response.Endpoint = builderNLConnectionEndpoint(envoyURL, generateReq)
		return response, nil
	}

	response.ProviderKind = builderNLProviderOpenAICompatible
	response.ModelName = builderNLDefaultModelAlias
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
