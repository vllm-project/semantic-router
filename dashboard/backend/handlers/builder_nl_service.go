package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	pathpkg "path"
	"regexp"
	"strconv"
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

	builderNLFallbackModelAlias = "MoM"
	builderNLRepairMaxAttempts  = 1
	builderNLModelCallTimeout   = 60 * time.Second
	builderNLHeartbeatInterval  = 5 * time.Second
)

var (
	builderNLRouteConditionPattern     = regexp.MustCompile(`(?m)^([ \t]*)(?:CONDITION|WHEN)\s+"language\s*[:=]\s*([A-Za-z0-9_.-]+)"\s*$`)
	builderNLConditionKeywordPattern   = regexp.MustCompile(`(?m)^([ \t]*)CONDITION\b`)
	builderNLLanguageSignalDeclPattern = regexp.MustCompile(`(?m)^\s*SIGNAL\s+language\s+(?:"([^"]+)"|([A-Za-z_][A-Za-z0-9_-]*))\b`)
	builderNLLanguageSignalRefPattern  = regexp.MustCompile(`language\("([^"]+)"\)`)
	builderNLFirstRoutePattern         = regexp.MustCompile(`(?m)^ROUTE\b`)
	builderNLDslIdentifierPattern      = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_-]*$`)
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

	targetModelName := builderNLDraftTargetModelName(baseConfig)
	knownModelNames := builderNLConfiguredModelNames(baseConfig)
	if len(knownModelNames) > 0 {
		contextMessage := fmt.Sprintf(
			"Resolved %d current router model card(s); preferred draft target is %q.",
			len(knownModelNames),
			targetModelName,
		)
		if connectionMode == builderNLConnectionModeDefault {
			contextMessage = fmt.Sprintf(
				"Resolved %d current router model card(s); Builder will still use %q to generate the draft while route references prefer %q.",
				len(knownModelNames),
				builderNLFallbackModelAlias,
				targetModelName,
			)
		}
		reportBuilderNLProgress(
			reporter,
			"context",
			builderNLProgressInfo,
			contextMessage,
			0,
		)
	} else {
		contextMessage := fmt.Sprintf(
			"No current router model cards were found; Builder will fall back to %q.",
			targetModelName,
		)
		if connectionMode == builderNLConnectionModeDefault {
			contextMessage = fmt.Sprintf(
				"No current router model cards were found; Builder will use %q for both draft generation and fallback route references.",
				builderNLFallbackModelAlias,
			)
		}
		reportBuilderNLProgress(
			reporter,
			"context",
			builderNLProgressWarning,
			contextMessage,
			0,
		)
	}
	generated, validation, err := generateValidatedBuilderNLDraft(
		ctx,
		envoyURL,
		req,
		prompt,
		strings.TrimSpace(req.CurrentDSL),
		targetModelName,
		knownModelNames,
		reporter,
	)
	if err != nil {
		return BuilderNLGenerateResponse{}, err
	}

	review := reviewValidatedBuilderNLDraft(targetModelName, validation, reporter)

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
	emitBuilderNLProgress(reporter, phase, level, message, attempt, "stage", 0)
}

func reportBuilderNLHeartbeat(
	reporter builderNLProgressReporter,
	phase string,
	message string,
	attempt int,
	elapsedSeconds int,
) {
	emitBuilderNLProgress(reporter, phase, builderNLProgressInfo, message, attempt, "heartbeat", elapsedSeconds)
}

func emitBuilderNLProgress(
	reporter builderNLProgressReporter,
	phase string,
	level string,
	message string,
	attempt int,
	kind string,
	elapsedSeconds int,
) {
	if reporter == nil {
		return
	}

	reporter(BuilderNLProgressEvent{
		Phase:          phase,
		Level:          level,
		Message:        strings.TrimSpace(message),
		Attempt:        attempt,
		Kind:           kind,
		ElapsedSeconds: elapsedSeconds,
		Timestamp:      time.Now().UnixMilli(),
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

func builderNLConfiguredModelNames(config *routerconfig.CanonicalConfig) []string {
	if config == nil {
		return nil
	}

	names := make([]string, 0, len(config.Routing.ModelCards)+len(config.Providers.Models))
	seen := make(map[string]struct{}, len(config.Routing.ModelCards)+len(config.Providers.Models))
	appendName := func(raw string) {
		name := strings.TrimSpace(raw)
		if name == "" {
			return
		}
		if _, ok := seen[name]; ok {
			return
		}
		seen[name] = struct{}{}
		names = append(names, name)
	}

	appendName(config.Providers.Defaults.DefaultModel)
	for _, model := range config.Routing.ModelCards {
		appendName(model.Name)
	}
	for _, model := range config.Providers.Models {
		appendName(model.Name)
	}

	return names
}

func builderNLDraftTargetModelName(config *routerconfig.CanonicalConfig) string {
	names := builderNLConfiguredModelNames(config)
	if len(names) > 0 {
		return names[0]
	}
	return builderNLFallbackModelAlias
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
	knownModelNames []string,
	connectionMode builderNLConnectionMode,
) (string, string) {
	preferredTarget := strings.TrimSpace(targetModelName)
	if preferredTarget == "" {
		preferredTarget = builderNLFallbackModelAlias
	}

	systemPrompt := strings.Join([]string{
		"You generate vLLM Semantic Router Builder DSL.",
		"Return ONLY valid JSON with keys dsl, summary, suggestedTestQuery.",
		"The dsl value must be plain DSL source without markdown fences.",
		"Use Builder-compatible syntax only: MODEL, SIGNAL, ROUTE, PLUGIN, PRIORITY, WHEN, MODEL refs, ALGORITHM, and PLUGIN refs.",
		"Never emit CONDITION inside a ROUTE block. Route predicates must use WHEN <bool_expr>.",
		"If a route uses language(\"name\") in WHEN, add a matching SIGNAL language name declaration unless it already exists in the current DSL.",
		"Connection settings live outside the DSL, so never emit YAML or providers/global blocks.",
		"Always include at least one MODEL declaration and one ROUTE declaration.",
		fmt.Sprintf("Prefer existing router model cards before inventing new model names. The preferred fallback or default route model is %q.", preferredTarget),
		"Keep summaries concise and action-oriented.",
	}, "\n")

	contextBlock := "No current DSL is available. Build a fresh draft."
	if strings.TrimSpace(currentDSL) != "" {
		contextBlock = "Current DSL context (preserve unrelated valid declarations unless the request clearly replaces them):\n" + currentDSL
	}

	userPrompt := strings.Join([]string{
		fmt.Sprintf("Preferred target model for route references: %s", preferredTarget),
		fmt.Sprintf("Connection mode: %s", connectionMode),
		fmt.Sprintf("Fallback Builder alias only when no current router model is available: %s.", builderNLFallbackModelAlias),
		fmt.Sprintf("Known current router model cards: %s", builderNLKnownModelList(knownModelNames)),
		"Use short descriptions, add a sensible fallback route when the request implies one, and prefer deterministic priorities.",
		"Do not invent provider wiring or deploy-time endpoint config inside the DSL.",
		"For multilingual routing, use SIGNAL language declarations plus WHEN language(\"...\") route guards.",
		"Valid language-routing example:",
		builderNLLanguageRoutingExample(preferredTarget),
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
		"Never emit CONDITION inside a ROUTE block. Rewrite route predicates to WHEN <bool_expr>.",
		"If a route uses language(\"name\") in WHEN, add a matching SIGNAL language name declaration unless it already exists.",
		"Fix the validation and compile problems without adding unrelated features.",
	}, "\n")

	userPrompt := strings.Join([]string{
		fmt.Sprintf("Preferred target model for route references: %s", strings.TrimSpace(targetModelName)),
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
		"Valid language-routing example:",
		builderNLLanguageRoutingExample(strings.TrimSpace(targetModelName)),
		"",
		`JSON response shape: {"dsl":"...","summary":"...","suggestedTestQuery":"..."}`,
	}, "\n")

	return systemPrompt, userPrompt
}

func emptyBuilderNLContext(currentDSL string) string {
	if strings.TrimSpace(currentDSL) == "" {
		return "No current DSL is available."
	}
	return currentDSL
}

func builderNLKnownModelList(knownModelNames []string) string {
	if len(knownModelNames) == 0 {
		return "(none found in the current router config)"
	}
	return strings.Join(knownModelNames, ", ")
}

func builderNLLanguageRoutingExample(targetModelName string) string {
	targetModel := strings.TrimSpace(targetModelName)
	if targetModel == "" {
		targetModel = builderNLFallbackModelAlias
	}

	return strings.Join([]string{
		`SIGNAL language zh { description: "Chinese prompts" }`,
		`SIGNAL language en { description: "English prompts" }`,
		"",
		`ROUTE zh_route (description = "Route Chinese prompts before the fallback.") {`,
		`  PRIORITY 220`,
		`  WHEN language("zh")`,
		fmt.Sprintf(`  MODEL %q (reasoning = false)`, targetModel),
		`}`,
		"",
		`ROUTE en_route (description = "Route English prompts before the fallback.") {`,
		`  PRIORITY 210`,
		`  WHEN language("en")`,
		fmt.Sprintf(`  MODEL %q (reasoning = false)`, targetModel),
		`}`,
		"",
		`ROUTE default_route (description = "General fallback route.") {`,
		`  PRIORITY 100`,
		fmt.Sprintf(`  MODEL %q (reasoning = false)`, targetModel),
		`}`,
	}, "\n")
}

func generateValidatedBuilderNLDraft(
	ctx context.Context,
	envoyURL string,
	req BuilderNLGenerateRequest,
	prompt string,
	currentDSL string,
	targetModelName string,
	knownModelNames []string,
	reporter builderNLProgressReporter,
) (builderNLLLMOutput, BuilderNLValidation, error) {
	connectionMode, err := builderNLConnectionModeOrDefault(req.ConnectionMode)
	if err != nil {
		return builderNLLLMOutput{}, BuilderNLValidation{}, err
	}

	systemPrompt, userPrompt := buildBuilderNLGenerationPrompts(prompt, currentDSL, targetModelName, knownModelNames, connectionMode)
	var lastGenerated builderNLLLMOutput
	var lastValidation BuilderNLValidation
	lastValidationSignature := ""
	lastDraftSignature := ""

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
		if normalizedDSL, normalizationNotes := normalizeBuilderNLDraftSyntax(generated.DSL); normalizedDSL != generated.DSL {
			generated.DSL = normalizedDSL
			message := "Applied Builder DSL quick fixes before validation."
			if len(normalizationNotes) > 0 {
				message = "Applied Builder DSL quick fixes before validation: " + strings.Join(normalizationNotes, " ")
			}
			reportBuilderNLProgress(reporter, "parse", builderNLProgressInfo, message, attemptNumber)
		}
		reportBuilderNLProgress(reporter, "parse", builderNLProgressSuccess, "Parsed model output into a candidate DSL draft.", attemptNumber)

		reportBuilderNLProgress(reporter, "validate", builderNLProgressInfo, "Running repository parse, validate, and compile checks against the staged draft.", attemptNumber)
		validation := validateBuilderNLDraft(generated.DSL)
		validationSignature := builderNLValidationSignature(validation)
		sameValidationAsPrevious := attempt > 0 && validationSignature != "" && validationSignature == lastValidationSignature
		sameDraftAsPrevious := attempt > 0 && strings.TrimSpace(generated.DSL) != "" && strings.TrimSpace(generated.DSL) == lastDraftSignature
		lastGenerated = generated
		lastValidation = validation
		lastValidationSignature = validationSignature
		lastDraftSignature = strings.TrimSpace(generated.DSL)
		if validation.Ready {
			reportBuilderNLProgress(reporter, "validate", builderNLProgressSuccess, "Repository validation passed for the staged draft.", attemptNumber)
			if formatted := strings.TrimSpace(validation.formattedDSL()); formatted != "" {
				reportBuilderNLProgress(reporter, "format", builderNLProgressSuccess, "Formatted the staged DSL after successful validation.", attemptNumber)
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
		if sameValidationAsPrevious || sameDraftAsPrevious {
			reportBuilderNLProgress(
				reporter,
				"repair",
				builderNLProgressWarning,
				"Repair findings were unchanged from the previous attempt, so Builder stopped early instead of spending another slow retry.",
				attemptNumber,
			)
			return lastGenerated, lastValidation, nil
		}

		if attempt == builderNLRepairMaxAttempts {
			reportBuilderNLProgress(reporter, "repair", builderNLProgressWarning, "Repair budget exhausted; preserving the latest staged draft for manual repair.", attemptNumber)
			return lastGenerated, lastValidation, nil
		}
		reportBuilderNLProgress(reporter, "repair", builderNLProgressInfo, "Preparing a repair prompt from the latest repository validation findings.", attemptNumber)

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
	targetModelName string,
	validation BuilderNLValidation,
	reporter builderNLProgressReporter,
) BuilderNLReview {
	reportBuilderNLProgress(reporter, "review", builderNLProgressInfo, "Building a readiness review from repository validation results.", 0)
	if !validation.Ready {
		warnings := builderNLValidationWarnings(validation)
		if len(warnings) == 0 {
			warnings = []string{"Builder validation still needs manual attention before this draft should be applied."}
		}
		reportBuilderNLProgress(reporter, "review", builderNLProgressWarning, "Readiness review is blocked because repository validation still has issues.", 0)
		return BuilderNLReview{
			Ready:    false,
			Summary:  "Repository validation found issues that still need repair before apply.",
			Warnings: warnings,
		}
	}

	checks := []string{
		"Repository validation passed.",
		"Compile pass completed without errors.",
	}
	if resolvedTarget := strings.TrimSpace(targetModelName); resolvedTarget != "" {
		checks = append(checks, fmt.Sprintf("Preferred target model resolved from the current router config: %s.", resolvedTarget))
	}
	reportBuilderNLProgress(reporter, "review", builderNLProgressSuccess, "Readiness review completed from repository validation results.", 0)
	return BuilderNLReview{
		Ready:   true,
		Summary: "Repository validation passed; the staged draft is ready for Builder apply.",
		Checks:  checks,
	}
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

func normalizeBuilderNLDraftSyntax(source string) (string, []string) {
	normalized := strings.TrimSpace(source)
	if normalized == "" {
		return source, nil
	}

	var notes []string
	if builderNLRouteConditionPattern.MatchString(normalized) {
		normalized = builderNLRouteConditionPattern.ReplaceAllString(normalized, `${1}WHEN language("${2}")`)
		notes = append(notes, `rewrote quoted language CONDITION clauses into valid WHEN language("...") guards.`)
	}
	if builderNLConditionKeywordPattern.MatchString(normalized) {
		normalized = builderNLConditionKeywordPattern.ReplaceAllString(normalized, `${1}WHEN`)
		notes = append(notes, "rewrote CONDITION into WHEN for route predicates.")
	}

	normalized, missingSignals := ensureBuilderNLLanguageSignals(normalized)
	if len(missingSignals) > 0 {
		notes = append(notes, fmt.Sprintf("added %d missing SIGNAL language declaration(s) to match the generated WHEN clauses.", len(missingSignals)))
	}

	if strings.TrimSpace(normalized) == strings.TrimSpace(source) {
		return source, nil
	}
	return normalized, uniqueBuilderNLNotes(notes)
}

func ensureBuilderNLLanguageSignals(source string) (string, []string) {
	existingSignals := make(map[string]struct{})
	for _, match := range builderNLLanguageSignalDeclPattern.FindAllStringSubmatch(source, -1) {
		name := strings.TrimSpace(match[1])
		if name == "" {
			name = strings.TrimSpace(match[2])
		}
		if name == "" {
			continue
		}
		existingSignals[strings.ToLower(name)] = struct{}{}
	}

	var missing []string
	seenMissing := make(map[string]struct{})
	for _, match := range builderNLLanguageSignalRefPattern.FindAllStringSubmatch(source, -1) {
		name := strings.TrimSpace(match[1])
		if name == "" {
			continue
		}
		key := strings.ToLower(name)
		if _, ok := existingSignals[key]; ok {
			continue
		}
		if _, ok := seenMissing[key]; ok {
			continue
		}
		seenMissing[key] = struct{}{}
		missing = append(missing, name)
	}
	if len(missing) == 0 {
		return source, nil
	}

	declarations := make([]string, 0, len(missing))
	for _, name := range missing {
		declarations = append(
			declarations,
			fmt.Sprintf(`SIGNAL language %s { description: %q }`, builderNLDslName(name), builderNLLanguageDescription(name)),
		)
	}
	block := strings.Join(declarations, "\n\n")

	if routeIndex := builderNLFirstRoutePattern.FindStringIndex(source); routeIndex != nil {
		prefix := strings.TrimRight(source[:routeIndex[0]], "\n")
		suffix := strings.TrimLeft(source[routeIndex[0]:], "\n")
		if prefix == "" {
			return block + "\n\n" + suffix, missing
		}
		return prefix + "\n\n" + block + "\n\n" + suffix, missing
	}
	return strings.TrimRight(source, "\n") + "\n\n" + block + "\n", missing
}

func builderNLDslName(name string) string {
	if builderNLDslIdentifierPattern.MatchString(name) {
		return name
	}
	return strconv.Quote(name)
}

func builderNLLanguageDescription(name string) string {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "zh", "zh-cn", "zh-hans", "zh-hant", "chinese":
		return "Chinese prompts"
	case "en", "en-us", "en-gb", "english":
		return "English prompts"
	default:
		return fmt.Sprintf("%s prompts", strings.TrimSpace(name))
	}
}

func uniqueBuilderNLNotes(notes []string) []string {
	if len(notes) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(notes))
	var result []string
	for _, note := range notes {
		note = strings.TrimSpace(note)
		if note == "" {
			continue
		}
		if _, ok := seen[note]; ok {
			continue
		}
		seen[note] = struct{}{}
		result = append(result, note)
	}
	return result
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
	if firstFinding := builderNLPrimaryValidationFinding(validation); firstFinding != "" {
		message += " First blocker: " + firstFinding
	}
	return message
}

func builderNLValidationSignature(validation BuilderNLValidation) string {
	return strings.TrimSpace(builderNLValidationSummary(validation))
}

func builderNLPrimaryValidationFinding(validation BuilderNLValidation) string {
	if firstCompileLine := builderNLFirstSummaryLine(validation.CompileError); firstCompileLine != "" {
		return firstCompileLine
	}
	for _, diag := range validation.Diagnostics {
		if line := builderNLFirstSummaryLine(diag.Message); line != "" {
			return line
		}
	}
	return ""
}

func builderNLFirstSummaryLine(raw string) string {
	for _, line := range strings.Split(raw, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if len(line) > 160 {
			return line[:157] + "..."
		}
		return line
	}
	return ""
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
	callCtx, cancel := context.WithTimeout(ctx, builderNLModelCallTimeout)
	defer cancel()

	return runBuilderNLModelCallWithProgress(callCtx, reporter, phase, attempt, func() (string, error) {
		if req.ConnectionMode == builderNLConnectionModeCustom {
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
			return callBuilderNLCustomConnection(callCtx, *req.CustomConnection, systemPrompt, userPrompt)
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
		return callBuilderNLOpenAICompatible(
			callCtx,
			strings.TrimRight(envoyURL, "/")+"/v1/chat/completions",
			builderNLFallbackModelAlias,
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
	reportBuilderNLProgress(
		reporter,
		phase,
		builderNLProgressInfo,
		fmt.Sprintf("Waiting for model response (timeout %s).", builderNLModelCallTimeout.Round(time.Second)),
		attempt,
	)

	go func() {
		ticker := time.NewTicker(builderNLHeartbeatInterval)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-done:
				return
			case <-ticker.C:
				elapsed := int(time.Since(start).Seconds())
				reportBuilderNLHeartbeat(
					reporter,
					phase,
					fmt.Sprintf(
						"Still waiting for model response (%ds elapsed of %ds timeout).",
						elapsed,
						int(builderNLModelCallTimeout.Seconds()),
					),
					attempt,
					elapsed,
				)
			}
		}
	}()

	content, err := call()
	close(done)
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) {
			reportBuilderNLProgress(
				reporter,
				phase,
				builderNLProgressError,
				fmt.Sprintf("Model call timed out after %s.", builderNLModelCallTimeout.Round(time.Second)),
				attempt,
			)
		} else {
			reportBuilderNLProgress(reporter, phase, builderNLProgressError, fmt.Sprintf("Model call failed: %s", err), attempt)
		}
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
	}
	targetModelName := ""
	if strings.TrimSpace(configPath) != "" {
		baseConfig, configErr := readBuilderNLBaseConfig(configPath)
		if configErr != nil {
			return BuilderNLVerifyResponse{}, configErr
		}
		targetModelName = builderNLDraftTargetModelName(baseConfig)
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
