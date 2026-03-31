package handlers

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"time"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl"
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
	builderNLRepairMaxAttempts  = 1
	builderNLModelCallTimeout   = 60 * time.Second
	builderNLHeartbeatInterval  = 5 * time.Second
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
		reportBuilderNLProgress(reporter, "context", builderNLProgressInfo, contextMessage, 0)
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
		reportBuilderNLProgress(reporter, "context", builderNLProgressWarning, contextMessage, 0)
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

	review := reviewValidatedBuilderNLDraft(targetModelName, validation, generated.Warnings, reporter)
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

func buildBuilderNLTaskContext(
	request string,
	currentDSL string,
	targetModelName string,
	knownModelNames []string,
	connectionMode builderNLConnectionMode,
) string {
	preferredTarget := strings.TrimSpace(targetModelName)
	if preferredTarget == "" {
		preferredTarget = builderNLFallbackModelAlias
	}

	lines := []string{
		fmt.Sprintf("Original Builder routing request: %s", strings.TrimSpace(request)),
		fmt.Sprintf("Preferred target model for route references: %s", preferredTarget),
		fmt.Sprintf("Connection mode: %s", connectionMode),
		fmt.Sprintf("Fallback Builder alias only when no current router model is available: %s.", builderNLFallbackModelAlias),
		fmt.Sprintf("Known current router model cards: %s", builderNLKnownModelList(knownModelNames)),
		"Preserve unrelated valid declarations from the current DSL unless the request clearly replaces them.",
		"Do not emit YAML, providers/global blocks, or deploy-time endpoint configuration.",
	}
	if trimmedDSL := strings.TrimSpace(currentDSL); trimmedDSL != "" {
		lines = append(lines, "Current Builder DSL context:\n"+trimmedDSL)
	} else {
		lines = append(lines, "No current DSL is available. Build a fresh draft.")
	}
	return strings.Join(lines, "\n")
}

func builderNLKnownModelList(knownModelNames []string) string {
	if len(knownModelNames) == 0 {
		return "(none found in the current router config)"
	}
	return strings.Join(knownModelNames, ", ")
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

	clientReq := BuilderNLGenerateRequest{
		ConnectionMode:   connectionMode,
		CustomConnection: req.CustomConnection,
	}
	client := newBuilderNLLLMClient(envoyURL, clientReq, reporter)
	taskContext := buildBuilderNLTaskContext(prompt, currentDSL, targetModelName, knownModelNames, connectionMode)

	var (
		attemptBase             int
		lastGenerated           builderNLLLMOutput
		lastValidation          BuilderNLValidation
		lastValidationSignature string
		lastDraftSignature      string
		reviewDraft             string
		reviewFeedback          string
	)

	progressReporter := func(event sharednlgen.ProgressEvent) {
		client.setAttempt(event.Attempt)
		forwardBuilderNLNLGenProgress(reporter, event)
	}

	for repairRound := 0; repairRound <= builderNLRepairMaxAttempts; repairRound++ {
		var nlResult *sharednlgen.NLResult
		if repairRound == 0 {
			nlResult, err = sharednlgen.GenerateFromNL(
				ctx,
				client,
				prompt,
				sharednlgen.WithTaskContext(taskContext),
				sharednlgen.WithProgressReporter(progressReporter),
				sharednlgen.WithMaxRetries(builderNLRepairMaxAttempts),
			)
		} else {
			reportBuilderNLProgress(reporter, "repair", builderNLProgressInfo, "Preparing a shared repair prompt from repository validation findings.", attemptBase)
			nlResult, err = sharednlgen.RepairFromFeedback(
				ctx,
				client,
				prompt,
				reviewDraft,
				reviewFeedback,
				sharednlgen.WithTaskContext(taskContext),
				sharednlgen.WithProgressReporter(progressReporter),
				sharednlgen.WithAttemptOffset(attemptBase),
				sharednlgen.WithMaxRetries(0),
			)
		}
		if err != nil {
			return builderNLLLMOutput{}, BuilderNLValidation{}, err
		}

		attemptBase += nlResult.Attempts
		generated := builderNLLLMOutput{
			DSL:                strings.TrimSpace(nlResult.DSL),
			Summary:            builderNLDraftSummary(prompt, false),
			SuggestedTestQuery: builderNLSuggestedTestQuery(prompt),
			Warnings:           append([]string(nil), nlResult.Warnings...),
		}
		reportBuilderNLProgress(reporter, "validate", builderNLProgressInfo, "Running repository parse, validate, and compile checks against the staged draft.", attemptBase)
		validation := validateBuilderNLDraft(generated.DSL)

		validationSignature := builderNLValidationSignature(validation)
		sameValidationAsPrevious := repairRound > 0 && validationSignature != "" && validationSignature == lastValidationSignature
		sameDraftAsPrevious := repairRound > 0 && strings.TrimSpace(generated.DSL) != "" && strings.TrimSpace(generated.DSL) == lastDraftSignature

		lastGenerated = generated
		lastValidation = validation
		lastValidationSignature = validationSignature
		lastDraftSignature = strings.TrimSpace(generated.DSL)

		if validation.Ready {
			reportBuilderNLProgress(reporter, "validate", builderNLProgressSuccess, "Repository validation passed for the staged draft.", attemptBase)
			if formatted := strings.TrimSpace(validation.formattedDSL()); formatted != "" {
				reportBuilderNLProgress(reporter, "format", builderNLProgressSuccess, "Formatted the staged DSL after successful validation.", attemptBase)
				lastGenerated.DSL = formatted
			}
			lastGenerated.Summary = builderNLDraftSummary(prompt, true)
			return lastGenerated, lastValidation, nil
		}

		reportBuilderNLProgress(
			reporter,
			"validate",
			builderNLProgressWarning,
			builderNLValidationProgressMessage(validation, attemptBase),
			attemptBase,
		)
		if sameValidationAsPrevious || sameDraftAsPrevious {
			reportBuilderNLProgress(
				reporter,
				"repair",
				builderNLProgressWarning,
				"Repair findings were unchanged from the previous attempt, so Builder stopped early instead of spending another slow retry.",
				attemptBase,
			)
			lastGenerated.Summary = builderNLDraftSummary(prompt, false)
			return lastGenerated, lastValidation, nil
		}

		if repairRound == builderNLRepairMaxAttempts {
			reportBuilderNLProgress(reporter, "repair", builderNLProgressWarning, "Repair budget exhausted; preserving the latest staged draft for manual repair.", attemptBase)
			lastGenerated.Summary = builderNLDraftSummary(prompt, false)
			return lastGenerated, lastValidation, nil
		}

		reviewDraft = generated.DSL
		reviewFeedback = builderNLValidationSummary(validation)
	}

	lastGenerated.Summary = builderNLDraftSummary(prompt, lastValidation.Ready)
	return lastGenerated, lastValidation, nil
}

func forwardBuilderNLNLGenProgress(reporter builderNLProgressReporter, event sharednlgen.ProgressEvent) {
	if reporter == nil {
		return
	}
	phase := strings.TrimSpace(event.Phase)
	if phase == "" || phase == "complete" {
		return
	}
	reportBuilderNLProgress(reporter, phase, builderNLProgressInfo, event.Message, event.Attempt)
}

func builderNLDraftSummary(prompt string, ready bool) string {
	request := summarizeBuilderNLRequest(prompt)
	if ready {
		return fmt.Sprintf("Generated a staged Builder DSL draft for %s.", request)
	}
	return fmt.Sprintf("Generated a staged Builder DSL draft for %s, but repository validation still found issues.", request)
}

func summarizeBuilderNLRequest(prompt string) string {
	trimmed := strings.TrimSpace(prompt)
	if trimmed == "" {
		return "the current request"
	}
	if len(trimmed) > 96 {
		trimmed = trimmed[:93] + "..."
	}
	return strconvQuoteIfNeeded(trimmed)
}

func builderNLSuggestedTestQuery(prompt string) string {
	return strings.TrimSpace(prompt)
}

func reviewValidatedBuilderNLDraft(
	targetModelName string,
	validation BuilderNLValidation,
	generationWarnings []string,
	reporter builderNLProgressReporter,
) BuilderNLReview {
	reportBuilderNLProgress(reporter, "review", builderNLProgressInfo, "Building a readiness review from shared nlgen output and repository validation results.", 0)
	if !validation.Ready {
		warnings := builderNLValidationWarnings(validation)
		warnings = uniqueBuilderNLStrings(append(warnings, generationWarnings...))
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
		"Shared nlgen generation completed successfully.",
		"Repository validation passed.",
		"Compile pass completed without errors.",
	}
	if resolvedTarget := strings.TrimSpace(targetModelName); resolvedTarget != "" {
		checks = append(checks, fmt.Sprintf("Preferred target model resolved from the current router config: %s.", resolvedTarget))
	}
	reportBuilderNLProgress(reporter, "review", builderNLProgressSuccess, "Readiness review completed from shared nlgen output and repository validation results.", 0)
	return BuilderNLReview{
		Ready:    true,
		Summary:  "Repository validation passed; the staged draft is ready for Builder apply.",
		Warnings: uniqueBuilderNLStrings(generationWarnings),
		Checks:   checks,
	}
}

func uniqueBuilderNLStrings(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	seen := make(map[string]struct{}, len(values))
	var result []string
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		result = append(result, value)
	}
	return result
}

func strconvQuoteIfNeeded(value string) string {
	if strings.ContainsAny(value, " \t\n\"'") {
		return fmt.Sprintf("%q", value)
	}
	return value
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

	content, err := callBuilderNLMessages(
		ctx,
		envoyURL,
		generateReq,
		[]sharednlgen.ChatMessage{
			{Role: "system", Content: "You verify that a Builder AI connection is reachable."},
			{Role: "user", Content: "Reply with a short confirmation that the connection is working."},
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
