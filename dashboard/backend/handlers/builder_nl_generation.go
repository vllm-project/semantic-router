package handlers

import (
	"context"
	"fmt"
	"strings"

	sharednlgen "github.com/vllm-project/semantic-router/src/semantic-router/pkg/nlgen"
)

func generateValidatedBuilderNLDraft(
	ctx context.Context,
	envoyURL string,
	req BuilderNLGenerateRequest,
	prompt string,
	currentDSL string,
	targetModelName string,
	knownModelNames []string,
	runtimeOptions builderNLRuntimeOptions,
	reporter builderNLProgressReporter,
) (builderNLLLMOutput, BuilderNLValidation, error) {
	connectionMode, err := builderNLConnectionModeOrDefault(req.ConnectionMode)
	if err != nil {
		return builderNLLLMOutput{}, BuilderNLValidation{}, err
	}

	client := newBuilderNLLLMClient(envoyURL, builderNLClientRequest(req, connectionMode), runtimeOptions, reporter)
	taskContext := buildBuilderNLTaskContext(prompt, currentDSL, targetModelName, knownModelNames, connectionMode)
	progressReporter := func(event sharednlgen.ProgressEvent) {
		client.setAttempt(event.Attempt)
		forwardBuilderNLNLGenProgress(reporter, event)
	}

	state := builderNLGenerationState{}
	for repairRound := 0; repairRound <= runtimeOptions.MaxRetries; repairRound++ {
		nlResult, err := runBuilderNLGenerationRound(
			ctx,
			client,
			prompt,
			taskContext,
			runtimeOptions,
			progressReporter,
			repairRound,
			state.attemptBase,
			state.reviewDraft,
			state.reviewFeedback,
			reporter,
		)
		if err != nil {
			return builderNLLLMOutput{}, BuilderNLValidation{}, err
		}

		state.attemptBase += nlResult.Attempts
		generated := builderNLOutputFromResult(prompt, nlResult)
		reportBuilderNLProgress(reporter, "validate", builderNLProgressInfo, "Running repository parse, validate, and compile checks against the staged draft.", state.attemptBase)
		validation := validateBuilderNLDraft(generated.DSL)

		outcome := evaluateBuilderNLGenerationAttempt(
			prompt,
			generated,
			validation,
			repairRound,
			runtimeOptions.MaxRetries,
			state,
			reporter,
		)
		state = outcome.state
		if outcome.done {
			return outcome.generated, outcome.validation, nil
		}
	}

	state.lastGenerated.Summary = builderNLDraftSummary(prompt, state.lastValidation.Ready)
	return state.lastGenerated, state.lastValidation, nil
}

type builderNLGenerationState struct {
	attemptBase             int
	lastGenerated           builderNLLLMOutput
	lastValidation          BuilderNLValidation
	lastValidationSignature string
	lastDraftSignature      string
	reviewDraft             string
	reviewFeedback          string
}

type builderNLGenerationOutcome struct {
	done       bool
	generated  builderNLLLMOutput
	validation BuilderNLValidation
	state      builderNLGenerationState
}

func builderNLClientRequest(
	req BuilderNLGenerateRequest,
	connectionMode builderNLConnectionMode,
) BuilderNLGenerateRequest {
	return BuilderNLGenerateRequest{
		ConnectionMode:   connectionMode,
		CustomConnection: req.CustomConnection,
		Temperature:      req.Temperature,
		MaxRetries:       req.MaxRetries,
		TimeoutSeconds:   req.TimeoutSeconds,
	}
}

func runBuilderNLGenerationRound(
	ctx context.Context,
	client *builderNLLLMClient,
	prompt string,
	taskContext string,
	runtimeOptions builderNLRuntimeOptions,
	progressReporter sharednlgen.ProgressReporter,
	repairRound int,
	attemptBase int,
	reviewDraft string,
	reviewFeedback string,
	reporter builderNLProgressReporter,
) (*sharednlgen.NLResult, error) {
	if repairRound == 0 {
		return sharednlgen.GenerateFromNL(
			ctx,
			client,
			prompt,
			sharednlgen.WithTemperature(runtimeOptions.Temperature),
			sharednlgen.WithMaxTokens(builderNLDefaultMaxTokens),
			sharednlgen.WithTaskContext(taskContext),
			sharednlgen.WithProgressReporter(progressReporter),
			sharednlgen.WithMaxRetries(runtimeOptions.MaxRetries),
		)
	}

	reportBuilderNLProgress(reporter, "repair", builderNLProgressInfo, "Preparing a shared repair prompt from repository validation findings.", attemptBase)
	return sharednlgen.RepairFromFeedback(
		ctx,
		client,
		prompt,
		reviewDraft,
		reviewFeedback,
		sharednlgen.WithTemperature(runtimeOptions.Temperature),
		sharednlgen.WithMaxTokens(builderNLDefaultMaxTokens),
		sharednlgen.WithTaskContext(taskContext),
		sharednlgen.WithProgressReporter(progressReporter),
		sharednlgen.WithAttemptOffset(attemptBase),
		sharednlgen.WithMaxRetries(0),
	)
}

func builderNLOutputFromResult(prompt string, result *sharednlgen.NLResult) builderNLLLMOutput {
	return builderNLLLMOutput{
		DSL:                strings.TrimSpace(result.DSL),
		Summary:            builderNLDraftSummary(prompt, false),
		SuggestedTestQuery: builderNLSuggestedTestQuery(prompt),
		Warnings:           append([]string(nil), result.Warnings...),
	}
}

func evaluateBuilderNLGenerationAttempt(
	prompt string,
	generated builderNLLLMOutput,
	validation BuilderNLValidation,
	repairRound int,
	maxRetries int,
	state builderNLGenerationState,
	reporter builderNLProgressReporter,
) builderNLGenerationOutcome {
	validationSignature := builderNLValidationSignature(validation)
	sameValidationAsPrevious := repairRound > 0 && validationSignature != "" && validationSignature == state.lastValidationSignature
	sameDraftAsPrevious := repairRound > 0 && strings.TrimSpace(generated.DSL) != "" && strings.TrimSpace(generated.DSL) == state.lastDraftSignature

	state.lastGenerated = generated
	state.lastValidation = validation
	state.lastValidationSignature = validationSignature
	state.lastDraftSignature = strings.TrimSpace(generated.DSL)

	if validation.Ready {
		reportBuilderNLProgress(reporter, "validate", builderNLProgressSuccess, "Repository validation passed for the staged draft.", state.attemptBase)
		if formatted := strings.TrimSpace(validation.formattedDSL()); formatted != "" {
			reportBuilderNLProgress(reporter, "format", builderNLProgressSuccess, "Formatted the staged DSL after successful validation.", state.attemptBase)
			state.lastGenerated.DSL = formatted
		}
		state.lastGenerated.Summary = builderNLDraftSummary(prompt, true)
		return builderNLGenerationOutcome{
			done:       true,
			generated:  state.lastGenerated,
			validation: state.lastValidation,
			state:      state,
		}
	}

	reportBuilderNLProgress(
		reporter,
		"validate",
		builderNLProgressWarning,
		builderNLValidationProgressMessage(validation, state.attemptBase),
		state.attemptBase,
	)
	if sameValidationAsPrevious || sameDraftAsPrevious {
		reportBuilderNLProgress(
			reporter,
			"repair",
			builderNLProgressWarning,
			"Repair findings were unchanged from the previous attempt, so Builder stopped early instead of spending another slow retry.",
			state.attemptBase,
		)
		state.lastGenerated.Summary = builderNLDraftSummary(prompt, false)
		return builderNLGenerationOutcome{
			done:       true,
			generated:  state.lastGenerated,
			validation: state.lastValidation,
			state:      state,
		}
	}

	if repairRound == maxRetries {
		reportBuilderNLProgress(reporter, "repair", builderNLProgressWarning, "Repair budget exhausted; preserving the latest staged draft for manual repair.", state.attemptBase)
		state.lastGenerated.Summary = builderNLDraftSummary(prompt, false)
		return builderNLGenerationOutcome{
			done:       true,
			generated:  state.lastGenerated,
			validation: state.lastValidation,
			state:      state,
		}
	}

	state.reviewDraft = generated.DSL
	state.reviewFeedback = builderNLValidationSummary(validation)
	return builderNLGenerationOutcome{state: state}
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
