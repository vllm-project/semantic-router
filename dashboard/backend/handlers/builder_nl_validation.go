package handlers

import (
	"fmt"
	"strings"

	routerauthoring "github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerauthoring"
)

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

func validateBuilderNLDraft(source string) BuilderNLValidation {
	diags, _, valErrs := routerauthoring.ValidateWithSymbols(source)
	diagnostics := convertBuilderNLDiagnostics(diags)
	diagnostics = append(diagnostics, validateBuilderNLMultiWordDomainSyntax(source)...)
	if prog, parseErrs := routerauthoring.Parse(source); len(parseErrs) == 0 {
		diagnostics = append(diagnostics, validateBuilderNLDomainSignals(prog)...)
	}

	compileError := ""
	if _, compileErrs := routerauthoring.Compile(source); len(compileErrs) > 0 {
		compileError = joinBuilderNLErrors(compileErrs)
	}
	if compileError == "" && len(valErrs) > 0 {
		compileError = joinBuilderNLErrors(valErrs)
	}

	errorCount := 0
	for _, diag := range diagnostics {
		if strings.EqualFold(diag.Level, "error") {
			errorCount++
		}
	}

	validation := BuilderNLValidation{
		Ready:        errorCount == 0 && compileError == "",
		Diagnostics:  diagnostics,
		ErrorCount:   errorCount,
		CompileError: strings.TrimSpace(compileError),
	}
	if validation.Ready {
		if formatted, err := routerauthoring.Format(source); err == nil {
			validation.draftDSL = formatted
		}
	}
	return validation
}

func convertBuilderNLDiagnostics(diags []routerauthoring.Diagnostic) []BuilderNLDiagnostic {
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
