package config

import (
	"fmt"

	"gopkg.in/yaml.v2"
)

// Evaluate validates a candidate document without mutating desired, active,
// runtime, persistence, or version-history state.
func Evaluate(candidate []byte, opts EvaluateOptions) EvaluateResult {
	result := EvaluateResult{
		ContractVersion: DiagnosticContractVersion,
		Errors:          []Diagnostic{},
		Warnings:        []Diagnostic{},
	}

	raw, err := parseRawConfigMap(candidate)
	if err != nil {
		result.Errors = append(result.Errors, classifyEvaluationError(err))
		attachDiff(&result, nil, opts)
		return result
	}
	if raw == nil {
		raw = map[string]interface{}{}
	}

	unknown := CollectUnknownFieldDiagnostics(raw)
	if len(unknown) > 0 {
		result.Errors = append(result.Errors, unknown...)
		attachDiff(&result, raw, opts)
		return result
	}

	if _, err := ParseYAMLBytesWithoutEnvExpansion(candidate); err != nil {
		result.Errors = append(result.Errors, classifyEvaluationError(err))
		attachDiff(&result, raw, opts)
		return result
	}

	normalized, err := marshalRedactedYAML(raw)
	if err != nil {
		result.Errors = append(result.Errors, Diagnostic{
			Code:     DiagnosticValidationError,
			Severity: SeverityError,
			Stage:    StageValidate,
			Message:  err.Error(),
		})
		attachDiff(&result, raw, opts)
		return result
	}

	result.Valid = true
	result.NormalizedYAML = string(normalized)
	attachDiff(&result, raw, opts)
	return result
}

func attachDiff(result *EvaluateResult, candidate map[string]interface{}, opts EvaluateOptions) {
	if result == nil || !opts.CompareToActive {
		return
	}
	if len(opts.ActiveYAML) == 0 {
		result.Warnings = append(result.Warnings, Diagnostic{
			Code:     DiagnosticNoActiveSnapshot,
			Severity: SeverityWarning,
			Stage:    StageValidate,
			Message:  "no active snapshot available for comparison",
		})
		return
	}
	active, err := parseRawConfigMap(opts.ActiveYAML)
	if err != nil {
		result.Warnings = append(result.Warnings, Diagnostic{
			Code:     DiagnosticYAMLParseError,
			Severity: SeverityWarning,
			Stage:    StageParse,
			Message:  fmt.Sprintf("active snapshot could not be parsed for diff: %v", err),
		})
		return
	}
	if candidate == nil {
		return
	}
	result.Diff = DiffCanonicalDocuments(active, candidate)
}

func marshalRedactedYAML(raw map[string]interface{}) ([]byte, error) {
	redacted := RedactSensitiveConfigValue(normalizeYAMLValue(raw))
	encoded, err := yaml.Marshal(redacted)
	if err != nil {
		return nil, fmt.Errorf("failed to encode redacted config: %w", err)
	}
	return encoded, nil
}
