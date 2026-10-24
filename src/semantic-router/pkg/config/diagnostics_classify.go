package config

import (
	"regexp"
	"strings"
)

var (
	recipePrefixPattern = regexp.MustCompile(`^routing recipe "([^"]+)":\s*`)
	decisionNamePattern = regexp.MustCompile(`(?i)decision ['"]([^'"]+)['"]`)
	fieldPathPattern    = regexp.MustCompile(
		`(?i)((?:[A-Za-z_][A-Za-z0-9_]*)(?:\[[0-9]+\]|\["[^"]+"\])?(?:\.(?:[A-Za-z_][A-Za-z0-9_]*)(?:\[[0-9]+\]|\["[^"]+"\])?)*)`,
	)
)

func classifyEvaluationError(err error) Diagnostic {
	if err == nil {
		return Diagnostic{}
	}
	message := err.Error()
	diagnostic := Diagnostic{
		Code:     DiagnosticValidationError,
		Severity: SeverityError,
		Stage:    StageValidate,
		Message:  message,
	}

	remainder := message
	if match := recipePrefixPattern.FindStringSubmatch(message); len(match) == 2 {
		diagnostic.Recipe = match[1]
		remainder = strings.TrimSpace(strings.TrimPrefix(message, match[0]))
	}

	applyDiagnosticClassification(&diagnostic, remainder)
	if diagnostic.Field == "" {
		diagnostic.Field = extractFieldPath(remainder)
	}
	if diagnostic.Resource == "" {
		diagnostic.Resource = resourceFromField(diagnostic.Field)
	}
	if diagnostic.Recipe == "" {
		diagnostic.Recipe = recipeFromMessage(remainder, diagnostic.Field)
	}
	return diagnostic
}

type classificationRule struct {
	code  string
	stage string
	terms []string
}

var diagnosticClassificationRules = []classificationRule{
	{code: DiagnosticYAMLParseError, stage: StageParse, terms: []string{"failed to parse", "yaml:"}},
	{stage: StageNormalize, terms: []string{"deprecated config fields", "removed config fields", "must use canonical"}},
	{code: DiagnosticCycleError, stage: StageResolve, terms: []string{"dependency cycle"}},
	{code: DiagnosticCapabilityError, stage: StageResolve, terms: []string{"outside decision modelrefs", "capability"}},
	{code: DiagnosticReferenceError, stage: StageResolve, terms: []string{
		"references unknown", "not declared", "undefined score", "references model", "references undefined",
	}},
	{code: DiagnosticQuorumError, terms: []string{
		"exceeds the configured panel", "exceeds the configured worker", "exceeds roles[", "exceeds max_parallel", "quorum",
	}},
	{code: DiagnosticBudgetError, terms: []string{"budget.", "target_tokens must be less"}},
	{code: DiagnosticFallbackError, terms: []string{"on_error", "fallback"}},
	{code: DiagnosticConflict, terms: []string{"duplicate", "conflict"}},
}

func applyDiagnosticClassification(diagnostic *Diagnostic, message string) {
	lower := strings.ToLower(message)
	if rule, ok := matchClassificationRule(lower); ok {
		if rule.code != "" {
			diagnostic.Code = rule.code
		}
		if rule.stage != "" {
			diagnostic.Stage = rule.stage
		}
	}
	if field := fieldFromDecisionMessage(message); field != "" && diagnostic.Field == "" {
		diagnostic.Field = field
	}
}

func matchClassificationRule(lower string) (classificationRule, bool) {
	for _, rule := range diagnosticClassificationRules {
		if containsAny(lower, rule.terms) {
			return rule, true
		}
	}
	return classificationRule{}, false
}

func containsAny(haystack string, terms []string) bool {
	for _, term := range terms {
		if strings.Contains(haystack, term) {
			return true
		}
	}
	return false
}

func extractFieldPath(message string) string {
	trimmed := strings.TrimSpace(message)
	if match := fieldPathPattern.FindStringSubmatch(trimmed); len(match) > 1 {
		candidate := match[1]
		if strings.Contains(candidate, ".") || strings.Contains(candidate, "[") {
			return strings.TrimSuffix(candidate, ":")
		}
		switch candidate {
		case "version", "listeners", "providers", "routing", "entrypoints", "recipes", "global":
			return candidate
		}
	}
	return ""
}

func fieldFromDecisionMessage(message string) string {
	match := decisionNamePattern.FindStringSubmatch(message)
	if len(match) != 2 {
		return ""
	}
	return `routing.decisions["` + match[1] + `"]` + decisionFieldSuffix(strings.ToLower(message))
}

func decisionFieldSuffix(lower string) string {
	switch {
	case strings.Contains(lower, "algorithm"):
		return ".algorithm" + algorithmFieldSuffix(lower)
	case strings.Contains(lower, "plugin"):
		return ".plugins"
	case strings.Contains(lower, "modelrefs") || strings.Contains(lower, "model '"):
		return ".modelRefs"
	case strings.Contains(lower, "signal") || strings.Contains(lower, "condition"):
		return ".rules"
	default:
		return ""
	}
}

func algorithmFieldSuffix(lower string) string {
	switch {
	case strings.Contains(lower, "on_error"):
		return ".on_error"
	case strings.Contains(lower, "fusion"):
		return ".fusion"
	case strings.Contains(lower, "workflows"):
		return ".workflows"
	case strings.Contains(lower, "prompt"):
		return ".prompt"
	default:
		return ""
	}
}

func recipeFromMessage(message, field string) string {
	if match := decisionNamePattern.FindStringSubmatch(message); len(match) == 2 {
		return string(DefaultRecipeName)
	}
	if strings.HasPrefix(field, "routing.") || strings.HasPrefix(field, "recipes.") {
		return string(DefaultRecipeName)
	}
	return ""
}

func resourceFromField(field string) string {
	if field == "" {
		return ""
	}
	trimmed := strings.TrimSpace(field)
	end := len(trimmed)
	for i, r := range trimmed {
		if r == '.' || r == '[' {
			end = i
			break
		}
	}
	return trimmed[:end]
}
