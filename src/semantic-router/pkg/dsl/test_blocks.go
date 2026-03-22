package dsl

import (
	"fmt"
	"strings"
)

// TestBlockResult is the runtime routing result for one TEST query.
type TestBlockResult struct {
	DecisionName string
	Confidence   float64
	MatchedRules []string
}

// TestBlockRunner evaluates one query against the routing signal pipeline.
type TestBlockRunner interface {
	EvaluateTestBlockQuery(query string) (*TestBlockResult, error)
}

// ProjectionPartitionRuntimeValidator emits native diagnostics that require runtime
// signal analysis beyond static AST validation.
type ProjectionPartitionRuntimeValidator interface {
	ValidateProjectionPartitions(prog *Program) []Diagnostic
}

// ValidateTestBlocks evaluates TEST blocks with a runtime runner and emits diagnostics
// for mismatches or runtime failures. Static TEST block diagnostics remain in the main validator.
func ValidateTestBlocks(prog *Program, runner TestBlockRunner) []Diagnostic {
	if prog == nil || runner == nil || len(prog.TestBlocks) == 0 {
		return nil
	}

	routeNames := make(map[string]bool, len(prog.Routes))
	for _, route := range prog.Routes {
		routeNames[route.Name] = true
	}

	diagnostics := make([]Diagnostic, 0)
	for _, tb := range prog.TestBlocks {
		for _, entry := range tb.Entries {
			if !shouldEvaluateTestEntry(entry, routeNames) {
				continue
			}
			if diag := validateTestEntry(tb, entry, runner); diag != nil {
				diagnostics = append(diagnostics, *diag)
			}
		}
	}

	return diagnostics
}

func shouldEvaluateTestEntry(entry *TestEntry, routeNames map[string]bool) bool {
	return entry.Query != "" && routeNames[entry.RouteName]
}

func validateTestEntry(tb *TestBlockDecl, entry *TestEntry, runner TestBlockRunner) *Diagnostic {
	result, err := runner.EvaluateTestBlockQuery(entry.Query)
	if err != nil {
		return runtimeTestFailureDiagnostic(tb.Name, entry, err)
	}
	if result == nil || strings.TrimSpace(result.DecisionName) == "" {
		return missingRuntimeRouteDiagnostic(tb.Name, entry)
	}
	if result.DecisionName == entry.RouteName {
		return nil
	}
	return mismatchedRuntimeRouteDiagnostic(tb.Name, entry, result)
}

func runtimeTestFailureDiagnostic(testName string, entry *TestEntry, err error) *Diagnostic {
	return &Diagnostic{
		Level: DiagError,
		Pos:   entry.Pos,
		Message: fmt.Sprintf(
			"TEST %s: runtime evaluation failed for %q: %v",
			testName,
			entry.Query,
			err,
		),
	}
}

func missingRuntimeRouteDiagnostic(testName string, entry *TestEntry) *Diagnostic {
	return &Diagnostic{
		Level: DiagError,
		Pos:   entry.Pos,
		Message: fmt.Sprintf(
			"TEST %s: query %q expected route %q, got no matching route",
			testName,
			entry.Query,
			entry.RouteName,
		),
	}
}

func mismatchedRuntimeRouteDiagnostic(testName string, entry *TestEntry, result *TestBlockResult) *Diagnostic {
	return &Diagnostic{
		Level: DiagError,
		Pos:   entry.Pos,
		Message: fmt.Sprintf(
			"TEST %s: query %q expected route %q, got %q (confidence %.3f; matched rules: %s)",
			testName,
			entry.Query,
			entry.RouteName,
			result.DecisionName,
			result.Confidence,
			formatMatchedRules(result.MatchedRules),
		),
	}
}

func appendFormattedTestBlocks(formatted string, testBlocks []*TestBlockDecl) string {
	if len(testBlocks) == 0 {
		return formatted
	}

	var sb strings.Builder
	sb.WriteString(strings.TrimRight(formatted, "\n"))
	sb.WriteString("\n\n")
	writeFormattedSection(&sb, "TESTS")
	for _, tb := range testBlocks {
		fmt.Fprintf(&sb, "TEST %s {\n", quoteName(tb.Name))
		for _, entry := range tb.Entries {
			fmt.Fprintf(&sb, "  %q -> %s\n", entry.Query, quoteName(entry.RouteName))
		}
		sb.WriteString("}\n\n")
	}
	return sb.String()
}

func writeFormattedSection(sb *strings.Builder, name string) {
	sb.WriteString("# =============================================================================\n")
	fmt.Fprintf(sb, "# %s\n", name)
	sb.WriteString("# =============================================================================\n\n")
}

func formatMatchedRules(matchedRules []string) string {
	if len(matchedRules) == 0 {
		return "none"
	}
	return strings.Join(matchedRules, ", ")
}

func programNeedsRuntimeValidation(prog *Program) bool {
	if prog == nil {
		return false
	}
	if len(prog.TestBlocks) > 0 {
		return true
	}
	for _, partition := range prog.ProjectionPartitions {
		if strings.EqualFold(partition.Semantics, "softmax_exclusive") {
			return true
		}
	}
	return false
}
