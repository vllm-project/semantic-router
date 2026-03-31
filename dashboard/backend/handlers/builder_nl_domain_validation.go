package handlers

import (
	"fmt"
	"strings"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl"
)

func validateBuilderNLDomainSignals(prog *dsl.Program) []BuilderNLDiagnostic {
	if prog == nil {
		return nil
	}

	supported := strings.Join(routerconfig.SupportedRoutingDomainNames(), ", ")
	diagnostics := make([]BuilderNLDiagnostic, 0)

	for _, signal := range prog.Signals {
		if signal == nil || signal.SignalType != "domain" {
			continue
		}

		name := strings.TrimSpace(signal.Name)
		if name == "" || builderNLDomainSignalHasExplicitCategories(signal) {
			continue
		}
		if routerconfig.IsSupportedRoutingDomainName(name) {
			continue
		}

		message := fmt.Sprintf(
			"SIGNAL domain %q must use a supported routing domain name (%s) or declare mmlu_categories explicitly",
			name,
			supported,
		)

		diagnostic := BuilderNLDiagnostic{
			Level:   "error",
			Message: message,
			Line:    signal.Pos.Line,
			Column:  signal.Pos.Column,
		}

		if suggestion := routerconfig.SuggestSupportedRoutingDomainName(name); suggestion != "" && suggestion != name {
			diagnostic.Message += fmt.Sprintf("; did you mean %q?", suggestion)
			diagnostic.Fixes = []builderNLQuickFix{{
				Description: fmt.Sprintf("Rename to %q and add explicit mmlu_categories if needed", suggestion),
				NewText:     suggestion,
			}}
		}

		diagnostics = append(diagnostics, diagnostic)
	}

	return diagnostics
}

func builderNLDomainSignalHasExplicitCategories(signal *dsl.SignalDecl) bool {
	raw, ok := signal.Fields["mmlu_categories"]
	if !ok {
		return false
	}

	arrayValue, ok := raw.(dsl.ArrayValue)
	if !ok {
		return true
	}

	return len(arrayValue.Items) > 0
}
