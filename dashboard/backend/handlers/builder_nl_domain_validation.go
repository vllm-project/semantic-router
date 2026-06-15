package handlers

import (
	"fmt"
	"regexp"
	"strings"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl"
)

type builderNLMultiWordDomainPattern struct {
	name        string
	declaration *regexp.Regexp
	reference   *regexp.Regexp
}

var builderNLMultiWordDomainPatterns = compileBuilderNLMultiWordDomainPatterns()

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

func validateBuilderNLMultiWordDomainSyntax(source string) []BuilderNLDiagnostic {
	if strings.TrimSpace(source) == "" {
		return nil
	}

	diagnostics := make([]BuilderNLDiagnostic, 0)
	lines := strings.Split(source, "\n")
	for index, line := range lines {
		for _, pattern := range builderNLMultiWordDomainPatterns {
			if matches := pattern.declaration.FindStringSubmatchIndex(line); len(matches) >= 4 {
				diagnostics = append(diagnostics, BuilderNLDiagnostic{
					Level: "error",
					Message: fmt.Sprintf(
						"Multi-word domain names must be quoted in DSL declarations. Use SIGNAL domain %q { ... } or choose an identifier such as %q and declare mmlu_categories explicitly.",
						pattern.name,
						strings.ReplaceAll(pattern.name, " ", "_"),
					),
					Line:   index + 1,
					Column: matches[2] + 1,
					Fixes: []builderNLQuickFix{{
						Description: fmt.Sprintf("Quote the multi-word domain name %q", pattern.name),
						NewText:     fmt.Sprintf("%q", pattern.name),
					}},
				})
			}
			if matches := pattern.reference.FindStringSubmatchIndex(line); len(matches) >= 4 {
				diagnostics = append(diagnostics, BuilderNLDiagnostic{
					Level: "error",
					Message: fmt.Sprintf(
						"Multi-word domain names must be quoted in route conditions. Use domain(%q).",
						pattern.name,
					),
					Line:   index + 1,
					Column: matches[2] + 1,
					Fixes: []builderNLQuickFix{{
						Description: fmt.Sprintf("Quote the domain reference %q", pattern.name),
						NewText:     fmt.Sprintf("%q", pattern.name),
					}},
				})
			}
		}
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

func compileBuilderNLMultiWordDomainPatterns() []builderNLMultiWordDomainPattern {
	domains := routerconfig.SupportedRoutingDomainNames()
	patterns := make([]builderNLMultiWordDomainPattern, 0, len(domains))
	for _, domain := range domains {
		domain = strings.TrimSpace(domain)
		if !strings.Contains(domain, " ") {
			continue
		}
		quoted := regexp.QuoteMeta(domain)
		spaced := strings.ReplaceAll(quoted, "\\ ", `\s+`)
		patterns = append(patterns, builderNLMultiWordDomainPattern{
			name:        domain,
			declaration: regexp.MustCompile(`(?i)\bSIGNAL\s+domain\s+(` + spaced + `)\s*\{`),
			reference:   regexp.MustCompile(`(?i)\bdomain\s*\(\s*(` + spaced + `)\s*\)`),
		})
	}
	return patterns
}
