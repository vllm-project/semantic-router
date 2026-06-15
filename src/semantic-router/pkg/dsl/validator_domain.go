package dsl

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (v *Validator) checkDomainSignalConstraints(s *SignalDecl, context string) {
	mmluCategories := getMMLUCategories(s)
	for _, mmluCategory := range mmluCategories {
		if config.IsSupportedRoutingDomainName(mmluCategory) {
			continue
		}
		v.addDiag(
			DiagConstraint,
			s.Pos,
			fmt.Sprintf(
				"%s: mmlu_categories contains unsupported value %q; supported values: %s%s",
				context,
				mmluCategory,
				strings.Join(config.SupportedRoutingDomainNames(), ", "),
				formatDomainQuickFixSuffix(mmluCategory),
			),
			domainQuickFix(mmluCategory),
		)
	}
}

func domainQuickFix(value string) *QuickFix {
	suggestion := config.SuggestSupportedRoutingDomainName(value)
	if suggestion == "" || suggestion == value {
		return nil
	}
	return &QuickFix{
		Description: fmt.Sprintf("Change to %q", suggestion),
		NewText:     suggestion,
	}
}

func formatDomainQuickFixSuffix(value string) string {
	suggestion := config.SuggestSupportedRoutingDomainName(value)
	if suggestion == "" || suggestion == value {
		return ""
	}
	return fmt.Sprintf("; did you mean %q?", suggestion)
}
