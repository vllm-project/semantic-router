package dsl

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// checkDomainSignalConstraints rejects a domain signal that matches a label
// that is not a domain: its name, when it has no labels.
func (v *Validator) checkDomainSignalConstraints(s *SignalDecl, context string) {
	_, labelled := s.Fields["labels"]
	for _, label := range domainLabels(s) {
		if config.IsSupportedRoutingDomainName(label) {
			continue
		}
		message := fmt.Sprintf("%s: labels contains %q, which is not a domain; supported values: %s%s",
			context, label, strings.Join(config.SupportedRoutingDomainNames(), ", "), formatDomainQuickFixSuffix(label))
		if !labelled {
			message = fmt.Sprintf("%s: the name %q is not a domain, so the signal names the domains it matches: set labels to some of %s%s",
				context, label, strings.Join(config.SupportedRoutingDomainNames(), ", "), formatDomainQuickFixSuffix(label))
		}
		v.addDiag(DiagConstraint, s.Pos, message, domainQuickFix(label))
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
