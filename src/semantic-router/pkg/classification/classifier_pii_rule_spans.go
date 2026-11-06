package classification

import (
	"context"
	"fmt"
	"strings"
)

// PIIRuleSpans returns the personal data the named pii rule finds in text:
// the spans its detector reports at the rule's threshold, without the types
// the rule's pii_types_allowed lets through. Masking replaces these spans. An
// error, including a scan that read only part of the text, means the text
// could not be read whole, so a caller that masks must not send it.
func (c *Classifier) PIIRuleSpans(ctx context.Context, ruleName string, text string) ([]PIIDetection, error) {
	rule := c.Config.PIIRule(ruleName)
	if rule == nil {
		return nil, fmt.Errorf("pii rule %q is not declared in this recipe", ruleName)
	}
	if !c.IsPIIEnabled() {
		return nil, fmt.Errorf("the detector of pii rule %q is unavailable", ruleName)
	}
	detections, err := c.ClassifyPIIWithDetailsAndThreshold(ctx, text, rule.Threshold)
	if err != nil {
		return nil, err
	}
	spans := detections[:0]
	for _, detection := range detections {
		if !allowedPIIType(rule.PIITypesAllowed, detection.EntityType) {
			spans = append(spans, detection)
		}
	}
	return spans, nil
}

func allowedPIIType(allowed []string, entityType string) bool {
	for _, candidate := range allowed {
		if strings.EqualFold(candidate, entityType) {
			return true
		}
	}
	return false
}
