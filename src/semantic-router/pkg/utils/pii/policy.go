package pii

import (
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// PolicyChecker handles PII policy validation based on decisions
type PolicyChecker struct {
	Config *config.RouterConfig
}

// IsPIIEnabled checks if PII detection is enabled for a given decision
func (c *PolicyChecker) IsPIIEnabled(decisionName string) bool {
	if decisionName == "" {
		logging.Infof("No decision specified, PII detection disabled")
		return false
	}

	decision := c.Config.GetDecisionByName(decisionName)
	if decision == nil {
		logging.Infof("Decision %s not found, PII detection disabled", decisionName)
		return false
	}

	piiConfig := decision.GetPIIConfig()
	if piiConfig == nil {
		logging.Infof("No PII config found for decision %s, PII detection disabled", decisionName)
		return false
	}

	// PII detection is enabled if the plugin is enabled
	return piiConfig.Enabled
}

// NewPolicyChecker creates a new PII policy checker
func NewPolicyChecker(cfg *config.RouterConfig) *PolicyChecker {
	return &PolicyChecker{
		Config: cfg,
	}
}

// CheckPolicy checks if the detected PII types are allowed for the given decision
func (pc *PolicyChecker) CheckPolicy(decisionName string, detectedPII []string) (bool, []string, error) {
	if !pc.IsPIIEnabled(decisionName) {
		logging.Infof("PII detection is disabled for decision %s, allowing request", decisionName)
		return true, nil, nil
	}

	decision := pc.Config.GetDecisionByName(decisionName)
	if decision == nil {
		logging.Infof("Decision %s not found, allowing request", decisionName)
		return true, nil, nil
	}

	policy := decision.GetDecisionPIIPolicy()
	var deniedPII []string

	for _, piiType := range detectedPII {
		if piiType == "NO_PII" {
			continue // Skip non-PII content
		}

		// If allow_by_default is true, all PII types are allowed
		if policy.AllowByDefault {
			continue
		}

		// If allow_by_default is false, check if this PII type is explicitly allowed
		isAllowed := slices.Contains(policy.PIITypes, piiType)
		if !isAllowed {
			deniedPII = append(deniedPII, piiType)
		}
	}

	if len(deniedPII) > 0 {
		logging.Warnf("PII policy violation for decision %s: denied PII types %v", decisionName, deniedPII)
		return false, deniedPII, nil
	}

	logging.Infof("PII policy check passed for decision %s", decisionName)
	return true, nil, nil
}

// ExtractAllContent extracts all content from user and non-user messages for PII analysis
func ExtractAllContent(userContent string, nonUserMessages []string) []string {
	var allContent []string
	if userContent != "" {
		allContent = append(allContent, userContent)
	}
	allContent = append(allContent, nonUserMessages...)
	return allContent
}
