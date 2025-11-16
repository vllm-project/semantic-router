package pii

import (
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// PolicyChecker handles PII policy validation
type PolicyChecker struct {
	Config       *config.RouterConfig
	ModelConfigs map[string]config.ModelParams
}

// IsPIIEnabled checks if PII detection is enabled and properly configured
// For LoRA adapters, it falls back to the base model's PII policy if not found
func (c *PolicyChecker) IsPIIEnabled(model string) bool {
	modelConfig, exists := c.ModelConfigs[model]
	if !exists {
		// Try to find base model for LoRA adapters
		baseModel := c.findBaseModelForLoRA(model)
		if baseModel != "" {
			logging.Infof("LoRA adapter '%s' not found in model configs, falling back to base model '%s'", model, baseModel)
			modelConfig, exists = c.ModelConfigs[baseModel]
		}
	}

	if !exists {
		logging.Infof("No PII policy found for model %s, allowing request", model)
		return false
	}
	// if it is allowed by default, then it is not enabled
	return !modelConfig.PIIPolicy.AllowByDefault
}

// NewPolicyChecker creates a new PII policy checker
func NewPolicyChecker(cfg *config.RouterConfig, modelConfigs map[string]config.ModelParams) *PolicyChecker {
	return &PolicyChecker{
		Config:       cfg,
		ModelConfigs: modelConfigs,
	}
}

// CheckPolicy checks if the detected PII types are allowed for the given model
// For LoRA adapters, it falls back to the base model's PII policy if not found
func (pc *PolicyChecker) CheckPolicy(model string, detectedPII []string) (bool, []string, error) {
	if !pc.IsPIIEnabled(model) {
		logging.Infof("PII detection is disabled, allowing request")
		return true, nil, nil
	}

	modelConfig, exists := pc.ModelConfigs[model]
	if !exists {
		// Try to find base model for LoRA adapters
		baseModel := pc.findBaseModelForLoRA(model)
		if baseModel != "" {
			logging.Infof("LoRA adapter '%s' not found in model configs, falling back to base model '%s' for PII policy", model, baseModel)
			modelConfig, exists = pc.ModelConfigs[baseModel]
		}
	}

	if !exists {
		// If no specific config, allow by default
		logging.Infof("No PII policy found for model %s, allowing request", model)
		return true, nil, nil
	}

	policy := modelConfig.PIIPolicy
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
		logging.Warnf("PII policy violation for model %s: denied PII types %v", model, deniedPII)
		return false, deniedPII, nil
	}

	logging.Infof("PII policy check passed for model %s", model)
	return true, nil, nil
}

// FilterModelsForPII filters the list of candidate models based on PII policy compliance
func (pc *PolicyChecker) FilterModelsForPII(candidateModels []string, detectedPII []string) []string {
	var allowedModels []string

	for _, model := range candidateModels {
		allowed, _, err := pc.CheckPolicy(model, detectedPII)
		if err != nil {
			logging.Errorf("Error checking PII policy for model %s: %v", model, err)
			continue
		}
		if allowed {
			allowedModels = append(allowedModels, model)
		}
	}

	return allowedModels
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

// findBaseModelForLoRA finds the base model for a given LoRA adapter name
// Returns empty string if the LoRA adapter is not found in any model's LoRA list
func (pc *PolicyChecker) findBaseModelForLoRA(loraName string) string {
	for modelName, modelConfig := range pc.ModelConfigs {
		for _, lora := range modelConfig.LoRAs {
			if lora.Name == loraName {
				logging.Debugf("Found base model '%s' for LoRA adapter '%s'", modelName, loraName)
				return modelName
			}
		}
	}
	return ""
}
