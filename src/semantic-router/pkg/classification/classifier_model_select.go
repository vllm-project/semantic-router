package classification

import (
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// SelectBestModelForCategory selects the best model from a decision based on score and TTFT
func (c *Classifier) SelectBestModelForCategory(categoryName string) string {
	decision := c.findDecision(categoryName)
	if decision == nil {
		logging.Warnf("Could not find matching decision %s in config, using default model", categoryName)
		return c.Config.DefaultModel
	}

	bestModel, bestScore := c.selectBestModelInternalForDecision(decision, nil)

	if bestModel == "" {
		logging.Warnf("No models found for decision %s, using default model", categoryName)
		return c.Config.DefaultModel
	}

	logging.Infof("Selected model %s for decision %s with score %.4f", bestModel, categoryName, bestScore)
	return bestModel
}

// findDecision finds the decision configuration by name (case-insensitive)
func (c *Classifier) findDecision(decisionName string) *config.Decision {
	for i, decision := range c.Config.Decisions {
		if strings.EqualFold(decision.Name, decisionName) {
			return &c.Config.Decisions[i]
		}
	}
	return nil
}

// GetDecisionByName returns the decision configuration by name (case-insensitive)
func (c *Classifier) GetDecisionByName(decisionName string) *config.Decision {
	return c.findDecision(decisionName)
}

// GetCategorySystemPrompt returns the system prompt for a specific category if available.
// This is useful when the MCP server provides category-specific system prompts that should
// be injected when processing queries in that category.
// Returns empty string and false if no system prompt is available for the category.
func (c *Classifier) GetCategorySystemPrompt(category string) (string, bool) {
	if c.CategoryMapping == nil {
		return "", false
	}
	return c.CategoryMapping.GetCategorySystemPrompt(category)
}

// GetCategoryDescription returns the description for a given category if available.
// This is useful for logging, debugging, or providing context to downstream systems.
// Returns empty string and false if the category has no description.
func (c *Classifier) GetCategoryDescription(category string) (string, bool) {
	if c.CategoryMapping == nil {
		return "", false
	}
	return c.CategoryMapping.GetCategoryDescription(category)
}

// buildCategoryNameMappings builds translation maps between MMLU-Pro and generic categories
// selectBestModelInternalForDecision performs the core model selection logic for decisions
//
// modelFilter is optional - if provided, only models passing the filter will be considered
func (c *Classifier) selectBestModelInternalForDecision(decision *config.Decision, modelFilter func(string) bool) (string, float64) {
	bestModel := ""

	// With new architecture, we only support one model per decision (first ModelRef)
	if len(decision.ModelRefs) > 0 {
		modelRef := decision.ModelRefs[0]
		model := modelRef.Model

		if modelFilter == nil || modelFilter(model) {
			// Use LoRA name if specified, otherwise use the base model name
			finalModelName := model
			if modelRef.LoRAName != "" {
				finalModelName = modelRef.LoRAName
				logging.Debugf("Using LoRA adapter '%s' for base model '%s'", finalModelName, model)
			}
			bestModel = finalModelName
		}
	}

	return bestModel, 1.0 // Return score 1.0 since we don't have scores anymore
}

// SelectBestModelFromList selects the best model from a list of candidate models for a given decision
func (c *Classifier) SelectBestModelFromList(candidateModels []string, categoryName string) string {
	if len(candidateModels) == 0 {
		return c.Config.DefaultModel
	}

	decision := c.findDecision(categoryName)
	if decision == nil {
		// Return first candidate if decision not found
		return candidateModels[0]
	}

	bestModel, bestScore := c.selectBestModelInternalForDecision(decision,
		func(model string) bool {
			return slices.Contains(candidateModels, model)
		})

	if bestModel == "" {
		logging.Warnf("No suitable model found from candidates for decision %s, using first candidate", categoryName)
		return candidateModels[0]
	}

	logging.Debugf("Selected best model %s for decision %s from candidates (score=%.4f)", bestModel, categoryName, bestScore)
	return bestModel
}

// GetModelsForCategory returns all models that are configured for the given decision
// If a ModelRef has a LoRAName specified, the LoRA name is returned instead of the base model name
func (c *Classifier) GetModelsForCategory(categoryName string) []string {
	var models []string

	for _, decision := range c.Config.Decisions {
		if strings.EqualFold(decision.Name, categoryName) {
			for _, modelRef := range decision.ModelRefs {
				// Use LoRA name if specified, otherwise use the base model name
				if modelRef.LoRAName != "" {
					models = append(models, modelRef.LoRAName)
				} else {
					models = append(models, modelRef.Model)
				}
			}
			break
		}
	}

	return models
}

// updateBestModel updates the best model, score if the new score is better.
func (c *Classifier) updateBestModel(score float64, model string, bestScore *float64, bestModel *string) {
	if score > *bestScore {
		*bestScore = score
		*bestModel = model
	}
}
