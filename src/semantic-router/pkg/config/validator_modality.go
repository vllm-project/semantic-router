package config

import (
	"fmt"
	"slices"
)

// validateRoutingModalityContracts checks the decisions a modality rule
// routes a request that asks for a generated image and text to: they
// reference an AR and a diffusion model, or an omni model.
func validateRoutingModalityContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	for _, decision := range cfg.AllRoutingDecisions() {
		for _, cond := range decision.Rules.Conditions {
			if cond.Type != SignalTypeModality || !cfg.modalityConditionMatches(cond, "text_image") {
				continue
			}
			if err := validateTextImageModalityDecision(cfg, decision); err != nil {
				return err
			}
		}
	}
	return nil
}

// modalityConditionMatches reports whether a modality condition matches
// label: it names the label, or a rule whose labels include it.
func (c *RouterConfig) modalityConditionMatches(cond RuleNode, label string) bool {
	if cond.Label != "" {
		return cond.Label == label
	}
	for _, rule := range c.ModalityRules {
		if rule.Name == cond.Name {
			return slices.Contains(rule.Labels, label)
		}
	}
	return false
}

func validateTextImageModalityDecision(cfg *RouterConfig, decision Decision) error {
	hasAR := false
	hasDiffusion := false
	hasOmni := false
	for _, ref := range decision.ModelRefs {
		if params, ok := cfg.ModelConfig[ref.Model]; ok {
			switch params.Modality {
			case "ar":
				hasAR = true
			case "diffusion":
				hasDiffusion = true
			case "omni":
				hasOmni = true
			}
		}
	}

	if hasOmni {
		return nil
	}
	if !hasAR || !hasDiffusion {
		return fmt.Errorf("decision %q routes on the modality label text_image, so its modelRefs must include both an AR model (modality: \"ar\") and a diffusion model (modality: \"diffusion\"), or an omni model (modality: \"omni\")", decision.Name)
	}
	return nil
}
