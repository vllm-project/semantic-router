package config

import "fmt"

func validateModalityContracts(cfg *RouterConfig) error {
	if cfg.ModalityDetector.Enabled {
		if err := cfg.ModalityDetector.Validate(); err != nil {
			return fmt.Errorf("modality_detector: %w", err)
		}
	}
	if err := validateImageGenBackends(cfg); err != nil {
		return err
	}
	if err := validateModalityDecisions(cfg); err != nil {
		return err
	}
	return validateModalityRules(cfg.ModalityRules)
}

// validateModalityRules validates modality rule configurations.
func validateModalityRules(rules []ModalityRule) error {
	validNames := map[string]bool{"AR": true, "DIFFUSION": true, "BOTH": true}
	for i, rule := range rules {
		if rule.Name == "" {
			return fmt.Errorf("modality_rules[%d]: name cannot be empty", i)
		}
		if !validNames[rule.Name] {
			return fmt.Errorf("modality_rules[%d] (%s): name must be one of \"AR\", \"DIFFUSION\", or \"BOTH\"", i, rule.Name)
		}
	}
	return nil
}

// validateModalityDecisions validates that decisions using modality signals have correct modelRefs.
// Specifically, a BOTH decision must reference both an AR and a diffusion model, OR a single omni model.
func validateModalityDecisions(cfg *RouterConfig) error {
	for _, decision := range cfg.Decisions {
		for _, cond := range decision.Rules.Conditions {
			if cond.Type != SignalTypeModality || cond.Name != "BOTH" {
				continue
			}
			if err := validateBothModalityDecision(cfg, decision); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateBothModalityDecision(cfg *RouterConfig, decision Decision) error {
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
		return fmt.Errorf("decision %q uses modality condition \"BOTH\" but modelRefs must include both an AR model (modality: \"ar\") and a diffusion model (modality: \"diffusion\"), or an omni model (modality: \"omni\")", decision.Name)
	}
	return nil
}

// validateImageGenBackends validates image_gen_backends entries and model_config references.
func validateImageGenBackends(cfg *RouterConfig) error {
	validTypes := map[string]bool{"vllm_omni": true, "openai": true}

	for name, entry := range cfg.ImageGenBackends {
		if entry.Type == "" {
			return fmt.Errorf("image_gen_backends[%s]: type is required (one of \"vllm_omni\", \"openai\")", name)
		}
		if !validTypes[entry.Type] {
			return fmt.Errorf("image_gen_backends[%s]: unknown type %q (must be \"vllm_omni\" or \"openai\")", name, entry.Type)
		}

		switch entry.Type {
		case "vllm_omni":
			if entry.BaseURL == "" {
				return fmt.Errorf("image_gen_backends[%s]: base_url is required for vllm_omni", name)
			}
		case "openai":
			if entry.APIKey == "" {
				return fmt.Errorf("image_gen_backends[%s]: api_key is required for openai", name)
			}
		}
	}

	for modelName, params := range cfg.ModelConfig {
		if params.ImageGenBackend == "" {
			continue
		}
		if _, ok := cfg.ImageGenBackends[params.ImageGenBackend]; !ok {
			return fmt.Errorf("model_config[%s]: image_gen_backend %q not found in image_gen_backends", modelName, params.ImageGenBackend)
		}
	}

	return nil
}
