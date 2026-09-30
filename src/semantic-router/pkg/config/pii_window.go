package config

import "fmt"

// ValidatePIIWindow separates the admitted document budget from one forward.
func ValidatePIIWindow(cfg *RouterConfig) error {
	pii := cfg.PIIModel
	if binding, bound := cfg.ModelBindings["pii_classifier"]; bound {
		deployment, exists := cfg.ModelDeployments[binding.Deployment]
		if !exists {
			return fmt.Errorf("classifier.pii binding names an unknown deployment")
		}
		if pii.Window == nil {
			if deployment.Input.Overflow == "window" {
				return fmt.Errorf("classifier.pii window overflow requires window geometry")
			}
			return nil
		}
		return pii.ValidateBoundWindow(deployment)
	}
	return pii.ValidateWindow()
}

func (cfg PIIModel) ValidateWindow() error {
	if cfg.Window == nil {
		return nil
	}
	if cfg.Backend != nil || !cfg.UseMmBERT32K {
		return fmt.Errorf("classifier.pii.window requires local mmbert32k")
	}
	return cfg.validateWindowParameters(cfg.MaxSequenceLength)
}

func (cfg PIIModel) ValidateBoundWindow(deployment ModelDeployment) error {
	if cfg.Window == nil {
		return nil
	}
	if cfg.Backend != nil || (deployment.Provider != "candle" && deployment.Provider != "ort") {
		return fmt.Errorf("classifier.pii.window requires a local deployment")
	}
	if deployment.Input.Overflow != "window" || deployment.Input.MaxTokens <= 0 {
		return fmt.Errorf("classifier.pii.window requires deployment input overflow=window and a positive document budget")
	}
	return cfg.validateWindowParameters(deployment.Input.MaxTokens)
}

func (cfg PIIModel) validateWindowParameters(maxTokens int) error {
	head := SequenceHeadModelConfig{MaxSequenceLength: maxTokens, Window: cfg.Window}
	if err := head.ValidateWindow(); err != nil {
		return fmt.Errorf("classifier.pii.%w", err)
	}
	return nil
}
