package config

import "fmt"

func validateFusionContracts(cfg *RouterConfig) error {
	if err := ValidateFusionRuntimeConfig(cfg.Looper.Fusion); err != nil {
		return fmt.Errorf("global.integrations.looper.fusion: %w", err)
	}
	return nil
}
