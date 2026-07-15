package config

import "fmt"

func validateFlowContracts(cfg *RouterConfig) error {
	if err := ValidateFlowRuntimeConfig(cfg.Looper.Flow); err != nil {
		return fmt.Errorf("global.integrations.looper.flow: %w", err)
	}
	return nil
}
