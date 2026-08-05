package config

import "fmt"

func validateReMoMContracts(cfg *RouterConfig) error {
	if err := ValidateReMoMRuntimeConfig(cfg.Looper.ReMoM); err != nil {
		return fmt.Errorf("global.integrations.looper.remom: %w", err)
	}
	return nil
}
