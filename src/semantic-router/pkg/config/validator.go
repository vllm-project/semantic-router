package config

import (
	"fmt"
)

func validateRoutingStrategy(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	return cfg.Strategy.Validate()
}

func validateModelSelectionConfig(cfg *RouterConfig) error {
	if isLookupTableConfigConfigured(cfg.ModelSelection.LookupTables) {
		return fmt.Errorf("global.router.model_selection.lookup_tables has moved to future Router Learning experience; remove lookup_tables from public config")
	}
	return nil
}
