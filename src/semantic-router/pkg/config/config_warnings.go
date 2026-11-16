package config

import (
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ConfigWarning is a finding that leaves a configuration valid but probably
// not what its author meant. The Router logs each one when it loads the
// configuration; validation without a load returns them instead.
type ConfigWarning struct {
	Code    string `json:"code"`
	Field   string `json:"field"`
	Message string `json:"message"`
}

// Warnings returns cfg's warnings in a stable order.
func Warnings(cfg *RouterConfig) []ConfigWarning {
	if cfg == nil || cfg.RoutingScope != "" {
		return nil
	}
	return slices.Concat(cfg.FormWarnings, modelResolutionWarnings(cfg))
}

// logConfigWarnings logs each warning that has no event of its own when the
// Router loads a configuration.
func logConfigWarnings(cfg *RouterConfig) error {
	if cfg == nil || cfg.RoutingScope != "" {
		return nil
	}
	for _, warning := range Warnings(cfg) {
		logging.ComponentWarnEvent("config", warning.Code, map[string]interface{}{
			"field":  warning.Field,
			"reason": warning.Message,
		})
	}
	return nil
}
