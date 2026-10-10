package config

import (
	"fmt"
	"sort"
	"strings"

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

const (
	warningModalityDetectorOff = "modality_detector_disabled"
)

// Warnings returns cfg's warnings in a stable order.
func Warnings(cfg *RouterConfig) []ConfigWarning {
	if cfg == nil || cfg.RoutingScope != "" {
		return nil
	}
	var warnings []ConfigWarning
	if warning, ok := modalityDetectorOffWarning(cfg); ok {
		warnings = append(warnings, warning)
	}
	return warnings
}

// modalityDetectorOffWarning reports the modality rules and conditions of a
// configuration whose modality detector is off. The signal then never
// matches, and nothing else at request time says why.
func modalityDetectorOffWarning(cfg *RouterConfig) (ConfigWarning, bool) {
	if cfg.ModalityDetector.Enabled {
		return ConfigWarning{}, false
	}
	var uses []string
	_ = visitRoutingProfileConfigs(cfg, func(profile *RouterConfig) error {
		scope := ""
		if profile.RoutingScope != "" && profile.RoutingScope != DefaultRecipeName {
			scope = fmt.Sprintf("recipe %q: ", profile.RoutingScope)
		}
		for _, rule := range profile.ModalityRules {
			uses = append(uses, fmt.Sprintf("%srouting.signals.modality %q", scope, rule.Name))
		}
		for _, decision := range profile.Decisions {
			names := map[string]bool{}
			collectRuleNames(decision.Rules, SignalTypeModality, names)
			if len(names) > 0 {
				uses = append(uses, fmt.Sprintf("%sdecision %q", scope, decision.Name))
			}
		}
		return nil
	})
	if len(uses) == 0 {
		return ConfigWarning{}, false
	}
	sort.Strings(uses)
	return ConfigWarning{
		Code:  warningModalityDetectorOff,
		Field: "global.model_catalog.modules.modality_detector",
		Message: fmt.Sprintf(
			"the modality signal never matches: %s use it, but global.model_catalog.modules.modality_detector is not enabled; "+
				"enable it with method: classifier and confidence_threshold: 0.51 (Vela 2.0 0.3B), or remove the modality rules",
			strings.Join(uses, ", "),
		),
	}, true
}

// logConfigWarnings logs each warning that has no event of its own when the
// Router loads a configuration.
func logConfigWarnings(cfg *RouterConfig) error {
	if cfg == nil || cfg.RoutingScope != "" {
		return nil
	}
	if warning, ok := modalityDetectorOffWarning(cfg); ok {
		logging.ComponentWarnEvent("config", warning.Code, map[string]interface{}{
			"field":  warning.Field,
			"reason": warning.Message,
		})
	}
	return nil
}
