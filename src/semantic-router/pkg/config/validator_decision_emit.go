package config

import "fmt"

const emitDirectiveKindRetention = "retention"

// validateDecisionEmitContracts keeps the YAML/config surface aligned with the
// DSL EMIT validation rules. DSL authors get diagnostics before compilation;
// direct YAML users still need the same hard contract checks during config load.
func validateDecisionEmitContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	for _, decision := range cfg.Decisions {
		seen := make(map[string]bool, len(decision.Emits))
		for i, emit := range decision.Emits {
			context := fmt.Sprintf("decision '%s', emits[%d]", decision.Name, i)
			if emit.Kind != emitDirectiveKindRetention {
				return fmt.Errorf("%s: unsupported EMIT kind %q; supported kinds: retention", context, emit.Kind)
			}
			if seen[emit.Kind] {
				return fmt.Errorf("%s: duplicate EMIT kind %q in the same decision", context, emit.Kind)
			}
			seen[emit.Kind] = true
			if err := validateRetentionDirectiveConfig(context, emit.Retention); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateRetentionDirectiveConfig(context string, retention *RetentionDirective) error {
	if retention == nil {
		return fmt.Errorf("%s: retention payload is required for kind %q", context, emitDirectiveKindRetention)
	}
	if retention.TTLTurns != nil && *retention.TTLTurns < 0 {
		return fmt.Errorf("%s retention: ttl_turns must be >= 0, got %d", context, *retention.TTLTurns)
	}
	if retention.Drop != nil && *retention.Drop && retention.TTLTurns != nil && *retention.TTLTurns > 0 {
		return fmt.Errorf("%s retention: drop=true conflicts with ttl_turns=%d", context, *retention.TTLTurns)
	}
	return nil
}
