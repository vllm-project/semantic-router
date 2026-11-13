package config

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// validateDecisionRetentionContracts keeps the YAML surface aligned with the
// DSL RETENTION validation rules. DSL authors get diagnostics before
// compilation; direct YAML users still need the same hard checks at load.
func validateDecisionRetentionContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	for _, decision := range cfg.AllRoutingDecisions() {
		if decision.Retention == nil {
			continue
		}
		if err := validateDecisionRetention(fmt.Sprintf("decision '%s'", decision.Name), decision.Retention); err != nil {
			return err
		}
	}
	return nil
}

func validateDecisionRetention(context string, retention *DecisionRetention) error {
	if retention.TTLTurns != nil && *retention.TTLTurns < 0 {
		return fmt.Errorf("%s retention: ttl_turns must be >= 0, got %d", context, *retention.TTLTurns)
	}
	if retention.Drop != nil && *retention.Drop && retention.TTLTurns != nil && *retention.TTLTurns > 0 {
		return fmt.Errorf("%s retention: drop=true conflicts with ttl_turns=%d", context, *retention.TTLTurns)
	}
	warnDecisionRetentionNoOp(context, retention)
	return nil
}

func warnDecisionRetentionNoOp(context string, retention *DecisionRetention) {
	if retention.Drop == nil && retention.TTLTurns == nil &&
		retention.KeepCurrentModel == nil && retention.PreferPrefixRetention == nil {
		logging.Warnf("%s retention: empty block has no effect. Set drop, ttl_turns, keep_current_model, or prefer_prefix_retention", context)
		return
	}
	if retention.TTLTurns != nil && *retention.TTLTurns == 0 && retention.Drop == nil {
		logging.Warnf("%s retention: ttl_turns=0 without drop is likely a no-op. Add drop: true to evict, or remove ttl_turns", context)
	}
}
