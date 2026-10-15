package kvtransfer

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

// PoliciesFromConfig preserves explicit enablement and exact directional identities.
func PoliciesFromConfig(cfg *config.KVTransferConfig) []PairPolicy {
	if cfg == nil || !cfg.Enabled {
		return nil
	}
	policies := make([]PairPolicy, 0, len(cfg.Pairs))
	for _, pair := range cfg.Pairs {
		policies = append(policies, PairPolicy{Mapper: Mapper{ID: pair.MapperID, Source: pair.Source, Target: pair.Target}, Enabled: pair.Enabled, MaxTransferTurn: pair.MaxTransferTurn})
	}
	return policies
}
