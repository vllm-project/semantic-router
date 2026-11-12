package config

import (
	"encoding/hex"
	"fmt"
)

// KVServingIdentity pins the weights, tokenizer and cache layout of a backend.
type KVServingIdentity struct {
	Model             string `yaml:"model" json:"model"`
	WeightRevision    string `yaml:"weight_revision" json:"weight_revision"`
	Tokenizer         string `yaml:"tokenizer" json:"tokenizer"`
	TokenizerRevision string `yaml:"tokenizer_revision" json:"tokenizer_revision"`
	Precision         string `yaml:"precision" json:"precision"`
	TensorParallel    int    `yaml:"tensor_parallel" json:"tensor_parallel"`
	KVHeads           int    `yaml:"kv_heads" json:"kv_heads"`
	HeadDim           int    `yaml:"head_dim" json:"head_dim"`
	HeadOrder         string `yaml:"head_order" json:"head_order"`
	AdapterID         string `yaml:"adapter_id,omitempty" json:"adapter_id,omitempty"`
}

// KVTransferBackend explicitly declares connector capabilities for a route.
type KVTransferBackend struct {
	ModelAlias  string            `yaml:"model_alias" json:"model_alias"`
	BackendName string            `yaml:"backend_name" json:"backend_name"`
	Identity    KVServingIdentity `yaml:"identity" json:"identity"`
	CanExport   bool              `yaml:"can_export" json:"can_export"`
	CanLoad     bool              `yaml:"can_load" json:"can_load"`
}

// KVTransferPair permits one directional mapper within an inclusive turn bound.
type KVTransferPair struct {
	MapperID        string            `yaml:"mapper_id" json:"mapper_id"`
	Source          KVServingIdentity `yaml:"source" json:"source"`
	Target          KVServingIdentity `yaml:"target" json:"target"`
	Enabled         bool              `yaml:"enabled" json:"enabled"`
	MaxTransferTurn int               `yaml:"max_transfer_turn" json:"max_transfer_turn"`
}

// KVTransferConfig is disabled unless explicitly enabled.
type KVTransferConfig struct {
	Enabled  bool                `yaml:"enabled" json:"enabled"`
	Backends []KVTransferBackend `yaml:"backends,omitempty" json:"backends,omitempty"`
	Pairs    []KVTransferPair    `yaml:"pairs,omitempty" json:"pairs,omitempty"`
}

func validateKVTransferConfig(cfg *RouterConfig) error {
	if cfg == nil || cfg.KVTransfer == nil || !cfg.KVTransfer.Enabled {
		return nil
	}
	transfer := cfg.KVTransfer
	if len(transfer.Backends) == 0 || len(transfer.Pairs) == 0 {
		return fmt.Errorf("enabled kv_transfer requires backends and pairs")
	}
	seen := map[string]bool{}
	for _, backend := range transfer.Backends {
		key := backend.ModelAlias + "\x00" + backend.BackendName
		if backend.ModelAlias == "" || backend.BackendName == "" || seen[key] {
			return fmt.Errorf("kv_transfer backend must have a unique model_alias and backend_name")
		}
		seen[key] = true
		if err := validateKVIdentity(backend.Identity); err != nil {
			return err
		}
	}
	for _, pair := range transfer.Pairs {
		if !pair.Enabled {
			continue
		}
		if pair.MapperID == "" || pair.MaxTransferTurn < 0 || pair.Source == pair.Target {
			return fmt.Errorf("kv_transfer pair requires a mapper and distinct identities with a nonnegative turn bound")
		}
		if err := validateKVIdentity(pair.Source); err != nil {
			return err
		}
		if err := validateKVIdentity(pair.Target); err != nil {
			return err
		}
	}
	return nil
}

func validateKVIdentity(identity KVServingIdentity) error {
	for _, revision := range []string{identity.WeightRevision, identity.TokenizerRevision} {
		if _, err := hex.DecodeString(revision); len(revision) != 40 || err != nil {
			return fmt.Errorf("kv_transfer identity requires immutable 40-character revisions")
		}
	}
	if identity.Model == "" || identity.Tokenizer == "" || identity.Precision != "bf16" || identity.TensorParallel != 1 || identity.KVHeads <= 0 || identity.HeadDim <= 0 || identity.HeadOrder == "" || identity.AdapterID != "" {
		return fmt.Errorf("kv_transfer requires a complete bf16 TP=1 identity without adapters")
	}
	return nil
}
