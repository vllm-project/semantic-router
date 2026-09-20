package extproc

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
)

// bindMemoryEmbedding isolates persisted vectors and the retrieval hot cache
// without modifying the user's configuration or deleting historical data.
func bindMemoryEmbedding(cfg *config.RouterConfig, sets ...*embedding.Set) (*config.RouterConfig, error) {
	var contractProvider embedding.DimensionContractProvider
	if len(sets) > 0 && sets[0] != nil {
		if provider, err := sets[0].Get("mmbert", 0, 0); err == nil {
			contractProvider, _ = provider.(embedding.DimensionContractProvider)
		}
	}
	return memoryConfigForIdentityWithContract(cfg, contractProvider, func(settings embedding.ConsumerSettings) (embedding.ContentIdentity, error) {
		if len(sets) == 0 || sets[0] == nil {
			return embedding.ContentIdentity{}, fmt.Errorf("memory embedding set was not prepared")
		}
		return sets[0].ResolveIdentity(settings)
	})
}

func memoryConfigForIdentity(cfg *config.RouterConfig, resolve func(embedding.ConsumerSettings) (embedding.ContentIdentity, error)) (*config.RouterConfig, error) {
	return memoryConfigForIdentityWithContract(cfg, nil, resolve)
}

func memoryConfigForIdentityWithContract(
	cfg *config.RouterConfig,
	contractProvider embedding.DimensionContractProvider,
	resolve func(embedding.ConsumerSettings) (embedding.ContentIdentity, error),
) (*config.RouterConfig, error) {
	if strings.ToLower(strings.TrimSpace(detectMemoryEmbeddingModel(cfg))) != "mmbert" {
		return cfg, nil
	}
	bound := *cfg
	bound.Memory = cfg.Memory
	bound.Memory.EmbeddingModel = "mmbert"
	backend := bound.Memory.Backend
	if backend == "" {
		backend = "milvus"
	}
	var dimension *int
	var logicalScope []string
	switch backend {
	case "milvus":
		if bound.Memory.Milvus.Collection == "" {
			bound.Memory.Milvus.Collection = "agentic_memory"
		}
		dimension = &bound.Memory.Milvus.Dimension
		logicalScope = []string{backend, bound.Memory.Milvus.Address, bound.Memory.Milvus.Collection}
	case "valkey":
		if cfg.Memory.Valkey == nil {
			return nil, fmt.Errorf("memory.valkey configuration is required")
		}
		vc := *cfg.Memory.Valkey
		bound.Memory.Valkey = &vc
		if vc.IndexName == "" {
			vc.IndexName = "mem_idx"
		}
		if vc.CollectionPrefix == "" {
			vc.CollectionPrefix = "mem:"
		}
		dimension = &vc.Dimension
		logicalScope = []string{backend, vc.Host, fmt.Sprint(vc.Port), fmt.Sprint(vc.Database), vc.IndexName, vc.CollectionPrefix}
	case "qdrant":
		if cfg.Memory.Qdrant == nil {
			return nil, fmt.Errorf("memory.qdrant configuration is required")
		}
		qc := *cfg.Memory.Qdrant
		bound.Memory.Qdrant = &qc
		if qc.Collection == "" {
			qc.Collection = "agentic_memory"
		}
		dimension = &qc.Dimension
		logicalScope = []string{backend, qc.Host, fmt.Sprint(qc.Port), qc.Collection}
	default:
		return nil, fmt.Errorf("unsupported memory backend: %q", backend)
	}
	_, deterministic := memory.DeterministicEmbeddingFingerprint(memory.EmbeddingConfig{Model: memory.EmbeddingModelMMBERT})
	if !deterministic && contractProvider != nil {
		contract, err := contractProvider.EmbeddingDimensionContract()
		if err != nil {
			return nil, fmt.Errorf("resolve memory embedding dimension contract: %w", err)
		}
		resolved, err := contract.Resolve(*dimension)
		if err != nil {
			return nil, fmt.Errorf("resolve memory embedding dimension: %w", err)
		}
		*dimension = resolved
	}
	if *dimension <= 0 {
		// Deterministic mode has no loaded model contract. Keep its stable,
		// legacy mmBERT width so simulated vectors retain their namespace.
		*dimension = 256
	}
	fingerprint, deterministic := memory.DeterministicEmbeddingFingerprint(memory.EmbeddingConfig{Model: memory.EmbeddingModelMMBERT, Dimension: *dimension})
	if !deterministic {
		identity, err := resolve(embedding.ConsumerSettings{
			ModelType: "mmbert", Dimension: *dimension, Layer: 0, InputPolicy: "memory-content-v1",
		})
		if err != nil {
			return nil, fmt.Errorf("bind memory embedding representation: %w", err)
		}
		fingerprint = identity.Fingerprint
	}
	if fingerprint == "" {
		return nil, fmt.Errorf("memory embedding identity is empty")
	}
	// Keep the namespace inputs compatible with the pre-contract layout. The
	// provider fingerprint is the authoritative representation identity and is
	// resolved with the effective layer and dimension above. Adding the width as
	// a second hash component would rename every existing memory collection on
	// upgrade, orphaning data even when the model and representation are
	// unchanged.
	logicalScope = append(logicalScope, fingerprint)
	encoded, err := json.Marshal(logicalScope)
	if err != nil {
		return nil, err
	}
	scope := fmt.Sprintf("%x", sha256.Sum256(encoded))
	switch backend {
	case "milvus":
		bound.Memory.Milvus.Collection = "memory_" + scope
	case "valkey":
		bound.Memory.Valkey.IndexName = "memory_idx_" + scope
		bound.Memory.Valkey.CollectionPrefix = "memory:" + scope + ":"
	case "qdrant":
		bound.Memory.Qdrant.Collection = "memory_" + scope
	}
	if cfg.Memory.RedisCache != nil {
		rc := *cfg.Memory.RedisCache
		bound.Memory.RedisCache = &rc
		if rc.KeyPrefix == "" {
			rc.KeyPrefix = "memory_cache:"
		}
		rc.KeyPrefix += "embedding:" + scope + ":"
	}
	return &bound, nil
}
