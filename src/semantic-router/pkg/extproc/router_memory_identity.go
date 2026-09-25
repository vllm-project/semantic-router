package extproc

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// bindMemoryEmbedding isolates persisted vectors and the retrieval hot cache
// without modifying the user's configuration or deleting historical data.
func bindMemoryEmbedding(cfg *config.RouterConfig, sets ...*embedding.Set) (*config.RouterConfig, error) {
	bound, err := memoryConfigForIdentity(cfg, func(settings embedding.ConsumerSettings) (embedding.ContentIdentity, error) {
		if len(sets) == 0 || sets[0] == nil {
			return embedding.ContentIdentity{}, fmt.Errorf("memory embedding set was not prepared")
		}
		provider, err := sets[0].Get(settings.ModelType, 0, 0)
		if err != nil {
			return embedding.ContentIdentity{}, err
		}
		return embedding.ResolveNamespaceIdentity(provider, settings)
	})
	if err == nil && len(sets) > 0 {
		warnUnboundRemoteMemoryEmbedding(bound, sets[0])
	}
	return bound, err
}

func warnUnboundRemoteMemoryEmbedding(cfg *config.RouterConfig, prepared *embedding.Set) {
	if prepared == nil {
		return
	}
	provider, err := prepared.Get(detectMemoryEmbeddingModel(cfg), 0, 0)
	if err != nil || provider.Backend() != config.EmbeddingBackendOpenAICompatible {
		return
	}
	var model string
	if described, ok := provider.(embedding.Described); ok {
		model = described.EmbeddingInfo().Artifact
	}
	logging.Warnf("Memory: the router cannot verify the identity of remote embedding model %q, so memories stay in the configured collection if the provider changes that model; after an embedding model change, point memory at a new collection or index", model)
}

func memoryConfigForIdentity(cfg *config.RouterConfig, resolve func(embedding.ConsumerSettings) (embedding.ContentIdentity, error)) (*config.RouterConfig, error) {
	model := strings.ToLower(strings.TrimSpace(detectMemoryEmbeddingModel(cfg)))
	if model != "mmbert" && model != "multimodal" && model != "bert" {
		return cfg, nil
	}
	bound := *cfg
	bound.Memory = cfg.Memory
	bound.Memory.EmbeddingModel = model
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
	if *dimension == 0 && model == "mmbert" {
		*dimension = 256
	}
	fingerprint, deterministic := memory.DeterministicEmbeddingFingerprint(memory.EmbeddingConfig{Model: memory.EmbeddingModelType(model), Dimension: *dimension})
	if deterministic && *dimension == 0 && model == "multimodal" {
		return nil, fmt.Errorf("simulated multimodal memory requires an explicit storage dimension")
	}
	if !deterministic {
		identity, err := resolve(embedding.ConsumerSettings{
			ModelType: model, Dimension: *dimension, Layer: 0, InputPolicy: "memory-content-v1",
		})
		if model == "bert" && errors.Is(err, embedding.ErrIdentityUnsupported) {
			return cfg, nil
		}
		if err != nil {
			return nil, fmt.Errorf("bind memory embedding representation: %w", err)
		}
		if identity.Descriptor.Dimension <= 0 || (*dimension > 0 && *dimension != identity.Descriptor.Dimension) {
			return nil, fmt.Errorf("memory dimension %d differs from prepared representation %d", *dimension, identity.Descriptor.Dimension)
		}
		*dimension = identity.Descriptor.Dimension
		fingerprint = identity.Fingerprint
	}
	if fingerprint == "" {
		return nil, fmt.Errorf("memory embedding identity is empty")
	}
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
