package config

import "strings"

// EmbeddingRequirement records a consumer's established vector/tokenizer
// contract. It does not infer a capability from an artifact or model alias.
type EmbeddingRequirement struct {
	Model, Consumer  string
	Dimension, Layer int
	Windows          bool
	Modality         string
	// LocalLayerHint preserves remote text embedding behavior: catalog layer
	// optimization has historically applied only to local model execution.
	LocalLayerHint bool
}

func EmbeddingRequirements(cfg *RouterConfig, primary string, sharedServices bool) []EmbeddingRequirement {
	cfg = cfg.ModelConsumerScope()
	var result []EmbeddingRequirement
	if len(cfg.EmbeddingRules) > 0 {
		options := cfg.EmbeddingConfig.WithDefaults()
		for _, rule := range cfg.EmbeddingRules {
			result = append(result, EmbeddingRequirement{Model: primary, Consumer: "embedding signal " + rule.Name, Dimension: options.TargetDimension, Layer: options.TargetLayer, LocalLayerHint: true, Modality: string(rule.EffectiveQueryModality())})
		}
	}
	if HasImageCandidatesInRules(cfg.ComplexityRules) {
		result = append(result, EmbeddingRequirement{Model: "multimodal", Consumer: "complexity image candidates", Modality: "image"})
	}
	for _, decision := range cfg.Decisions {
		if compression := decision.GetContextCompressionConfig(); compression != nil && compression.EffectiveScoring().Method != ContextCompressionScoringBM25 {
			result = append(result, EmbeddingRequirement{Model: primary, Consumer: "context compression", Dimension: cfg.EmbeddingConfig.TargetDimension, Layer: cfg.EmbeddingConfig.TargetLayer, LocalLayerHint: true})
		}
		if rag := decision.GetRAGConfig(); rag != nil && rag.Enabled {
			windows := rag.Backend == "milvus" || rag.Backend == "qdrant" || rag.Backend == "hybrid"
			if rag.Backend == "external_api" {
				external, err := rag.ExternalAPIBackendConfig()
				windows = err == nil && strings.Contains(external.RequestFormat, "embedding")
			}
			if windows {
				result = append(result, EmbeddingRequirement{Model: "bert", Consumer: "RAG query windows", Windows: true})
			}
		}
	}
	if !sharedServices {
		return result
	}
	if cfg.SemanticCache.Enabled {
		requirement := EmbeddingRequirement{Model: SemanticCacheEmbeddingModel(cfg), Consumer: "response cache", Windows: true}
		if cfg.SemanticCache.BackendType == "" || cfg.SemanticCache.BackendType == "memory" {
			switch requirement.Model {
			case "mmbert":
				requirement.Dimension, requirement.Layer = 256, 6
			case "multimodal":
				requirement.Dimension = 384
			}
		}
		result = append(result, requirement)
	}
	if MemoryConfigured(cfg) {
		model := MemoryEmbeddingModel(cfg)
		dimension := cfg.Memory.Milvus.Dimension
		if cfg.Memory.Backend == "valkey" && cfg.Memory.Valkey != nil {
			dimension = cfg.Memory.Valkey.Dimension
		}
		if cfg.Memory.Backend == "qdrant" && cfg.Memory.Qdrant != nil {
			dimension = cfg.Memory.Qdrant.Dimension
		}
		// Existing memory paths use complete output for BERT/Qwen/Gemma.
		switch model {
		case "mmbert":
			if dimension <= 0 {
				dimension = 256
			}
		case "multimodal":
			if dimension <= 0 {
				dimension = 384
			}
		default:
			dimension = 0
		}
		result = append(result, EmbeddingRequirement{Model: model, Consumer: "memory", Dimension: dimension})
	}
	if cfg.VectorStore != nil && cfg.VectorStore.Enabled {
		model := cfg.VectorStore.EmbeddingModel
		if model == "" {
			model = "bert"
		}
		result = append(result, EmbeddingRequirement{Model: model, Consumer: "vector store ingestion", Dimension: cfg.VectorStore.EmbeddingDimension})
	}
	return result
}
