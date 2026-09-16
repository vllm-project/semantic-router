package config

import "strings"

// EmbeddingModelsNeeded identifies actual consumers in a single prepared scope.
func EmbeddingModelsNeeded(cfg *RouterConfig, primary string, sharedServices bool) map[string]bool {
	cfg = cfg.ModelConsumerScope()
	needed := map[string]bool{}
	if len(cfg.EmbeddingRules) > 0 || len(cfg.ReaskRules) > 0 || len(cfg.KnowledgeBases) > 0 || (len(cfg.ComplexityRules) > 0 && cfg.ComplexityModel.Backend == nil) {
		needed[primary] = true
	}
	if len(cfg.PreferenceRules) > 0 && cfg.PreferenceModel.ContrastiveEnabled() {
		model := strings.ToLower(strings.TrimSpace(cfg.PreferenceModel.EmbeddingModel))
		if model == "" {
			model = "mmbert"
		}
		if cfg.EmbeddingModels.UsesRemoteEmbeddingBackend() {
			model = primary
		}
		needed[model] = true
	}
	for _, rule := range cfg.JailbreakRules {
		if rule.Method == "contrastive" {
			needed[primary] = true
		}
	}
	if HasImageCandidatesInRules(cfg.ComplexityRules) {
		needed["multimodal"] = true
	}
	if sharedServices && cfg.Tools.Enabled {
		needed[primary] = true
	}
	if sharedServices && cfg.SemanticCache.Enabled {
		needed[SemanticCacheEmbeddingModel(cfg)] = true
	}
	if sharedServices && MemoryConfigured(cfg) {
		needed[MemoryEmbeddingModel(cfg)] = true
	}
	if sharedServices && cfg.VectorStore != nil && cfg.VectorStore.Enabled {
		model := cfg.VectorStore.EmbeddingModel
		if model == "" {
			model = "bert"
		}
		needed[model] = true
	}
	if cfg.ModelSelection.Enabled && cfg.ModelSelection.ML.ModelsPath != "" {
		needed[primary] = true
	}
	for _, decision := range cfg.Decisions {
		if algorithm := decision.Algorithm; algorithm != nil {
			switch algorithm.Type {
			case "knn", "kmeans", "svm", "mlp":
				needed[mlSelectionEmbeddingModel(cfg.ModelSelection.ML, primary)] = true
			case "router_dc", "automix", "hybrid":
				needed[primary] = true
			}
		}
		if decision.HasPlugin("tool_selection") {
			needed[primary] = true
		}
		if compression := decision.GetContextCompressionConfig(); compression != nil && compression.EffectiveScoring().Method != ContextCompressionScoringBM25 {
			needed[primary] = true
		}
		if rag := decision.GetRAGConfig(); rag != nil && rag.Enabled {
			switch rag.Backend {
			case "milvus", "qdrant", "hybrid":
				needed["bert"] = true
			case "external_api":
				ext, err := rag.ExternalAPIBackendConfig()
				if err == nil && strings.Contains(ext.RequestFormat, "embedding") {
					needed["bert"] = true
				}
			}
		}
	}
	return needed
}

func mlSelectionEmbeddingModel(ml MLSelectionConfig, primary string) string {
	// The selector factory uses its default embedding configuration when no
	// artifact paths are configured. Keep preparation aligned with that choice.
	if ml.ModelsPath == "" && ml.KNN.PretrainedPath == "" && ml.KMeans.PretrainedPath == "" &&
		ml.SVM.PretrainedPath == "" && ml.MLP.PretrainedPath == "" {
		return primary
	}
	if model := strings.ToLower(strings.TrimSpace(ml.ModelType)); model != "" {
		return model
	}
	return primary
}

func MemoryConfigured(cfg *RouterConfig) bool {
	if cfg.Memory.Enabled {
		return true
	}
	for _, decision := range cfg.Decisions {
		if decision.HasPlugin("memory") {
			return true
		}
	}
	return false
}

func SemanticCacheEmbeddingModel(cfg *RouterConfig) string {
	embeddingModel := strings.ToLower(strings.TrimSpace(cfg.EmbeddingModel))
	if embeddingModel != "" {
		return embeddingModel
	}

	switch {
	case cfg.MmBertModelPath != "":
		return "mmbert"
	case cfg.MultiModalModelPath != "":
		return "multimodal"
	case cfg.Qwen3ModelPath != "":
		return "qwen3"
	case cfg.GemmaModelPath != "":
		return "gemma"
	default:
		return "bert"
	}
}

func MemoryEmbeddingModel(cfg *RouterConfig) string {
	embeddingModel := strings.ToLower(strings.TrimSpace(cfg.Memory.EmbeddingModel))
	if embeddingModel != "" {
		return embeddingModel
	}

	switch {
	case cfg.MmBertModelPath != "":
		return "mmbert"
	case cfg.MultiModalModelPath != "":
		return "multimodal"
	case cfg.Qwen3ModelPath != "":
		return "qwen3"
	case cfg.GemmaModelPath != "":
		return "gemma"
	default:
		return "bert"
	}
}
