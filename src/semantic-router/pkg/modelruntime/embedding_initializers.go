package modelruntime

import (
	"strings"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

var (
	initModel                    = candle_binding.InitModel
	initEmbeddingModels          = candle_binding.InitEmbeddingModels
	initEmbeddingModelsBatched   = candle_binding.InitEmbeddingModelsBatched
	initMultiModalEmbeddingModel = candle_binding.InitMultiModalEmbeddingModel
)

func initializeUnifiedEmbeddingModels(cfg *config.RouterConfig, component string, paths embeddingPaths) bool {
	if !paths.hasUnifiedModels() {
		return false
	}

	semanticCacheNeedsBatched, mlSelectionNeedsBatched := batchedEmbeddingNeeds(cfg, paths.qwen3)
	useBatched := semanticCacheNeedsBatched || mlSelectionNeedsBatched
	logBatchedEmbeddingNeeds(component, semanticCacheNeedsBatched, mlSelectionNeedsBatched)
	if err := initUnifiedEmbeddingModelFactory(paths, useBatched); err != nil {
		logging.ComponentErrorEvent(component, "embedding_models_init_failed", map[string]interface{}{
			"use_batched": useBatched,
			"error":       err.Error(),
		})
		logging.ComponentWarnEvent(component, "embedding_runtime_degraded", map[string]interface{}{
			"embedding_api_placeholder": true,
			"tools_database_disabled":   true,
		})
		return false
	}

	logging.ComponentEvent(component, "embedding_models_initialized", map[string]interface{}{
		"use_batched": useBatched,
	})
	return true
}

func batchedEmbeddingNeeds(cfg *config.RouterConfig, qwen3Path string) (bool, bool) {
	semanticCacheNeedsBatched := cfg.Enabled &&
		strings.ToLower(strings.TrimSpace(cfg.EmbeddingModel)) == "qwen3" &&
		qwen3Path != ""
	mlSelectionNeedsBatched := cfg.ModelSelection.Enabled &&
		cfg.ModelSelection.ML.ModelsPath != "" &&
		cfg.Qwen3ModelPath != ""
	return semanticCacheNeedsBatched, mlSelectionNeedsBatched
}

func logBatchedEmbeddingNeeds(component string, semanticCacheNeedsBatched bool, mlSelectionNeedsBatched bool) {
	if !semanticCacheNeedsBatched && !mlSelectionNeedsBatched {
		return
	}
	logging.ComponentDebugEvent(component, "batched_embedding_mode_required", map[string]interface{}{
		"semantic_cache":  semanticCacheNeedsBatched,
		"model_selection": mlSelectionNeedsBatched,
	})
}

func initUnifiedEmbeddingModelFactory(paths embeddingPaths, useBatched bool) error {
	if !useBatched {
		return initEmbeddingModels(paths.qwen3, paths.gemma, paths.mmBert, paths.useCPU)
	}

	if err := initEmbeddingModelsBatched(paths.qwen3, 64, 10, paths.useCPU); err != nil {
		return err
	}
	return initEmbeddingModels(paths.qwen3, paths.gemma, paths.mmBert, paths.useCPU)
}

func initializeBERTModel(component string, useCPU bool, bertPath string, eventPrefix string) bool {
	if bertPath == "" {
		return false
	}

	logging.ComponentEvent(component, eventPrefix+"_init_started", map[string]interface{}{
		"model_ref": bertPath,
		"use_cpu":   useCPU,
	})
	if err := initModel(bertPath, useCPU); err != nil {
		logging.ComponentWarnEvent(component, eventPrefix+"_init_failed", map[string]interface{}{
			"model_ref": bertPath,
			"error":     err.Error(),
		})
		return false
	}
	logging.ComponentEvent(component, eventPrefix+"_initialized", map[string]interface{}{
		"model_ref": bertPath,
	})
	return true
}

func initializeMultiModalEmbeddingModel(component string, useCPU bool, multiModalPath string) bool {
	if multiModalPath == "" {
		return false
	}

	logging.ComponentEvent(component, "multimodal_embedding_init_started", map[string]interface{}{
		"model_ref": multiModalPath,
		"use_cpu":   useCPU,
	})
	if err := initMultiModalEmbeddingModel(multiModalPath, useCPU); err != nil {
		logging.ComponentWarnEvent(component, "multimodal_embedding_init_failed", map[string]interface{}{
			"model_ref":               multiModalPath,
			"error":                   err.Error(),
			"multimodal_routes_ready": false,
		})
		return false
	}
	logging.ComponentEvent(component, "multimodal_embedding_initialized", map[string]interface{}{
		"model_ref": multiModalPath,
	})
	return true
}
