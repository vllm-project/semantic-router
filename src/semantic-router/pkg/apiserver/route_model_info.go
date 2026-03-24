//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"net/http"
	"path"
	"runtime"
	"strings"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

func (s *ClassificationAPIServer) handleModelsInfo(w http.ResponseWriter, _ *http.Request) {
	response := s.buildModelsInfoResponse()
	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleEmbeddingModelsInfo handles GET /api/v1/embeddings/models
// Returns ONLY embedding models information
func (s *ClassificationAPIServer) handleEmbeddingModelsInfo(w http.ResponseWriter, r *http.Request) {
	embeddingModels := s.getEmbeddingModelsInfo(s.loadModelsRuntimeState())

	response := map[string]interface{}{
		"models": embeddingModels,
		"count":  len(embeddingModels),
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) handleClassifierInfo(w http.ResponseWriter, _ *http.Request) {
	cfg := s.currentConfig()
	if cfg == nil {
		s.writeJSONResponse(w, http.StatusOK, map[string]interface{}{
			"status": "no_config",
			"config": nil,
		})
		return
	}

	// Return the config directly
	s.writeJSONResponse(w, http.StatusOK, map[string]interface{}{
		"status": "config_loaded",
		"config": jsonCompatibleValue(cfg),
	})
}

type classifierModelAvailability struct {
	core                   bool
	factCheck              bool
	hallucination          bool
	hallucinationExplainer bool
	feedback               bool
}

// buildModelsInfoResponse builds the models info response
func (s *ClassificationAPIServer) buildModelsInfoResponse() ModelsInfoResponse {
	runtimeState := s.loadModelsRuntimeState()
	models := s.getClassifierModelsInfo(s.classifierModelAvailability(), runtimeState)

	// Add embedding models information
	embeddingModels := s.getEmbeddingModelsInfo(runtimeState)
	models = append(models, embeddingModels...)

	// Get system information
	systemInfo := s.getSystemInfo()

	return ModelsInfoResponse{
		Models:  models,
		Summary: buildModelsInfoSummary(runtimeState, models),
		System:  systemInfo,
	}
}

func (s *ClassificationAPIServer) classifierModelAvailability() classifierModelAvailability {
	if s == nil || s.classificationSvc == nil {
		return classifierModelAvailability{}
	}

	return classifierModelAvailability{
		core:                   s.classificationSvc.HasClassifier(),
		factCheck:              s.classificationSvc.HasFactCheckClassifier(),
		hallucination:          s.classificationSvc.HasHallucinationDetector(),
		hallucinationExplainer: s.classificationSvc.HasHallucinationExplainer(),
		feedback:               s.classificationSvc.HasFeedbackDetector(),
	}
}

// getClassifierModelsInfo returns information about configured classifier models.
func (s *ClassificationAPIServer) getClassifierModelsInfo(
	availability classifierModelAvailability,
	runtimeState *startupstatus.State,
) []ModelInfo {
	cfg := s.currentConfig()
	if cfg == nil {
		return s.getPlaceholderModelsInfo(runtimeState)
	}

	models := appendConfiguredModels(nil, cfg, availability)

	for i := range models {
		models[i] = enrichModelInfo(models[i], runtimeState)
	}

	return models
}

func appendConfiguredModels(
	models []ModelInfo,
	cfg *routerconfig.RouterConfig,
	availability classifierModelAvailability,
) []ModelInfo {
	models = append(models, buildRoutingClassifierModels(cfg, availability)...)
	models = append(models, buildHallucinationModels(cfg, availability)...)
	models = append(models, buildFeedbackAndSimilarityModels(cfg, availability)...)
	return models
}

func buildRoutingClassifierModels(
	cfg *routerconfig.RouterConfig,
	availability classifierModelAvailability,
) []ModelInfo {
	var models []ModelInfo
	categoryModel := cfg.CategoryModel
	if cfg.IsCategoryClassifierEnabled() {
		models = append(models, ModelInfo{
			Name:       "category_classifier",
			Type:       "intent_classification",
			Loaded:     availability.core,
			ModelPath:  categoryModel.ModelID,
			Categories: configuredCategoryNames(cfg),
			Metadata: map[string]string{
				"mapping_path": categoryModel.CategoryMappingPath,
				"model_type":   resolveInlineModelType(categoryModel.UseMmBERT32K, categoryModel.UseModernBERT, false),
				"threshold":    fmt.Sprintf("%.2f", categoryModel.Threshold),
			},
		})
	}

	piiModel := cfg.PIIModel
	if cfg.IsPIIClassifierEnabled() {
		models = append(models, ModelInfo{
			Name:      "pii_classifier",
			Type:      "pii_detection",
			Loaded:    availability.core,
			ModelPath: piiModel.ModelID,
			Metadata: map[string]string{
				"mapping_path": piiModel.PIIMappingPath,
				"model_type":   resolveInlineModelType(piiModel.UseMmBERT32K, false, true),
				"threshold":    fmt.Sprintf("%.2f", piiModel.Threshold),
			},
		})
	}

	promptGuard := cfg.PromptGuard
	if cfg.IsPromptGuardEnabled() {
		models = append(models, ModelInfo{
			Name:      "jailbreak_classifier",
			Type:      "security_detection",
			Loaded:    availability.core,
			ModelPath: promptGuard.ModelID,
			Metadata: map[string]string{
				"enabled":                "true",
				"jailbreak_mapping_path": promptGuard.JailbreakMappingPath,
				"model_type":             resolveInlineModelType(promptGuard.UseMmBERT32K, promptGuard.UseModernBERT, false),
			},
		})
	}

	return models
}

func buildHallucinationModels(
	cfg *routerconfig.RouterConfig,
	availability classifierModelAvailability,
) []ModelInfo {
	var models []ModelInfo
	factCheckModel := cfg.HallucinationMitigation.FactCheckModel
	if cfg.IsFactCheckClassifierEnabled() {
		models = append(models, ModelInfo{
			Name:      "fact_check_classifier",
			Type:      "fact_check_classification",
			Loaded:    availability.factCheck,
			ModelPath: factCheckModel.ModelID,
			Metadata: map[string]string{
				"model_type": resolveInlineModelType(factCheckModel.UseMmBERT32K, false, false),
				"threshold":  fmt.Sprintf("%.2f", factCheckModel.Threshold),
				"use_cpu":    fmt.Sprintf("%t", factCheckModel.UseCPU),
			},
		})
	}

	if !cfg.IsHallucinationModelEnabled() {
		return models
	}

	hallucinationModel := cfg.HallucinationMitigation.HallucinationModel
	models = append(models, ModelInfo{
		Name:      "hallucination_detector",
		Type:      "hallucination_detection",
		Loaded:    availability.hallucination,
		ModelPath: hallucinationModel.ModelID,
		Metadata: map[string]string{
			"model_type":            "modernbert",
			"threshold":             fmt.Sprintf("%.2f", hallucinationModel.Threshold),
			"min_span_length":       fmt.Sprintf("%d", hallucinationModel.MinSpanLength),
			"min_span_confidence":   fmt.Sprintf("%.2f", hallucinationModel.MinSpanConfidence),
			"context_window_size":   fmt.Sprintf("%d", hallucinationModel.ContextWindowSize),
			"nli_filtering_enabled": fmt.Sprintf("%t", hallucinationModel.EnableNLIFiltering),
			"use_cpu":               fmt.Sprintf("%t", hallucinationModel.UseCPU),
		},
	})

	nliModel := cfg.HallucinationMitigation.NLIModel
	if nliModel.ModelID != "" {
		models = append(models, ModelInfo{
			Name:      "hallucination_explainer",
			Type:      "nli_explainer",
			Loaded:    availability.hallucinationExplainer,
			ModelPath: nliModel.ModelID,
			Metadata: map[string]string{
				"model_type": "modernbert_nli",
				"threshold":  fmt.Sprintf("%.2f", nliModel.Threshold),
				"use_cpu":    fmt.Sprintf("%t", nliModel.UseCPU),
			},
		})
	}

	return models
}

func buildFeedbackAndSimilarityModels(
	cfg *routerconfig.RouterConfig,
	availability classifierModelAvailability,
) []ModelInfo {
	var models []ModelInfo
	feedbackModel := cfg.FeedbackDetector
	if cfg.IsFeedbackDetectorEnabled() {
		models = append(models, ModelInfo{
			Name:      "feedback_detector",
			Type:      "feedback_detection",
			Loaded:    availability.feedback,
			ModelPath: feedbackModel.ModelID,
			Metadata: map[string]string{
				"model_type": resolveInlineModelType(feedbackModel.UseMmBERT32K, feedbackModel.UseModernBERT, false),
				"threshold":  fmt.Sprintf("%.2f", feedbackModel.Threshold),
				"use_cpu":    fmt.Sprintf("%t", feedbackModel.UseCPU),
			},
		})
	}

	bertModelPath := cfg.BertModelPath
	if bertModelPath != "" {
		models = append(models, ModelInfo{
			Name:      "bert_similarity_model",
			Type:      "similarity",
			Loaded:    availability.core,
			ModelPath: bertModelPath,
			Metadata: map[string]string{
				"model_type": "sentence_transformer",
				"threshold":  fmt.Sprintf("%.2f", cfg.MinSimilarityThreshold()),
				"use_cpu":    fmt.Sprintf("%t", cfg.UseCPU),
			},
		})
	}

	return models
}

func configuredCategoryNames(cfg *routerconfig.RouterConfig) []string {
	categories := make([]string, 0, len(cfg.Categories))
	for _, cat := range cfg.Categories {
		categories = append(categories, cat.Name)
	}
	return categories
}

// getPlaceholderModelsInfo returns placeholder model information
func (s *ClassificationAPIServer) getPlaceholderModelsInfo(runtimeState *startupstatus.State) []ModelInfo {
	models := []ModelInfo{
		{
			Name:   "category_classifier",
			Type:   "intent_classification",
			Loaded: false,
			Metadata: map[string]string{
				"status": "not_initialized",
			},
		},
		{
			Name:   "pii_classifier",
			Type:   "pii_detection",
			Loaded: false,
			Metadata: map[string]string{
				"status": "not_initialized",
			},
		},
		{
			Name:   "jailbreak_classifier",
			Type:   "security_detection",
			Loaded: false,
			Metadata: map[string]string{
				"status": "not_initialized",
			},
		},
		{
			Name:   "fact_check_classifier",
			Type:   "fact_check_classification",
			Loaded: false,
			Metadata: map[string]string{
				"status": "not_initialized",
			},
		},
		{
			Name:   "hallucination_detector",
			Type:   "hallucination_detection",
			Loaded: false,
			Metadata: map[string]string{
				"status": "not_initialized",
			},
		},
		{
			Name:   "hallucination_explainer",
			Type:   "nli_explainer",
			Loaded: false,
			Metadata: map[string]string{
				"status": "not_initialized",
			},
		},
		{
			Name:   "feedback_detector",
			Type:   "feedback_detection",
			Loaded: false,
			Metadata: map[string]string{
				"status": "not_initialized",
			},
		},
	}

	for i := range models {
		models[i] = enrichModelInfo(models[i], runtimeState)
	}

	return models
}

// getSystemInfo returns system information
func (s *ClassificationAPIServer) getSystemInfo() SystemInfo {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	return SystemInfo{
		GoVersion:    runtime.Version(),
		Architecture: runtime.GOARCH,
		OS:           runtime.GOOS,
		MemoryUsage:  fmt.Sprintf("%.2f MB", float64(m.Alloc)/1024/1024),
		GPUAvailable: false, // TODO: Implement GPU detection
	}
}

// getEmbeddingModelsInfo returns information about loaded embedding models
func (s *ClassificationAPIServer) getEmbeddingModelsInfo(runtimeState *startupstatus.State) []ModelInfo {
	var models []ModelInfo

	// Query embedding models info from Rust FFI
	embeddingInfo, err := candle_binding.GetEmbeddingModelsInfo()
	if err != nil {
		logging.Warnf("Failed to get embedding models info: %v", err)
		return models
	}

	// Convert to ModelInfo format
	for _, model := range embeddingInfo.Models {
		modelPath := normalizeEmbeddingModelPath(model.ModelPath, model.ModelName)
		if modelPath == "" {
			modelPath = strings.TrimSpace(model.ModelPath)
		}
		if modelPath == "" {
			modelPath = strings.TrimSpace(model.ModelName)
		}

		models = append(models, ModelInfo{
			Name:      fmt.Sprintf("%s_embedding_model", model.ModelName),
			Type:      "embedding",
			Loaded:    model.IsLoaded,
			ModelPath: modelPath,
			Metadata: map[string]string{
				"model_type":           model.ModelName,
				"max_sequence_length":  fmt.Sprintf("%d", model.MaxSequenceLength),
				"default_dimension":    fmt.Sprintf("%d", model.DefaultDimension),
				"matryoshka_supported": "true",
			},
		})
	}

	for i := range models {
		models[i] = enrichModelInfo(models[i], runtimeState)
	}

	return models
}

func normalizeEmbeddingModelPath(runtimePath, modelName string) string {
	for _, candidate := range embeddingModelPathCandidates(runtimePath, modelName) {
		if spec := routerconfig.GetModelByPath(candidate); spec != nil {
			return spec.LocalPath
		}
	}

	return ""
}

func embeddingModelPathCandidates(values ...string) []string {
	seen := make(map[string]struct{})
	var candidates []string

	for _, value := range values {
		trimmed := strings.TrimSpace(value)
		if trimmed == "" {
			continue
		}

		candidates = appendEmbeddingModelPathCandidate(candidates, seen, trimmed)

		extracted := extractEmbeddingModelPath(trimmed)
		if extracted != "" {
			candidates = appendEmbeddingModelPathCandidate(candidates, seen, extracted)
		}
	}

	return candidates
}

func appendEmbeddingModelPathCandidate(
	candidates []string,
	seen map[string]struct{},
	value string,
) []string {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return candidates
	}

	if _, ok := seen[trimmed]; !ok {
		seen[trimmed] = struct{}{}
		candidates = append(candidates, trimmed)
	}

	base := path.Base(trimmed)
	if base == "." || base == "/" || base == trimmed {
		return candidates
	}

	if _, ok := seen[base]; !ok {
		seen[base] = struct{}{}
		candidates = append(candidates, base)
	}

	if !strings.HasPrefix(base, "models/") {
		modelsBase := "models/" + base
		if _, ok := seen[modelsBase]; !ok {
			seen[modelsBase] = struct{}{}
			candidates = append(candidates, modelsBase)
		}
	}

	return candidates
}

func extractEmbeddingModelPath(value string) string {
	if value == "" {
		return ""
	}

	const marker = "path="
	index := strings.Index(value, marker)
	if index == -1 {
		return ""
	}

	trimmed := value[index+len(marker):]
	if end := strings.IndexAny(trimmed, ",)"); end >= 0 {
		trimmed = trimmed[:end]
	}

	return strings.TrimSpace(trimmed)
}

func (s *ClassificationAPIServer) loadModelsRuntimeState() *startupstatus.State {
	if s == nil || s.configPath == "" {
		return nil
	}

	state, err := startupstatus.Load(startupstatus.StatusPathFromConfigPath(s.configPath))
	if err != nil {
		return nil
	}

	return state
}

func buildModelsInfoSummary(runtimeState *startupstatus.State, models []ModelInfo) ModelsInfoSummary {
	loadedModels := 0
	for _, model := range models {
		if model.Loaded {
			loadedModels++
		}
	}

	totalModels := len(models)
	summary := ModelsInfoSummary{
		Ready:        totalModels == 0 || loadedModels == totalModels,
		LoadedModels: loadedModels,
		TotalModels:  totalModels,
	}

	if runtimeState == nil {
		if summary.Ready {
			summary.Phase = "ready"
			summary.Message = "All known router models are ready."
		} else if totalModels > 0 {
			summary.Phase = "starting"
			summary.Message = "Router models are still initializing."
		}
		return summary
	}

	summary.Ready = runtimeState.Ready
	summary.Phase = runtimeState.Phase
	summary.Message = runtimeState.Message
	summary.DownloadingModel = runtimeState.DownloadingModel
	summary.PendingModels = runtimeState.PendingModels
	summary.UpdatedAt = runtimeState.UpdatedAt
	if runtimeState.TotalModels > summary.TotalModels {
		summary.TotalModels = runtimeState.TotalModels
	}
	if loadedModels == 0 && runtimeState.ReadyModels > 0 {
		summary.LoadedModels = runtimeState.ReadyModels
	}

	return summary
}

func enrichModelInfo(model ModelInfo, runtimeState *startupstatus.State) ModelInfo {
	resolvedPath := canonicalModelPath(model.ModelPath)
	if resolvedPath != "" && resolvedPath != model.ModelPath {
		model.ResolvedModelPath = resolvedPath
	}

	if registry := lookupModelRegistryInfo(model.ModelPath); registry != nil {
		model.Registry = registry
	}

	model.State = resolveModelState(model, runtimeState)
	return model
}

func resolveModelState(model ModelInfo, runtimeState *startupstatus.State) string {
	if model.Loaded {
		return "ready"
	}

	if runtimeState == nil {
		return "not_loaded"
	}

	modelPath := canonicalModelPath(model.ModelPath)
	downloadingPath := canonicalModelPath(runtimeState.DownloadingModel)
	if modelPath != "" && modelPath == downloadingPath {
		return "downloading"
	}

	for _, pending := range runtimeState.PendingModels {
		if modelPath != "" && modelPath == canonicalModelPath(pending) {
			return "pending"
		}
	}

	switch runtimeState.Phase {
	case "downloading_models":
		return "pending"
	case "checking_models", "initializing_models", "starting":
		return "initializing"
	default:
		return "not_loaded"
	}
}

func lookupModelRegistryInfo(modelPath string) *ModelRegistryInfo {
	resolvedPath := canonicalModelPath(modelPath)
	if resolvedPath == "" {
		return nil
	}

	return routerconfig.GetModelRegistryInfoByPath(resolvedPath)
}

func canonicalModelPath(modelPath string) string {
	if modelPath == "" {
		return ""
	}

	return routerconfig.ResolveModelPath(modelPath)
}

func resolveInlineModelType(useMmBERT32K, useModernBERT, tokenLevel bool) string {
	switch {
	case useMmBERT32K && tokenLevel:
		return "mmbert_32k_token"
	case useMmBERT32K:
		return "mmbert_32k"
	case useModernBERT && tokenLevel:
		return "modernbert_token"
	case useModernBERT:
		return "modernbert"
	case tokenLevel:
		return "bert_token"
	default:
		return "bert"
	}
}
