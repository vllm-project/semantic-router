package extproc

import (
	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func createModelSelectorRegistry(cfg *config.RouterConfig) *selection.Registry {
	modelSelectionCfg := buildModelSelectionConfig(cfg)
	backendModels := cfg.BackendModels
	selectionFactory := selection.NewFactory(modelSelectionCfg)

	if backendModels.ModelConfig != nil {
		selectionFactory = selectionFactory.WithModelConfig(backendModels.ModelConfig)
	}
	if len(cfg.Categories) > 0 {
		selectionFactory = selectionFactory.WithCategories(cfg.Categories)
	}
	selectionFactory = selectionFactory.WithEmbeddingFunc(func(text string) ([]float32, error) {
		output, err := candle_binding.GetEmbeddingBatched(text, "qwen3", 1024)
		if err != nil {
			return nil, err
		}
		return output.Embedding, nil
	})

	registry := selectionFactory.CreateAll()
	selection.GlobalRegistry = registry
	logging.ComponentEvent("extproc", "model_selection_registry_initialized", map[string]interface{}{
		"mode": "per_decision_algorithm_config",
	})
	return registry
}

func buildModelSelectionConfig(cfg *config.RouterConfig) *selection.ModelSelectionConfig {
	modelSelectionCfg := &selection.ModelSelectionConfig{
		Method: "static",
	}

	eloFromDecision, routerDCFromDecision := findDecisionScopedSelectionConfigs(cfg)
	modelSelectionCfg.Elo = buildEloSelectionConfig(cfg, eloFromDecision)
	modelSelectionCfg.RouterDC = buildRouterDCSelectionConfig(cfg, routerDCFromDecision)
	modelSelectionCfg.AutoMix = buildAutoMixSelectionConfig(cfg)
	modelSelectionCfg.Hybrid = buildHybridSelectionConfig(cfg)
	modelSelectionCfg.ML = buildMLSelectionConfig(cfg)
	return modelSelectionCfg
}

func findDecisionScopedSelectionConfigs(
	cfg *config.RouterConfig,
) (*config.EloSelectionConfig, *config.RouterDCSelectionConfig) {
	intelligentRouting := cfg.IntelligentRouting
	var eloFromDecision *config.EloSelectionConfig
	var routerDCFromDecision *config.RouterDCSelectionConfig

	for _, decision := range intelligentRouting.Decisions {
		if decision.Algorithm == nil {
			continue
		}
		if decision.Algorithm.Type == "elo" &&
			decision.Algorithm.Elo != nil &&
			eloFromDecision == nil {
			eloFromDecision = decision.Algorithm.Elo
		}
		if decision.Algorithm.Type == "router_dc" &&
			decision.Algorithm.RouterDC != nil &&
			routerDCFromDecision == nil {
			routerDCFromDecision = decision.Algorithm.RouterDC
		}
	}

	return eloFromDecision, routerDCFromDecision
}

func buildEloSelectionConfig(
	cfg *config.RouterConfig,
	decisionCfg *config.EloSelectionConfig,
) *selection.EloConfig {
	intelligentRouting := cfg.IntelligentRouting
	eloCfg := intelligentRouting.ModelSelection.Elo
	result := &selection.EloConfig{
		InitialRating:     eloCfg.InitialRating,
		KFactor:           eloCfg.KFactor,
		CategoryWeighted:  eloCfg.CategoryWeighted,
		DecayFactor:       eloCfg.DecayFactor,
		MinComparisons:    eloCfg.MinComparisons,
		CostScalingFactor: eloCfg.CostScalingFactor,
		StoragePath:       eloCfg.StoragePath,
		AutoSaveInterval:  eloCfg.AutoSaveInterval,
	}

	if decisionCfg == nil {
		return result
	}

	if decisionCfg.StoragePath != "" {
		result.StoragePath = decisionCfg.StoragePath
	}
	if decisionCfg.AutoSaveInterval != "" {
		result.AutoSaveInterval = decisionCfg.AutoSaveInterval
	}
	if decisionCfg.KFactor != 0 {
		result.KFactor = decisionCfg.KFactor
	}
	if decisionCfg.InitialRating != 0 {
		result.InitialRating = decisionCfg.InitialRating
	}
	result.CategoryWeighted = decisionCfg.CategoryWeighted
	return result
}

func buildRouterDCSelectionConfig(
	cfg *config.RouterConfig,
	decisionCfg *config.RouterDCSelectionConfig,
) *selection.RouterDCConfig {
	intelligentRouting := cfg.IntelligentRouting
	routerDCCfg := intelligentRouting.ModelSelection.RouterDC
	result := &selection.RouterDCConfig{
		Temperature:         routerDCCfg.Temperature,
		DimensionSize:       routerDCCfg.DimensionSize,
		MinSimilarity:       routerDCCfg.MinSimilarity,
		UseQueryContrastive: routerDCCfg.UseQueryContrastive,
		UseModelContrastive: routerDCCfg.UseModelContrastive,
		RequireDescriptions: routerDCCfg.RequireDescriptions,
		UseCapabilities:     routerDCCfg.UseCapabilities,
	}

	if decisionCfg == nil {
		return result
	}

	if decisionCfg.Temperature != 0 {
		result.Temperature = decisionCfg.Temperature
	}
	result.RequireDescriptions = decisionCfg.RequireDescriptions
	result.UseCapabilities = decisionCfg.UseCapabilities
	return result
}

func buildAutoMixSelectionConfig(cfg *config.RouterConfig) *selection.AutoMixConfig {
	intelligentRouting := cfg.IntelligentRouting
	autoMixCfg := intelligentRouting.ModelSelection.AutoMix
	return &selection.AutoMixConfig{
		VerificationThreshold:  autoMixCfg.VerificationThreshold,
		MaxEscalations:         autoMixCfg.MaxEscalations,
		CostAwareRouting:       autoMixCfg.CostAwareRouting,
		CostQualityTradeoff:    autoMixCfg.CostQualityTradeoff,
		DiscountFactor:         autoMixCfg.DiscountFactor,
		UseLogprobVerification: autoMixCfg.UseLogprobVerification,
	}
}

func buildHybridSelectionConfig(cfg *config.RouterConfig) *selection.HybridConfig {
	intelligentRouting := cfg.IntelligentRouting
	hybridCfg := intelligentRouting.ModelSelection.Hybrid
	return &selection.HybridConfig{
		EloWeight:           hybridCfg.EloWeight,
		RouterDCWeight:      hybridCfg.RouterDCWeight,
		AutoMixWeight:       hybridCfg.AutoMixWeight,
		CostWeight:          hybridCfg.CostWeight,
		QualityGapThreshold: hybridCfg.QualityGapThreshold,
		NormalizeScores:     hybridCfg.NormalizeScores,
	}
}

func buildMLSelectionConfig(cfg *config.RouterConfig) *selection.MLSelectorConfig {
	intelligentRouting := cfg.IntelligentRouting
	mlCfg := intelligentRouting.ModelSelection.ML
	if mlCfg.ModelsPath == "" &&
		mlCfg.KNN.PretrainedPath == "" &&
		mlCfg.KMeans.PretrainedPath == "" &&
		mlCfg.SVM.PretrainedPath == "" &&
		mlCfg.MLP.PretrainedPath == "" {
		return nil
	}

	logging.Infof("[Router] ML model selection enabled with models_path=%s", mlCfg.ModelsPath)
	return &selection.MLSelectorConfig{
		ModelsPath:   mlCfg.ModelsPath,
		EmbeddingDim: mlCfg.EmbeddingDim,
		KNN: &selection.KNNConfig{
			K:              mlCfg.KNN.K,
			PretrainedPath: mlCfg.KNN.PretrainedPath,
		},
		KMeans: &selection.KMeansConfig{
			NumClusters:      mlCfg.KMeans.NumClusters,
			EfficiencyWeight: mlCfg.KMeans.EfficiencyWeight,
			PretrainedPath:   mlCfg.KMeans.PretrainedPath,
		},
		SVM: &selection.SVMConfig{
			Kernel:         mlCfg.SVM.Kernel,
			Gamma:          mlCfg.SVM.Gamma,
			PretrainedPath: mlCfg.SVM.PretrainedPath,
		},
		MLP: &selection.MLPConfig{
			Device:         mlCfg.MLP.Device,
			PretrainedPath: mlCfg.MLP.PretrainedPath,
		},
	}
}
