package extproc

import (
	"context"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/lookuptable"
)

func createModelSelectorRegistry(cfg *config.RouterConfig, replayReader store.Reader) (*selection.Registry, lookuptable.LookupTableStorage, func()) {
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

	lt, cancel := buildLookupTable(cfg, replayReader)
	if lt != nil {
		selectionFactory = selectionFactory.WithLookupTable(lt)
	}

	registry := selectionFactory.CreateAll()
	selection.GlobalRegistry = registry
	logging.ComponentEvent("extproc", "model_selection_registry_initialized", map[string]interface{}{
		"mode": "per_decision_algorithm_config",
	})
	return registry, lt, cancel
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

	logging.ComponentEvent("extproc", "ml_model_selection_enabled", map[string]interface{}{
		"models_path":       mlCfg.ModelsPath,
		"embedding_dim":     mlCfg.EmbeddingDim,
		"knn_pretrained":    mlCfg.KNN.PretrainedPath != "",
		"kmeans_pretrained": mlCfg.KMeans.PretrainedPath != "",
		"svm_pretrained":    mlCfg.SVM.PretrainedPath != "",
		"mlp_pretrained":    mlCfg.MLP.PretrainedPath != "",
	})
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

// buildLookupTable constructs a LookupTable from the router config.
// Returns (nil, nil) when lookup tables are disabled or not configured.
func buildLookupTable(cfg *config.RouterConfig, replayReader store.Reader) (lookuptable.LookupTableStorage, func()) {
	ltCfg := cfg.ModelSelection.LookupTables
	if !ltCfg.Enabled {
		return nil, nil
	}

	storage, cancelStorage := buildLookupTableStorage(ltCfg)
	var cancelFuncs []func()
	if cancelStorage != nil {
		cancelFuncs = append(cancelFuncs, cancelStorage)
	}

	maybePopulateFromReplay(ltCfg, storage, replayReader, &cancelFuncs)
	applyLookupTableOverrides(ltCfg, storage)

	logging.ComponentEvent("extproc", "lookuptable_initialized", map[string]interface{}{
		"storage_path":         ltCfg.StoragePath,
		"entry_count":          len(storage.All()),
		"populate_from_replay": ltCfg.PopulateFromReplay,
		"has_overrides":        len(ltCfg.QualityGaps)+len(ltCfg.HandoffPenalties)+len(ltCfg.RemainingTurnPriors) > 0,
	})

	cancel := func() {
		for _, f := range cancelFuncs {
			f()
		}
	}
	return storage, cancel
}

// buildLookupTableStorage creates the storage backend (file or memory) and
// returns a cancel function for any background goroutines it starts.
func buildLookupTableStorage(ltCfg config.LookupTableConfig) (lookuptable.LookupTableStorage, func()) {
	if ltCfg.StoragePath == "" {
		return lookuptable.NewMemoryStorage(), nil
	}

	fs, err := lookuptable.NewFileStorage(ltCfg.StoragePath)
	if err != nil {
		logging.Errorf("[RouterSelection] Failed to create lookup table file storage: %v", err)
		return lookuptable.NewMemoryStorage(), nil
	}

	if err := fs.Load(); err != nil {
		logging.Warnf("[RouterSelection] Failed to load lookup table from %s: %v", ltCfg.StoragePath, err)
	}

	cancel := func() { _ = fs.Close() }
	if ltCfg.AutoSaveInterval != "" {
		if interval, err := time.ParseDuration(ltCfg.AutoSaveInterval); err == nil {
			fs.StartAutoSave(interval)
		} else {
			logging.Warnf("[RouterSelection] Invalid lookup table auto_save_interval %q: %v", ltCfg.AutoSaveInterval, err)
		}
	}
	return fs, cancel
}

// maybePopulateFromReplay starts async replay derivation and periodic
// re-population when configured.
func maybePopulateFromReplay(
	ltCfg config.LookupTableConfig,
	storage lookuptable.LookupTableStorage,
	reader store.Reader,
	cancelFuncs *[]func(),
) {
	if !ltCfg.PopulateFromReplay || reader == nil {
		return
	}
	go populateFromReplay(storage, reader)

	if ltCfg.PopulateInterval == "" {
		return
	}
	interval, err := time.ParseDuration(ltCfg.PopulateInterval)
	if err != nil {
		logging.Warnf("[RouterSelection] Invalid lookup table populate_interval %q: %v", ltCfg.PopulateInterval, err)
		return
	}
	*cancelFuncs = append(*cancelFuncs, startLookupTablePopulator(storage, reader, interval))
}

// applyLookupTableOverrides writes manual config values on top of derived ones.
func applyLookupTableOverrides(ltCfg config.LookupTableConfig, storage lookuptable.LookupTableStorage) {
	now := time.Now()
	for _, o := range ltCfg.QualityGaps {
		_ = storage.Set(lookuptable.QualityGapKey(o.TaskFamily, o.CurrentModel, o.CandidateModel),
			lookuptable.Entry{Value: o.Value, Source: lookuptable.SourceConfigOverride, UpdatedAt: now})
	}
	for _, o := range ltCfg.HandoffPenalties {
		_ = storage.Set(lookuptable.HandoffPenaltyKey(o.FromModel, o.ToModel),
			lookuptable.Entry{Value: o.Value, Source: lookuptable.SourceConfigOverride, UpdatedAt: now})
	}
	for _, o := range ltCfg.RemainingTurnPriors {
		_ = storage.Set(lookuptable.RemainingTurnPriorKey(o.IntentOrDomain),
			lookuptable.Entry{Value: o.Value, Source: lookuptable.SourceConfigOverride, UpdatedAt: now})
	}
}

// populateFromReplay fetches all records from the replay reader and runs the
// builder synchronously. Errors are logged but do not prevent startup.
func populateFromReplay(storage lookuptable.LookupTableStorage, reader store.Reader) {
	records, err := reader.List(context.Background())
	if err != nil {
		logging.Errorf("[RouterSelection] Failed to list replay records for lookup table population: %v", err)
		return
	}
	if len(records) == 0 {
		logging.Debugf("[RouterSelection] No replay records available for lookup table population")
		return
	}
	builder := lookuptable.NewBuilder(storage)
	if err := builder.PopulateFromRecords(records); err != nil {
		logging.Errorf("[RouterSelection] Failed to populate lookup table from replay records: %v", err)
		return
	}
	logging.ComponentEvent("extproc", "lookuptable_populated_from_replay", map[string]interface{}{
		"record_count": len(records),
		"entry_count":  len(storage.All()),
	})
}

// startLookupTablePopulator launches a background goroutine that periodically
// re-derives lookup table entries from the replay store.
// The returned cancel function stops the goroutine.
func startLookupTablePopulator(storage lookuptable.LookupTableStorage, reader store.Reader, interval time.Duration) func() {
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				populateFromReplay(storage, reader)
			case <-ctx.Done():
				return
			}
		}
	}()
	logging.ComponentEvent("extproc", "lookuptable_populator_started", map[string]interface{}{
		"interval": interval.String(),
	})
	return cancel
}
