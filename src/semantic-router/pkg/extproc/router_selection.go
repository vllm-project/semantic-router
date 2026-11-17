package extproc

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/lookuptable"
)

// createModelSelectorRegistries leaves publication to publishRouterState.
func createModelSelectorRegistries(cfg *config.RouterConfig, replayReader store.Reader) (map[config.RecipeName]*selection.Registry, *selection.Registry, lookuptable.LookupTableStorage, func()) {
	lt, cancel := buildLookupTable(cfg, replayReader)
	registries := make(map[config.RecipeName]*selection.Registry)

	if len(cfg.Recipes) == 0 {
		registry := createModelSelectorRegistry(cfg)
		registries[cfg.DefaultRecipe().Name] = registry
		return registries, registry, lt, cancel
	}

	for i := range cfg.Recipes {
		recipe := &cfg.Recipes[i]
		registries[recipe.Name] = createModelSelectorRegistry(cfg.ConfigForRecipe(recipe))
	}
	defaultRegistry := registries[cfg.DefaultRecipe().Name]
	return registries, defaultRegistry, lt, cancel
}

func createModelSelectorRegistry(cfg *config.RouterConfig) *selection.Registry {
	backendModels := cfg.BackendModels
	selectionFactory := selection.NewFactory()

	if backendModels.ModelConfig != nil {
		selectionFactory = selectionFactory.WithModelConfig(backendModels.ModelConfig)
	}
	if len(cfg.Categories) > 0 {
		selectionFactory = selectionFactory.WithCategories(cfg.Categories)
	}

	registry := selectionFactory.CreateAll()

	// Collect algorithm methods actually configured in decisions
	configuredMethods := collectConfiguredAlgorithmMethods(cfg)

	// Warn about experimental algorithms and check dependency health
	selection.WarnExperimentalAlgorithms(registry, configuredMethods)
	selection.CheckDependencyHealth(registry, configuredMethods)

	logging.ComponentEvent("extproc", "model_selection_registry_initialized", map[string]interface{}{
		"mode": "per_decision_algorithm_config",
	})
	return registry
}

func collectConfiguredAlgorithmMethods(cfg *config.RouterConfig) []selection.SelectionMethod {
	seen := make(map[string]bool)
	var methods []selection.SelectionMethod

	for _, decision := range cfg.AllRoutingDecisions() {
		if decision.Algorithm == nil || decision.Algorithm.Type == "" {
			continue
		}
		if !seen[decision.Algorithm.Type] {
			seen[decision.Algorithm.Type] = true
			methods = append(methods, selection.SelectionMethod(decision.Algorithm.Type))
		}
	}

	return methods
}

func buildMultiFactorSelectionConfig(decisionCfg *config.MultiFactorSelectionConfig) *selection.MultiFactorConfig {
	result := selection.DefaultMultiFactorConfig()
	if decisionCfg == nil {
		return result
	}

	result.ExpectedOutputTokens = decisionCfg.ExpectedOutputTokens
	if decisionCfg.Weights != nil {
		result.Weights = selection.MultiFactorWeights{
			Quality: decisionCfg.Weights.Quality,
			Latency: decisionCfg.Weights.Latency,
			Cost:    decisionCfg.Weights.Cost,
			Load:    decisionCfg.Weights.Load,
		}
	}
	if decisionCfg.Objective != nil {
		result.Objective = selection.MultiFactorObjective{
			Strategy:   decisionCfg.Objective.Strategy,
			Priorities: make([]selection.MultiFactorPriority, 0, len(decisionCfg.Objective.Priorities)),
		}
		for _, priority := range decisionCfg.Objective.Priorities {
			result.Objective.Priorities = append(result.Objective.Priorities, selection.MultiFactorPriority{
				Factor: priority.Factor, Tolerance: priority.Tolerance,
			})
		}
	}
	if decisionCfg.SLO != nil {
		result.SLO = selection.MultiFactorSLO{
			MaxTPOTMs:    decisionCfg.SLO.MaxTPOTMs,
			MaxTTFTMs:    decisionCfg.SLO.MaxTTFTMs,
			MaxCostPer1M: decisionCfg.SLO.MaxCostPer1M,
			MaxInflight:  decisionCfg.SLO.MaxInflight,
		}
	}
	if decisionCfg.Quality != nil {
		result.QualityIndex = decisionCfg.Quality.Index
		result.QualityOnMissing = decisionCfg.Quality.OnMissing
		result.QualityMinCoverage = decisionCfg.Quality.MinCoverage
		if decisionCfg.Quality.MinScore != nil {
			value := *decisionCfg.Quality.MinScore
			result.QualityMinScore = &value
		}
		if result.QualityOnMissing == "" {
			result.QualityOnMissing = config.QualityEvidenceOnMissingExclude
		}
	}
	if decisionCfg.LatencyPercentile != 0 {
		result.LatencyPercentile = decisionCfg.LatencyPercentile
	}
	if decisionCfg.LatencyMetric != "" {
		result.LatencyMetric = decisionCfg.LatencyMetric
	}
	if decisionCfg.OnNoCandidates != "" {
		result.OnNoCandidates = decisionCfg.OnNoCandidates
	}
	return result
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
		for i := len(cancelFuncs) - 1; i >= 0; i-- {
			cancelFuncs[i]()
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
		if interval, err := config.ParsePeriodicInterval(ltCfg.AutoSaveInterval, 0); err == nil {
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
	var interval time.Duration
	if ltCfg.PopulateInterval != "" {
		var err error
		interval, err = config.ParsePeriodicInterval(ltCfg.PopulateInterval, 0)
		if err != nil {
			logging.Warnf("[RouterSelection] Invalid lookup table populate_interval %q: %v", ltCfg.PopulateInterval, err)
			interval = 0
		}
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
func populateFromReplay(ctx context.Context, storage lookuptable.LookupTableStorage, reader store.Reader) {
	records, err := reader.List(ctx)
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

// startLookupTablePopulator derives entries once and repeats when interval is positive.
func startLookupTablePopulator(storage lookuptable.LookupTableStorage, reader store.Reader, interval time.Duration) func() {
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	goSafely("lookup_table_populator", func() {
		defer close(done)
		populateFromReplay(ctx, storage, reader)
		if interval <= 0 {
			return
		}
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			default:
			}
			select {
			case <-ticker.C:
				populateFromReplay(ctx, storage, reader)
			case <-ctx.Done():
				return
			}
		}
	})
	if interval > 0 {
		logging.ComponentEvent("extproc", "lookuptable_populator_started", map[string]interface{}{
			"interval": interval.String(),
		})
	}
	return func() {
		cancel()
		<-done
	}
}

func closeRecipeModelSelectors(registries map[config.RecipeName]*selection.Registry) error {
	var errs []error
	closed := make(map[*selection.Registry]bool, len(registries))
	for recipeName, registry := range registries {
		if registry == nil || closed[registry] {
			continue
		}
		closed[registry] = true
		if err := registry.Close(); err != nil {
			errs = append(errs, fmt.Errorf("closing selection registry for routing recipe %q: %w", recipeName, err))
		}
	}
	return errors.Join(errs...)
}
