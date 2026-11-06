package config

// ModelSelectionConfig represents configuration for advanced model selection algorithms.
type ModelSelectionConfig struct {
	// Method specifies the selection algorithm to use.
	Method string `yaml:"method,omitempty"`

	// Enabled indicates if model selection is enabled.
	Enabled bool `yaml:"enabled,omitempty"`

	// Family-specific configuration blocks.
	Elo      EloSelectionConfig      `yaml:"-"`
	RouterDC RouterDCSelectionConfig `yaml:"router_dc,omitempty"`
	AutoMix  AutoMixSelectionConfig  `yaml:"automix,omitempty"`
	Hybrid   HybridSelectionConfig   `yaml:"hybrid,omitempty"`
	ML       MLSelectionConfig       `yaml:"ml,omitempty"`

	// LookupTables configures persisted lookup tables for session-aware routing.
	LookupTables LookupTableConfig `yaml:"-"`
}

// LookupTableConfig configures session-routing lookup tables that replace
// hardcoded constants with data-driven values.
type LookupTableConfig struct {
	// Enabled activates lookup table resolution during model selection.
	Enabled bool `yaml:"enabled,omitempty"`

	// StoragePath is the path to the JSON file used for persistence.
	// When empty, an in-memory backend is used (data lost on restart).
	StoragePath string `yaml:"storage_path,omitempty"`

	// AutoSaveInterval is the interval at which dirty entries are flushed to
	// disk (e.g. "5m"). Requires StoragePath to be set.
	AutoSaveInterval string `yaml:"auto_save_interval,omitempty"`

	// PopulateFromReplay enables automatic derivation of lookup table entries
	// from router replay records. Requires a replay store to be configured.
	PopulateFromReplay bool `yaml:"populate_from_replay,omitempty"`

	// PopulateInterval is how often to re-derive entries from the replay store
	// (e.g. "15m"). When empty, entries are only derived once at startup.
	// Requires PopulateFromReplay to be true.
	PopulateInterval string `yaml:"populate_interval,omitempty"`

	// QualityGaps are manual overrides for quality_gap entries.
	// Override values take precedence over replay-derived values.
	QualityGaps []QualityGapOverride `yaml:"quality_gaps,omitempty"`

	// HandoffPenalties are manual overrides for handoff_penalty entries.
	HandoffPenalties []HandoffPenaltyOverride `yaml:"handoff_penalties,omitempty"`

	// RemainingTurnPriors are manual overrides for remaining_turn_prior entries.
	RemainingTurnPriors []RemainingTurnPriorOverride `yaml:"remaining_turn_priors,omitempty"`
}

// QualityGapOverride manually sets a quality_gap entry.
type QualityGapOverride struct {
	TaskFamily     string  `yaml:"task_family"`
	CurrentModel   string  `yaml:"current_model"`
	CandidateModel string  `yaml:"candidate_model"`
	Value          float64 `yaml:"value"`
}

// HandoffPenaltyOverride manually sets a handoff_penalty entry.
type HandoffPenaltyOverride struct {
	FromModel string  `yaml:"from_model"`
	ToModel   string  `yaml:"to_model"`
	Value     float64 `yaml:"value"`
}

// RemainingTurnPriorOverride manually sets a remaining_turn_prior entry.
type RemainingTurnPriorOverride struct {
	IntentOrDomain string  `yaml:"intent_or_domain"`
	Value          float64 `yaml:"value"`
}

// MLSelectionConfig holds configuration for the shared ML-based selectors.
type MLSelectionConfig struct {
	ModelsPath   string         `yaml:"models_path,omitempty"`
	ModelType    string         `yaml:"model_type,omitempty"`
	EmbeddingDim int            `yaml:"embedding_dim,omitempty"`
	KNN          MLKNNConfig    `yaml:"knn,omitempty"`
	KMeans       MLKMeansConfig `yaml:"kmeans,omitempty"`
	SVM          MLSVMConfig    `yaml:"svm,omitempty"`
	MLP          MLMLPConfig    `yaml:"mlp,omitempty"`
}

type MLKNNConfig struct {
	K              int    `yaml:"k,omitempty"`
	PretrainedPath string `yaml:"pretrained_path,omitempty"`
}

type MLKMeansConfig struct {
	NumClusters      int     `yaml:"num_clusters,omitempty"`
	EfficiencyWeight float64 `yaml:"efficiency_weight,omitempty"`
	PretrainedPath   string  `yaml:"pretrained_path,omitempty"`
}

type MLSVMConfig struct {
	Kernel         string  `yaml:"kernel,omitempty"`
	Gamma          float64 `yaml:"gamma,omitempty"`
	PretrainedPath string  `yaml:"pretrained_path,omitempty"`
}

type MLMLPConfig struct {
	Device         string `yaml:"device,omitempty"`
	PretrainedPath string `yaml:"pretrained_path,omitempty"`
}

type EloSelectionConfig struct {
	InitialRating     float64 `yaml:"initial_rating,omitempty"`
	KFactor           float64 `yaml:"k_factor,omitempty"`
	CategoryWeighted  bool    `yaml:"category_weighted,omitempty"`
	DecayFactor       float64 `yaml:"decay_factor,omitempty"`
	MinComparisons    int     `yaml:"min_comparisons,omitempty"`
	CostScalingFactor float64 `yaml:"cost_scaling_factor,omitempty"`
	StoragePath       string  `yaml:"storage_path,omitempty"`
	AutoSaveInterval  string  `yaml:"auto_save_interval,omitempty"`
}

type RouterDCSelectionConfig struct {
	Temperature         float64 `yaml:"temperature,omitempty"`
	DimensionSize       int     `yaml:"dimension_size,omitempty"`
	MinSimilarity       float64 `yaml:"min_similarity,omitempty"`
	UseQueryContrastive bool    `yaml:"use_query_contrastive"`
	UseModelContrastive bool    `yaml:"use_model_contrastive"`
	RequireDescriptions bool    `yaml:"require_descriptions"`
	UseCapabilities     bool    `yaml:"use_capabilities"`
}

type AutoMixSelectionConfig struct {
	VerificationThreshold  float64 `yaml:"verification_threshold,omitempty"`
	MaxEscalations         int     `yaml:"max_escalations,omitempty"`
	CostAwareRouting       bool    `yaml:"cost_aware_routing,omitempty"`
	CostQualityTradeoff    float64 `yaml:"cost_quality_tradeoff,omitempty"`
	DiscountFactor         float64 `yaml:"discount_factor,omitempty"`
	UseLogprobVerification bool    `yaml:"use_logprob_verification,omitempty"`
}

type HybridSelectionConfig struct {
	ExperienceWeight    float64 `yaml:"experience_weight,omitempty"`
	RouterDCWeight      float64 `yaml:"router_dc_weight,omitempty"`
	AutoMixWeight       float64 `yaml:"automix_weight,omitempty"`
	CostWeight          float64 `yaml:"cost_weight,omitempty"`
	QualityGapThreshold float64 `yaml:"quality_gap_threshold,omitempty"`
	NormalizeScores     bool    `yaml:"normalize_scores"`
}

// MultiFactorSelectionConfig configures the multi_factor selector, which
// compares quality/latency/cost/load with a weighted or ordered objective.
type MultiFactorSelectionConfig struct {
	Objective *MultiFactorObjectiveConfig `yaml:"objective,omitempty"`
	Weights   *MultiFactorWeightsConfig   `yaml:"weights,omitempty"`
	SLO       *MultiFactorSLOConfig       `yaml:"slo,omitempty"`
	Quality   *QualityEvidenceConfig      `yaml:"quality,omitempty"`

	// ExpectedOutputTokens is a cost forecast, not a generation limit.
	ExpectedOutputTokens *int `yaml:"expected_output_tokens,omitempty" jsonschema:"minimum=1"`

	// LatencyPercentile selects which percentile (e.g. 95) is read from
	// pkg/latency when computing the latency signal. Defaults to 95.
	LatencyPercentile int `yaml:"latency_percentile,omitempty"`
	// LatencyMetric compares one consistent measurement across candidates.
	// Omission preserves the legacy TPOT-then-TTFT fallback.
	LatencyMetric string `yaml:"latency_metric,omitempty"`

	// OnNoCandidates controls behavior when SLO filtering removes every
	// candidate. Valid values: "cheapest" (default), "first", "fail".
	OnNoCandidates string `yaml:"on_no_candidates,omitempty"`
}

// QualityEvidenceConfig selects the versioned catalog index used as the
// multi_factor quality signal. Capability indices and the overall index share
// this contract, so a decision can be capability-aware without inventing a
// partial overall score.
type QualityEvidenceConfig struct {
	Index       string   `yaml:"index"`
	OnMissing   string   `yaml:"on_missing,omitempty"`
	MinCoverage float64  `yaml:"min_coverage,omitempty"`
	MinScore    *float64 `yaml:"min_score,omitempty"`
}

const (
	QualityEvidenceOnMissingExclude = "exclude"
	QualityEvidenceOnMissingDisable = "disable_quality"

	MultiFactorObjectiveWeighted      = "weighted"
	MultiFactorObjectiveLexicographic = "lexicographic"

	MultiFactorFactorQuality = "quality"
	MultiFactorFactorLatency = "latency"
	MultiFactorFactorCost    = "cost"
	MultiFactorFactorLoad    = "load"
)

// MultiFactorObjectiveConfig chooses how factor utilities are compared. The
// weighted strategy uses Weights. Lexicographic compares priorities in order,
// retaining candidates within each relative tolerance before considering the
// next factor.
type MultiFactorObjectiveConfig struct {
	Strategy   string                      `yaml:"strategy,omitempty"`
	Priorities []MultiFactorPriorityConfig `yaml:"priorities,omitempty"`
}

// MultiFactorPriorityConfig is one lexicographic comparison step. Tolerance is
// the allowed relative gap from the best observed value (0.02 means 2%).
type MultiFactorPriorityConfig struct {
	Factor    string  `yaml:"factor"`
	Tolerance float64 `yaml:"tolerance,omitempty"`
}

// MultiFactorWeightsConfig holds per-signal weights for the multi_factor
// scoring formula score = w_q*quality + w_l*latency + w_c*cost + w_L*load.
// Weights are normalized to sum to 1 if they do not. All four default to 0.25.
type MultiFactorWeightsConfig struct {
	Quality float64 `yaml:"quality,omitempty"`
	Latency float64 `yaml:"latency,omitempty"`
	Cost    float64 `yaml:"cost,omitempty"`
	Load    float64 `yaml:"load,omitempty"`
}

// MultiFactorSLOConfig sets hard ceilings that prune candidates before
// scoring. A zero value means "no ceiling" for that dimension.
type MultiFactorSLOConfig struct {
	MaxTPOTMs    float64 `yaml:"max_tpot_ms,omitempty"`
	MaxTTFTMs    float64 `yaml:"max_ttft_ms,omitempty"`
	MaxCostPer1M float64 `yaml:"max_cost_per_1m,omitempty"`
	MaxInflight  int     `yaml:"max_inflight,omitempty"`
}

// LatencyAwareAlgorithmConfig configures TPOT/TTFT percentile routing policies.
type LatencyAwareAlgorithmConfig struct {
	TPOTPercentile int    `yaml:"tpot_percentile,omitempty"`
	TTFTPercentile int    `yaml:"ttft_percentile,omitempty"`
	Description    string `yaml:"description,omitempty"`
}

// MLModelSelectionConfig configures the ML-based algorithm used from per-decision policies.
type MLModelSelectionConfig struct {
	Type             string   `yaml:"type"`
	ModelsPath       string   `yaml:"models_path,omitempty"`
	K                int      `yaml:"k,omitempty"`
	NumClusters      int      `yaml:"num_clusters,omitempty"`
	Kernel           string   `yaml:"kernel,omitempty"`
	Gamma            float64  `yaml:"gamma,omitempty"`
	EfficiencyWeight *float64 `yaml:"efficiency_weight,omitempty"`
	Device           string   `yaml:"device,omitempty"`
}
