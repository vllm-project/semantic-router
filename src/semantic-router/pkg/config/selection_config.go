package config

// ModelSelectionConfig represents configuration for advanced model selection algorithms.
type ModelSelectionConfig struct {
	// Method specifies the selection algorithm to use.
	Method string `yaml:"method,omitempty"`

	// Enabled indicates if model selection is enabled.
	Enabled bool `yaml:"enabled,omitempty"`

	// Family-specific configuration blocks.
	Elo      EloSelectionConfig      `yaml:"elo,omitempty"`
	RouterDC RouterDCSelectionConfig `yaml:"router_dc,omitempty"`
	AutoMix  AutoMixSelectionConfig  `yaml:"automix,omitempty"`
	Hybrid   HybridSelectionConfig   `yaml:"hybrid,omitempty"`
	ML       MLSelectionConfig       `yaml:"ml,omitempty"`

	Momentum MomentumSelectionConfig `yaml:"momentum,omitempty"`
	// LookupTables configures persisted lookup tables for session-aware routing.
	LookupTables LookupTableConfig `yaml:"lookup_tables,omitempty"`
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

// MomentumSelectionConfig configures Conversational Routing Momentum (CRM).
type MomentumSelectionConfig struct {
	Enabled   bool    `yaml:"enabled,omitempty"`
	Attack    float64 `yaml:"attack,omitempty"`
	Release   float64 `yaml:"release,omitempty"`
	Threshold float64 `yaml:"threshold,omitempty"`
}

// MLSelectionConfig holds configuration for the shared ML-based selectors.
type MLSelectionConfig struct {
	ModelsPath   string         `yaml:"models_path,omitempty"`
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
	UseQueryContrastive bool    `yaml:"use_query_contrastive,omitempty"`
	UseModelContrastive bool    `yaml:"use_model_contrastive,omitempty"`
	RequireDescriptions bool    `yaml:"require_descriptions,omitempty"`
	UseCapabilities     bool    `yaml:"use_capabilities,omitempty"`
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
	EloWeight           float64 `yaml:"elo_weight,omitempty"`
	RouterDCWeight      float64 `yaml:"router_dc_weight,omitempty"`
	AutoMixWeight       float64 `yaml:"automix_weight,omitempty"`
	CostWeight          float64 `yaml:"cost_weight,omitempty"`
	QualityGapThreshold float64 `yaml:"quality_gap_threshold,omitempty"`
	NormalizeScores     bool    `yaml:"normalize_scores,omitempty"`
}

// RLDrivenSelectionConfig configures Router-R1 style reinforcement-learning-based routing.
type RLDrivenSelectionConfig struct {
	ExplorationRate       float64 `yaml:"exploration_rate,omitempty"`
	UseThompsonSampling   bool    `yaml:"use_thompson_sampling,omitempty"`
	EnablePersonalization bool    `yaml:"enable_personalization,omitempty"`
	PersonalizationBlend  float64 `yaml:"personalization_blend,omitempty"`
	CostAwareness         bool    `yaml:"cost_awareness,omitempty"`
	CostWeight            float64 `yaml:"cost_weight,omitempty"`
	UseRouterR1Rewards    bool    `yaml:"use_router_r1_rewards,omitempty"`
	EnableLLMRouting      bool    `yaml:"enable_llm_routing,omitempty"`
	RouterR1ServerURL     string  `yaml:"router_r1_server_url,omitempty"`
}

// GMTRouterSelectionConfig configures graph-based personalized routing.
type GMTRouterSelectionConfig struct {
	EnablePersonalization             bool   `yaml:"enable_personalization,omitempty"`
	HistorySampleSize                 int    `yaml:"history_sample_size,omitempty"`
	MinInteractionsForPersonalization int    `yaml:"min_interactions_for_personalization,omitempty"`
	MaxInteractionsPerUser            int    `yaml:"max_interactions_per_user,omitempty"`
	ModelPath                         string `yaml:"model_path,omitempty"`
	StoragePath                       string `yaml:"storage_path,omitempty"`
}

// LatencyAwareAlgorithmConfig configures TPOT/TTFT percentile routing policies.
type LatencyAwareAlgorithmConfig struct {
	TPOTPercentile int    `yaml:"tpot_percentile,omitempty"`
	TTFTPercentile int    `yaml:"ttft_percentile,omitempty"`
	Description    string `yaml:"description,omitempty"`
}

// MLModelSelectionConfig configures the ML-based algorithm used from per-decision policies.
type MLModelSelectionConfig struct {
	Type             string             `yaml:"type"`
	ModelsPath       string             `yaml:"models_path,omitempty"`
	K                int                `yaml:"k,omitempty"`
	NumClusters      int                `yaml:"num_clusters,omitempty"`
	Kernel           string             `yaml:"kernel,omitempty"`
	Gamma            float64            `yaml:"gamma,omitempty"`
	EfficiencyWeight *float64           `yaml:"efficiency_weight,omitempty"`
	Device           string             `yaml:"device,omitempty"`
	FeatureWeights   map[string]float64 `yaml:"feature_weights,omitempty"`
}
