package v1alpha1

// PrototypeScoringConfig controls one rule's prototype banks. These fields have
// no admission defaults, so omitted and empty overrides remain distinct.
type PrototypeScoringConfig struct {
	// Enabled controls clustering and the representative cap, not aggregation.
	// +optional
	Enabled *bool `json:"enabled,omitempty" yaml:"enabled,omitempty"`
	// ClusterSimilarityThreshold controls membership in prototype clusters.
	// +optional
	ClusterSimilarityThreshold float32 `json:"clusterSimilarityThreshold,omitempty" yaml:"clusterSimilarityThreshold,omitempty"`
	// MaxPrototypes caps representatives when clustering is enabled.
	// +optional
	MaxPrototypes int `json:"maxPrototypes,omitempty" yaml:"maxPrototypes,omitempty"`
	// BestWeight weights the best similarity against top-M support.
	// +optional
	BestWeight float32 `json:"bestWeight,omitempty" yaml:"bestWeight,omitempty"`
	// TopM is the number of top similarities used for support.
	// +optional
	TopM int `json:"topM,omitempty" yaml:"topM,omitempty"`
	// MarginThreshold controls the winner-versus-runner-up score margin.
	// +optional
	MarginThreshold float32 `json:"marginThreshold,omitempty" yaml:"marginThreshold,omitempty"`
}
