package config

// PrototypeScoringConfig controls prototype-bank construction and bank scoring
// for embedding-backed signal families.
type PrototypeScoringConfig struct {
	Enabled                    *bool   `json:"enabled,omitempty" yaml:"enabled,omitempty"`
	ClusterSimilarityThreshold float32 `json:"cluster_similarity_threshold,omitempty" yaml:"cluster_similarity_threshold,omitempty"`
	MaxPrototypes              int     `json:"max_prototypes,omitempty" yaml:"max_prototypes,omitempty"`
	BestWeight                 float32 `json:"best_weight,omitempty" yaml:"best_weight,omitempty"`
	TopM                       int     `json:"top_m,omitempty" yaml:"top_m,omitempty"`
	MarginThreshold            float32 `json:"margin_threshold,omitempty" yaml:"margin_threshold,omitempty"`
}

func (c PrototypeScoringConfig) WithDefaults() PrototypeScoringConfig {
	result := c
	if result.Enabled == nil {
		enabled := true
		result.Enabled = &enabled
	}
	if result.ClusterSimilarityThreshold <= 0 {
		result.ClusterSimilarityThreshold = 0.9
	}
	if result.MaxPrototypes <= 0 {
		result.MaxPrototypes = 8
	}
	if result.BestWeight <= 0 {
		result.BestWeight = 0.75
	}
	if result.BestWeight > 1 {
		result.BestWeight = 1
	}
	if result.TopM <= 0 {
		result.TopM = 2
	}
	if result.MarginThreshold < 0 {
		result.MarginThreshold = 0
	}
	return result
}

func (c PrototypeScoringConfig) IsEnabled() bool {
	result := c.WithDefaults()
	return result.Enabled != nil && *result.Enabled
}
