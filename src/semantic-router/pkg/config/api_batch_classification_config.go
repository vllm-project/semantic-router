package config

// BatchClassificationConfig controls router-side batch and concurrency behavior
// for classification-heavy request paths and APIs.
type BatchClassificationConfig struct {
	MaxBatchSize int `yaml:"max_batch_size,omitempty"`
	// Deprecated: superseded by global.model_catalog.admission. A non-zero
	// MaxConcurrency migrates into a wait-mode admission default for every
	// system deployment that has no explicit admission entry.
	ConcurrencyThreshold int                              `yaml:"concurrency_threshold,omitempty"`
	MaxConcurrency       int                              `yaml:"max_concurrency,omitempty"`
	Metrics              BatchClassificationMetricsConfig `yaml:"metrics,omitempty"`
}
