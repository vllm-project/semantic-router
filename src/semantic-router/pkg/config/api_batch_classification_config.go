package config

// BatchClassificationConfig controls router-side batch and concurrency behavior
// for classification-heavy request paths and APIs.
type BatchClassificationConfig struct {
	MaxBatchSize int                              `yaml:"max_batch_size,omitempty"`
	Metrics      BatchClassificationMetricsConfig `yaml:"metrics,omitempty"`
}
