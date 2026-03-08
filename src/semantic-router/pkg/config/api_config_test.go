package config

import (
	"testing"

	"gopkg.in/yaml.v3"
)

func TestBatchClassificationAPILimitsParse(t *testing.T) {
	yamlContent := `
api:
  batch_classification:
    max_batch_size: 64
    concurrency_threshold: 3
    max_concurrency: 6
    metrics:
      enabled: true
`

	var cfg RouterConfig
	if err := yaml.Unmarshal([]byte(yamlContent), &cfg); err != nil {
		t.Fatalf("unmarshal api config: %v", err)
	}

	batchConfig := cfg.API.BatchClassification
	if batchConfig.MaxBatchSize != 64 {
		t.Fatalf("max_batch_size = %d, want 64", batchConfig.MaxBatchSize)
	}
	if batchConfig.ConcurrencyThreshold != 3 {
		t.Fatalf(
			"concurrency_threshold = %d, want 3",
			batchConfig.ConcurrencyThreshold,
		)
	}
	if batchConfig.MaxConcurrency != 6 {
		t.Fatalf("max_concurrency = %d, want 6", batchConfig.MaxConcurrency)
	}
	if !batchConfig.Metrics.Enabled {
		t.Fatalf("metrics.enabled = false, want true")
	}
}
