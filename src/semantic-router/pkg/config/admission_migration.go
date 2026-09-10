package config

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func applyBatchConcurrencyMigration(cfg *RouterConfig) {
	maxConcurrency := cfg.API.BatchClassification.MaxConcurrency
	if maxConcurrency <= 0 {
		return
	}
	if cfg.ModelAdmission == nil {
		cfg.ModelAdmission = make(map[string]AdmissionConfig, len(admissionDeploymentKeys))
	}
	migrated := 0
	for deployment := range admissionDeploymentKeys {
		if _, ok := cfg.ModelAdmission[deployment]; ok {
			continue
		}
		cfg.ModelAdmission[deployment] = AdmissionConfig{
			MaxConcurrency: maxConcurrency,
			MaxQueue:       maxConcurrency,
			OnOverflow:     "wait",
		}
		migrated++
	}
	logging.Warnf(
		"api.batch_classification.max_concurrency/concurrency_threshold are deprecated; migrated max_concurrency=%d into wait-mode global.model_catalog.admission defaults for %d deployment(s). Configure admission directly instead.",
		maxConcurrency,
		migrated,
	)
}
