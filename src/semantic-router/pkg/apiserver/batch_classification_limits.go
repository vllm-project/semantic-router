//go:build !windows && cgo

package apiserver

const defaultBatchClassificationMaxBatchSize = 100

func (s *ClassificationAPIServer) effectiveBatchClassificationMaxBatchSize() int {
	cfg := s.currentConfig()
	if cfg == nil || cfg.API.BatchClassification.MaxBatchSize <= 0 {
		return defaultBatchClassificationMaxBatchSize
	}
	return cfg.API.BatchClassification.MaxBatchSize
}
