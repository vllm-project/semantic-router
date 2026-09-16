package cache

import "strings"

// normalizeMilvusMetricType canonicalizes a configured metric type:
// empty defaults to "IP", anything else is uppercased.
func normalizeMilvusMetricType(metricType string) string {
	if metricType == "" {
		return "IP"
	}
	return strings.ToUpper(metricType)
}

// milvusScoreToSimilarity maps a raw Milvus score to a similarity: L2 is the
// squared Euclidean distance (smaller is closer) and gets inverted; other
// metrics are similarities already (https://milvus.io/docs/metric.md).
func milvusScoreToSimilarity(metricType string, score float32) float32 {
	if strings.ToUpper(metricType) == "L2" {
		return 1.0 / (1.0 + score)
	}
	return score
}
