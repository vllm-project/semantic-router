package evaluation

import "fmt"

// ParsePrometheusMetrics parses Prometheus query results into metrics.
func ParsePrometheusMetrics(queryResult map[string]interface{}, metricType string) (map[string]interface{}, error) {
	metrics := make(map[string]interface{})
	data, ok := queryResult["data"].(map[string]interface{})
	if !ok {
		return metrics, nil
	}
	mergePrometheusVectorResults(metrics, metricType, data)
	return metrics, nil
}

func mergePrometheusVectorResults(metrics map[string]interface{}, metricType string, data map[string]interface{}) {
	items, ok := data["result"].([]interface{})
	if !ok {
		return
	}
	for i, raw := range items {
		resultMap, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		mergeOnePrometheusResult(metrics, metricType, i, resultMap)
	}
}

func mergeOnePrometheusResult(metrics map[string]interface{}, metricType string, index int, resultMap map[string]interface{}) {
	if metric, ok := resultMap["metric"].(map[string]interface{}); ok {
		prefix := fmt.Sprintf("%s_%d", metricType, index)
		for k, v := range metric {
			metrics[fmt.Sprintf("%s_%s", prefix, k)] = v
		}
	}
	if value, ok := resultMap["value"].([]interface{}); ok && len(value) == 2 {
		metrics[fmt.Sprintf("%s_%d_value", metricType, index)] = value[1]
	}
}
