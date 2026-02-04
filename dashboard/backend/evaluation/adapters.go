package evaluation

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ParseHallucinationOutput parses the output from the hallucination benchmark.
// It first tries to read from the output file, then falls back to parsing stdout.
func ParseHallucinationOutput(stdout, outputPath string) (map[string]interface{}, error) {
	var result map[string]interface{}

	// First, try to read from the output file
	if outputPath != "" {
		// The script generates files with timestamp, so find the latest one
		dir := filepath.Dir(outputPath)
		files, err := os.ReadDir(dir)
		if err == nil {
			var latestFile string
			var latestTime int64
			for _, f := range files {
				if strings.HasPrefix(f.Name(), "results_") && strings.HasSuffix(f.Name(), ".json") {
					filePath := filepath.Join(dir, f.Name())
					info, err := f.Info()
					if err == nil {
						modTime := info.ModTime().UnixNano()
						if modTime > latestTime {
							latestTime = modTime
							latestFile = filePath
						}
					}
				}
			}
			if latestFile != "" {
				data, err := os.ReadFile(latestFile)
				if err == nil {
					if err := json.Unmarshal(data, &result); err == nil {
						return extractHallucinationMetrics(result), nil
					}
				}
			}
		}
	}

	// Fall back to parsing stdout for JSON
	lines := strings.Split(stdout, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "{") && strings.HasSuffix(line, "}") {
			if err := json.Unmarshal([]byte(line), &result); err == nil {
				return extractHallucinationMetrics(result), nil
			}
		}
	}

	// Return empty metrics if parsing failed
	return map[string]interface{}{
		"error":  "Failed to parse benchmark output",
		"stdout": stdout,
		"status": "parse_error",
	}, nil
}

// extractHallucinationMetrics extracts the relevant metrics from hallucination results.
func extractHallucinationMetrics(raw map[string]interface{}) map[string]interface{} {
	metrics := make(map[string]interface{})

	// Copy over all numeric and string metrics
	keys := []string{
		"dataset_name", "endpoint", "model",
		"total_samples", "successful_requests", "failed_requests",
		"total_hallucinations_detected",
		"true_positives", "false_positives", "true_negatives", "false_negatives",
		"precision", "recall", "f1_score", "accuracy",
		"avg_latency_ms", "p50_latency_ms", "p99_latency_ms",
		"fact_check_needed_count", "detection_skipped_count",
		"avg_context_length", "estimated_detection_time_ms",
		"actual_detection_time_ms", "time_saved_ms", "efficiency_gain_percent",
	}

	for _, key := range keys {
		if val, ok := raw[key]; ok {
			metrics[key] = val
		}
	}

	metrics["status"] = "success"
	return metrics
}

// ParseReasoningOutput parses the output from the reasoning mode benchmark.
func ParseReasoningOutput(stdout, summaryPath string) (map[string]interface{}, error) {
	var result map[string]interface{}

	// Try to read from the summary file
	if summaryPath != "" {
		if data, err := os.ReadFile(summaryPath); err == nil {
			if err := json.Unmarshal(data, &result); err == nil {
				return extractReasoningMetrics(result), nil
			}
		}
	}

	// Fall back to parsing stdout
	lines := strings.Split(stdout, "\n")
	var jsonContent strings.Builder
	inJSON := false

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "{") {
			inJSON = true
			jsonContent.Reset()
		}
		if inJSON {
			jsonContent.WriteString(line)
		}
		if inJSON && strings.HasSuffix(line, "}") {
			if err := json.Unmarshal([]byte(jsonContent.String()), &result); err == nil {
				return extractReasoningMetrics(result), nil
			}
			inJSON = false
		}
	}

	return map[string]interface{}{
		"error":  "Failed to parse reasoning benchmark output",
		"stdout": stdout,
		"status": "parse_error",
	}, nil
}

// extractReasoningMetrics extracts the relevant metrics from reasoning results.
func extractReasoningMetrics(raw map[string]interface{}) map[string]interface{} {
	metrics := make(map[string]interface{})

	// Copy basic metadata
	if val, ok := raw["generated_at"]; ok {
		metrics["generated_at"] = val
	}
	if val, ok := raw["title"]; ok {
		metrics["title"] = val
	}

	// Extract comparisons
	if comparisons, ok := raw["comparisons"].([]interface{}); ok && len(comparisons) > 0 {
		// Get the first comparison as the primary result
		if comp, ok := comparisons[0].(map[string]interface{}); ok {
			metrics["dataset"] = comp["dataset"]
			metrics["model"] = comp["model"]

			// Extract standard mode metrics
			if standardMode, ok := comp["standard_mode"].(map[string]interface{}); ok {
				metrics["standard_accuracy"] = standardMode["accuracy"]
				metrics["standard_avg_response_time_sec"] = standardMode["avg_response_time_sec"]
				metrics["standard_token_usage_ratio"] = standardMode["token_usage_ratio"]
				metrics["standard_time_per_output_token_ms"] = standardMode["time_per_output_token_ms"]
			}

			// Extract reasoning mode metrics
			if reasoningMode, ok := comp["reasoning_mode"].(map[string]interface{}); ok {
				metrics["reasoning_accuracy"] = reasoningMode["accuracy"]
				metrics["reasoning_avg_response_time_sec"] = reasoningMode["avg_response_time_sec"]
				metrics["reasoning_token_usage_ratio"] = reasoningMode["token_usage_ratio"]
				metrics["reasoning_time_per_output_token_ms"] = reasoningMode["time_per_output_token_ms"]
			}

			// Extract improvement summary
			if improvement, ok := comp["improvement_summary"].(map[string]interface{}); ok {
				metrics["accuracy_delta"] = improvement["accuracy_delta"]
				metrics["accuracy_improvement_pct"] = improvement["accuracy_improvement_pct"]
				metrics["token_usage_ratio_delta"] = improvement["token_usage_ratio_delta"]
				metrics["response_time_delta_sec"] = improvement["response_time_delta_sec"]
			}

			// Include category breakdown if present
			if categories, ok := comp["category_breakdown"].(map[string]interface{}); ok {
				metrics["category_breakdown"] = categories
			}
		}

		// Store all comparisons for multi-dataset evaluations
		metrics["comparisons"] = comparisons
	}

	// Include VSR config recommendation if present
	if vsrConfig, ok := raw["vsr_config_recommendation"]; ok {
		metrics["vsr_config_recommendation"] = vsrConfig
	}

	metrics["status"] = "success"
	return metrics
}

// ParseAccuracyOutput parses the output from the accuracy benchmark.
func ParseAccuracyOutput(stdout, outputPath string) (map[string]interface{}, error) {
	var result map[string]interface{}

	// Try to read from the output file
	if outputPath != "" {
		if data, err := os.ReadFile(outputPath); err == nil {
			if err := json.Unmarshal(data, &result); err == nil {
				return extractAccuracyMetrics(result), nil
			}
		}
	}

	// Parse stdout for metrics
	metrics := make(map[string]interface{})
	lines := strings.Split(stdout, "\n")

	for _, line := range lines {
		line = strings.TrimSpace(line)

		// Look for accuracy report lines
		if strings.Contains(line, "Accuracy:") {
			var accuracy float64
			_, _ = fmt.Sscanf(line, "Accuracy: %f", &accuracy)
			metrics["accuracy"] = accuracy
		}

		// Look for confusion matrix values
		if strings.Contains(line, "TP:") {
			var tp int
			_, _ = fmt.Sscanf(line, "TP: %d", &tp)
			metrics["true_positives"] = tp
		}
		if strings.Contains(line, "FP:") {
			var fp int
			_, _ = fmt.Sscanf(line, "FP: %d", &fp)
			metrics["false_positives"] = fp
		}
		if strings.Contains(line, "TN:") {
			var tn int
			_, _ = fmt.Sscanf(line, "TN: %d", &tn)
			metrics["true_negatives"] = tn
		}
		if strings.Contains(line, "FN:") {
			var fn int
			_, _ = fmt.Sscanf(line, "FN: %d", &fn)
			metrics["false_negatives"] = fn
		}
	}

	if len(metrics) == 0 {
		return map[string]interface{}{
			"error":  "Failed to parse accuracy benchmark output",
			"stdout": stdout,
			"status": "parse_error",
		}, nil
	}

	metrics["status"] = "success"
	return metrics, nil
}

// extractAccuracyMetrics extracts the relevant metrics from accuracy results.
func extractAccuracyMetrics(raw map[string]interface{}) map[string]interface{} {
	metrics := make(map[string]interface{})

	// Copy over standard accuracy metrics
	keys := []string{
		"overall_accuracy", "accuracy", "true_positives", "false_positives",
		"true_negatives", "false_negatives",
		"precision", "recall", "f1_score",
		"total_samples", "total_questions", "correct_predictions",
		"successful_queries", "failed_queries",
		"category_metrics", "category_accuracy",
		"avg_response_time", "avg_prompt_tokens", "avg_completion_tokens", "avg_total_tokens",
		"by_mode",
	}

	for _, key := range keys {
		if val, ok := raw[key]; ok {
			metrics[key] = val
		}
	}

	// Map overall_accuracy to accuracy for consistency
	if overallAcc, ok := raw["overall_accuracy"]; ok {
		metrics["accuracy"] = overallAcc
	}

	metrics["status"] = "success"
	return metrics
}

// ParsePrometheusMetrics parses Prometheus query results into metrics.
func ParsePrometheusMetrics(queryResult map[string]interface{}, metricType string) (map[string]interface{}, error) {
	metrics := make(map[string]interface{})

	// Prometheus API response format:
	// { "status": "success", "data": { "resultType": "vector/matrix", "result": [...] } }
	if data, ok := queryResult["data"].(map[string]interface{}); ok {
		if result, ok := data["result"].([]interface{}); ok {
			for i, r := range result {
				if resultMap, ok := r.(map[string]interface{}); ok {
					// Get metric labels
					if metric, ok := resultMap["metric"].(map[string]interface{}); ok {
						prefix := fmt.Sprintf("%s_%d", metricType, i)
						for k, v := range metric {
							metrics[fmt.Sprintf("%s_%s", prefix, k)] = v
						}
					}

					// Get value(s)
					if value, ok := resultMap["value"].([]interface{}); ok && len(value) == 2 {
						metrics[fmt.Sprintf("%s_%d_value", metricType, i)] = value[1]
					}
				}
			}
		}
	}

	return metrics, nil
}

// CalculateDerivedMetrics calculates additional metrics from raw results.
func CalculateDerivedMetrics(metrics map[string]interface{}) map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range metrics {
		result[k] = v
	}

	// Calculate precision if TP and FP are available
	tp, tpOk := getFloat(metrics, "true_positives")
	fp, fpOk := getFloat(metrics, "false_positives")
	fn, fnOk := getFloat(metrics, "false_negatives")

	if tpOk && fpOk && (tp+fp) > 0 {
		result["precision"] = tp / (tp + fp)
	}

	if tpOk && fnOk && (tp+fn) > 0 {
		result["recall"] = tp / (tp + fn)
	}

	// Calculate F1 score
	precision, precOk := getFloat(result, "precision")
	recall, recallOk := getFloat(result, "recall")
	if precOk && recallOk && (precision+recall) > 0 {
		result["f1_score"] = 2 * precision * recall / (precision + recall)
	}

	return result
}

// getFloat attempts to extract a float64 value from a map.
func getFloat(m map[string]interface{}, key string) (float64, bool) {
	if v, ok := m[key]; ok {
		switch val := v.(type) {
		case float64:
			return val, true
		case int:
			return float64(val), true
		case int64:
			return float64(val), true
		}
	}
	return 0, false
}
