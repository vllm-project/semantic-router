package metrics

import (
	"bufio"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli"
)

// MetricsData holds router metrics
type MetricsData struct {
	TotalRequests    int64
	SuccessRate      float64
	ErrorRate        float64
	AvgLatency       float64
	P95Latency       float64
	P99Latency       float64
	IntentClassCount int64
	PIIDetectCount   int64
	SecurityCount    int64
	Available        bool
	Error            string
}

// Collector handles metrics collection from Prometheus
type Collector struct {
	metricsURL string
	client     *http.Client
}

// NewCollector creates a new metrics collector
func NewCollector(metricsURL string) *Collector {
	return &Collector{
		metricsURL: metricsURL,
		client: &http.Client{
			Timeout: 5 * time.Second,
		},
	}
}

// Collect fetches metrics from Prometheus endpoint
func (c *Collector) Collect() (*MetricsData, error) {
	resp, err := c.client.Get(c.metricsURL)
	if err != nil {
		return &MetricsData{
			Available: false,
			Error:     fmt.Sprintf("Failed to connect to metrics endpoint: %v", err),
		}, nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return &MetricsData{
			Available: false,
			Error:     fmt.Sprintf("Metrics endpoint returned status: %d", resp.StatusCode),
		}, nil
	}

	// Parse Prometheus text format
	metrics := &MetricsData{Available: true}
	if err := c.parsePrometheusMetrics(resp.Body, metrics); err != nil {
		return &MetricsData{
			Available: false,
			Error:     fmt.Sprintf("Failed to parse metrics: %v", err),
		}, nil
	}

	return metrics, nil
}

// parsePrometheusMetrics parses Prometheus text format
func (c *Collector) parsePrometheusMetrics(r io.Reader, data *MetricsData) error {
	scanner := bufio.NewScanner(r)

	var totalErrors int64

	for scanner.Scan() {
		line := scanner.Text()

		// Skip comments and empty lines
		if strings.HasPrefix(line, "#") || line == "" {
			continue
		}

		// Parse metric line: metric_name{labels} value
		parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}

		metricLine := parts[0]
		valueStr := parts[1]

		value, err := strconv.ParseFloat(valueStr, 64)
		if err != nil {
			continue
		}

		// Extract metric name (before { or space)
		metricName := metricLine
		if idx := strings.Index(metricLine, "{"); idx > 0 {
			metricName = metricLine[:idx]
		}

		// Map metrics to data structure
		switch {
		case strings.Contains(metricName, "llm_category_classifications_total"):
			// Category classification count
			if strings.Contains(metricLine, "category=\"intent\"") {
				data.IntentClassCount = int64(value)
			} else if strings.Contains(metricLine, "category=\"pii\"") {
				data.PIIDetectCount = int64(value)
			} else if strings.Contains(metricLine, "category=\"security\"") {
				data.SecurityCount = int64(value)
			}

		case strings.Contains(metricName, "http_requests_total"):
			data.TotalRequests += int64(value)

		case strings.Contains(metricName, "llm_request_errors_total"):
			totalErrors += int64(value)

		case strings.Contains(metricName, "http_request_duration_seconds_sum"):
			// This will need the count to calculate average
			// For now, store temporarily

		case strings.Contains(metricName, "http_request_duration_seconds"):
			// Latency metrics
			if strings.Contains(metricLine, "quantile=\"0.95\"") {
				data.P95Latency = value * 1000 // Convert to ms
			} else if strings.Contains(metricLine, "quantile=\"0.99\"") {
				data.P99Latency = value * 1000 // Convert to ms
			} else if strings.Contains(metricLine, "quantile=\"0.5\"") {
				data.AvgLatency = value * 1000 // Use median as avg
			}
		}
	}

	// Calculate rates if we have total requests
	if data.TotalRequests > 0 {
		data.ErrorRate = (float64(totalErrors) / float64(data.TotalRequests)) * 100.0
		data.SuccessRate = 100.0 - data.ErrorRate
	} else {
		data.SuccessRate = 0.0
		data.ErrorRate = 0.0
	}

	return scanner.Err()
}

// DetectMetricsEndpoint tries to find an available metrics endpoint
func DetectMetricsEndpoint() string {
	client := &http.Client{Timeout: 2 * time.Second}

	// Try endpoints in order of preference
	endpoints := []string{
		"http://localhost:9190/metrics",      // Router direct metrics (local/docker)
		"http://localhost:9090/api/v1/query", // Prometheus API (docker with observability)
		"http://localhost:9091/metrics",      // Local observability Prometheus
	}

	for _, endpoint := range endpoints {
		resp, err := client.Get(endpoint)
		if err == nil && resp.StatusCode == http.StatusOK {
			resp.Body.Close()

			// For Prometheus API endpoints, we'd need different handling
			// For now, only use direct metrics endpoints
			if strings.Contains(endpoint, "/metrics") {
				return endpoint
			}
		}
		if resp != nil {
			resp.Body.Close()
		}
	}

	return ""
}

// FormatMetricsTable formats metrics as a table display
func FormatMetricsTable(data *MetricsData, since string) {
	cli.Info("╔═══════════════════════════════════════════════════════════════╗")
	cli.Info("║                     Router Metrics                            ║")
	cli.Info("╠═══════════════════════════════════════════════════════════════╣")
	cli.Info(fmt.Sprintf("║ Time Range: %-48s║", since))
	cli.Info("╠═══════════════════════════════════════════════════════════════╣")

	if !data.Available {
		cli.Info("║                                                               ║")
		cli.Info("║ ❌ Metrics not available                                      ║")
		cli.Info(fmt.Sprintf("║ Error: %-56s║", truncate(data.Error, 56)))
		cli.Info("║                                                               ║")
		cli.Info("╚═══════════════════════════════════════════════════════════════╝")
		fmt.Println()
		cli.Warning("Metrics endpoint not accessible")
		cli.Info("Ensure router is running with metrics enabled")
		cli.Info("Try: vsr status")
		return
	}

	cli.Info("║                                                               ║")
	cli.Info("║ 📊 Request Statistics                                         ║")
	cli.Info(fmt.Sprintf("║   Total Requests:        %-36d║", data.TotalRequests))
	cli.Info(fmt.Sprintf("║   Success Rate:          %-32.1f%%    ║", data.SuccessRate))
	cli.Info(fmt.Sprintf("║   Error Rate:            %-32.1f%%    ║", data.ErrorRate))
	cli.Info("║                                                               ║")
	cli.Info("║ ⏱️  Latency                                                    ║")
	cli.Info(fmt.Sprintf("║   Avg Response Time:     %-28.2f ms    ║", data.AvgLatency))
	cli.Info(fmt.Sprintf("║   P95 Response Time:     %-28.2f ms    ║", data.P95Latency))
	cli.Info(fmt.Sprintf("║   P99 Response Time:     %-28.2f ms    ║", data.P99Latency))
	cli.Info("║                                                               ║")
	cli.Info("║ 🤖 Model Usage                                                ║")
	cli.Info(fmt.Sprintf("║   Intent Classifier:     %-36d║", data.IntentClassCount))
	cli.Info(fmt.Sprintf("║   PII Detector:          %-36d║", data.PIIDetectCount))
	cli.Info(fmt.Sprintf("║   Security Classifier:   %-36d║", data.SecurityCount))
	cli.Info("║                                                               ║")
	cli.Info("╚═══════════════════════════════════════════════════════════════╝")

	if data.TotalRequests == 0 {
		fmt.Println()
		cli.Info("No requests processed yet. Send some traffic to see metrics.")
	}
}

// truncate truncates a string to maxLen
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
