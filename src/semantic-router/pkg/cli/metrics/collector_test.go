package metrics

import (
	"strings"
	"testing"
)

func TestParsePrometheusMetrics_ErrorRates(t *testing.T) {
	mockData := `
http_requests_total{method="GET"} 100
http_requests_total{method="POST"} 200
llm_request_errors_total{reason="timeout"} 5
llm_request_errors_total{reason="upstream_5xx"} 3
llm_request_errors_total{reason="parse_error"} 12
`

	collector := NewCollector("http://test")
	data := &MetricsData{Available: true}

	err := collector.parsePrometheusMetrics(strings.NewReader(mockData), data)
	if err != nil {
		t.Fatalf("parsePrometheusMetrics failed: %v", err)
	}

	// Total: 300, Errors: 20, Error rate: 6.67%, Success: 93.33%
	if data.TotalRequests != 300 {
		t.Errorf("TotalRequests = %d, want 300", data.TotalRequests)
	}

	expectedErrorRate := 6.67
	if data.ErrorRate < 6.6 || data.ErrorRate > 6.7 {
		t.Errorf("ErrorRate = %.2f, want ~%.2f", data.ErrorRate, expectedErrorRate)
	}

	expectedSuccessRate := 93.33
	if data.SuccessRate < 93.3 || data.SuccessRate > 93.4 {
		t.Errorf("SuccessRate = %.2f, want ~%.2f", data.SuccessRate, expectedSuccessRate)
	}
}

func TestParsePrometheusMetrics_NoData(t *testing.T) {
	mockData := `# Just comments`

	collector := NewCollector("http://test")
	data := &MetricsData{Available: true}

	err := collector.parsePrometheusMetrics(strings.NewReader(mockData), data)
	if err != nil {
		t.Fatalf("parsePrometheusMetrics failed: %v", err)
	}

	// No data = 0% rates, not fake 95%
	if data.SuccessRate != 0.0 || data.ErrorRate != 0.0 {
		t.Errorf("Rates should be 0.0 with no data, got success=%.1f error=%.1f",
			data.SuccessRate, data.ErrorRate)
	}
}

func TestParsePrometheusMetrics_ZeroErrors(t *testing.T) {
	mockData := `
http_requests_total{method="GET"} 100
`

	collector := NewCollector("http://test")
	data := &MetricsData{Available: true}

	err := collector.parsePrometheusMetrics(strings.NewReader(mockData), data)
	if err != nil {
		t.Fatalf("parsePrometheusMetrics failed: %v", err)
	}

	if data.TotalRequests != 100 {
		t.Errorf("TotalRequests = %d, want 100", data.TotalRequests)
	}

	if data.ErrorRate != 0.0 {
		t.Errorf("ErrorRate = %.2f, want 0.0", data.ErrorRate)
	}

	if data.SuccessRate != 100.0 {
		t.Errorf("SuccessRate = %.2f, want 100.0", data.SuccessRate)
	}
}

func TestParsePrometheusMetrics_AllErrors(t *testing.T) {
	mockData := `
http_requests_total{method="POST"} 50
llm_request_errors_total{reason="timeout"} 50
`

	collector := NewCollector("http://test")
	data := &MetricsData{Available: true}

	err := collector.parsePrometheusMetrics(strings.NewReader(mockData), data)
	if err != nil {
		t.Fatalf("parsePrometheusMetrics failed: %v", err)
	}

	if data.TotalRequests != 50 {
		t.Errorf("TotalRequests = %d, want 50", data.TotalRequests)
	}

	if data.ErrorRate != 100.0 {
		t.Errorf("ErrorRate = %.2f, want 100.0", data.ErrorRate)
	}

	if data.SuccessRate != 0.0 {
		t.Errorf("SuccessRate = %.2f, want 0.0", data.SuccessRate)
	}
}
