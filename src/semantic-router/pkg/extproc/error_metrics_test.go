package extproc

import (
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

// getCounterValue returns the sum of a counter across metrics matching the given labels
func getCounterValue(metricName string, want map[string]string) float64 {
	var sum float64
	mfs, _ := prometheus.DefaultGatherer.Gather()
	for _, fam := range mfs {
		if fam.GetName() != metricName || fam.GetType() != dto.MetricType_COUNTER {
			continue
		}
		for _, m := range fam.GetMetric() {
			labels := m.GetLabel()
			match := true
			for k, v := range want {
				found := false
				for _, l := range labels {
					if l.GetName() == k && l.GetValue() == v {
						found = true
						break
					}
				}
				if !found {
					match = false
					break
				}
			}
			if match && m.GetCounter() != nil {
				sum += m.GetCounter().GetValue()
			}
		}
	}
	return sum
}

func TestRequestParseErrorIncrementsErrorCounter(t *testing.T) {
	r := &OpenAIRouter{}
	r.InitializeForTesting()

	ctx := &RequestContext{}
	// Invalid JSON triggers parse error
	badBody := []byte("not-json")
	v := &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: badBody},
	}

	before := getCounterValue("llm_request_errors_total", map[string]string{"reason": "parse_error", "model": "unknown"})

	// Use test helper wrapper to access unexported method
	_, _ = r.HandleRequestBody(v, ctx)

	after := getCounterValue("llm_request_errors_total", map[string]string{"reason": "parse_error", "model": "unknown"})
	if !(after > before) {
		t.Fatalf("expected llm_request_errors_total(parse_error,unknown) to increase: before=%v after=%v", before, after)
	}
}

func TestResponseParseErrorIncrementsErrorCounter(t *testing.T) {
	r := &OpenAIRouter{}
	r.InitializeForTesting()

	ctx := &RequestContext{RequestModel: "model-a"}
	// Invalid JSON triggers parse error in response body handler
	badJSON := []byte("{invalid}")
	v := &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: badJSON},
	}

	before := getCounterValue("llm_request_errors_total", map[string]string{"reason": "parse_error", "model": "model-a"})

	_, _ = r.HandleResponseBody(v, ctx)

	after := getCounterValue("llm_request_errors_total", map[string]string{"reason": "parse_error", "model": "model-a"})
	if !(after > before) {
		t.Fatalf("expected llm_request_errors_total(parse_error,model-a) to increase: before=%v after=%v", before, after)
	}
}

func TestUpstreamStatusIncrements4xx5xxCounters(t *testing.T) {
	r := &OpenAIRouter{}
	r.InitializeForTesting()

	ctx := &RequestContext{RequestModel: "m"}

	// 503 -> upstream_5xx
	hdrs5xx := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{Headers: []*core.HeaderValue{{Key: ":status", Value: "503"}}},
		},
	}

	before5xx := getCounterValue("llm_request_errors_total", map[string]string{"reason": "upstream_5xx", "model": "m"})
	_, _ = r.HandleResponseHeaders(hdrs5xx, ctx)
	after5xx := getCounterValue("llm_request_errors_total", map[string]string{"reason": "upstream_5xx", "model": "m"})
	if !(after5xx > before5xx) {
		t.Fatalf("expected upstream_5xx to increase for model m: before=%v after=%v", before5xx, after5xx)
	}

	// 404 -> upstream_4xx
	hdrs4xx := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{Headers: []*core.HeaderValue{{Key: ":status", Value: "404"}}},
		},
	}

	before4xx := getCounterValue("llm_request_errors_total", map[string]string{"reason": "upstream_4xx", "model": "m"})
	_, _ = r.HandleResponseHeaders(hdrs4xx, ctx)
	after4xx := getCounterValue("llm_request_errors_total", map[string]string{"reason": "upstream_4xx", "model": "m"})
	if !(after4xx > before4xx) {
		t.Fatalf("expected upstream_4xx to increase for model m: before=%v after=%v", before4xx, after4xx)
	}
}
