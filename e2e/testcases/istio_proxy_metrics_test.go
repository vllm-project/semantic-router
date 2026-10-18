package testcases

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestIstioProxyMetricsRequiresSuccessfulLiveSample(t *testing.T) {
	for _, test := range []struct {
		name      string
		status    int
		body      string
		wantError bool
	}{
		{"live", 200, "# TYPE envoy_server_live gauge\nenvoy_server_live 1\n", false},
		{"not ready", 200, "envoy_server_live 0\n", true},
		{"HTTP failure", 503, "envoy_server_live 1\n", true},
		{"HTML", 200, "<html>envoy_server_live</html>", true},
		{"comment only", 200, "# envoy_server_live 1\n", true},
		{"unrelated metric", 200, "envoy_server_uptime 42\n", true},
		{"invalid value", 200, "envoy_server_live NaN\n", true},
		{"missing", 200, "", true},
	} {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/stats/prometheus" {
					t.Errorf("unexpected request path %s", r.URL.Path)
				}
				w.WriteHeader(test.status)
				_, _ = w.Write([]byte(test.body))
			}))
			defer server.Close()
			err := checkIstioProxyMetrics(context.Background(), server.Client(), server.URL+"/stats/prometheus")
			if (err != nil) != test.wantError {
				t.Fatalf("error = %v, wantError = %v", err, test.wantError)
			}
		})
	}
}
