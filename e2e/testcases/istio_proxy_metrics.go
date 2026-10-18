package testcases

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	"github.com/vllm-project/semantic-router/e2e/pkg/modelruntime"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("istio-proxy-metrics", pkgtestcases.TestCase{
		Description: "Verify the injected Envoy proxy serves a live Prometheus gauge over HTTP",
		Tags:        []string{"istio", "metrics"},
		Fn:          testIstioProxyMetrics,
	})
}

// This is a metrics endpoint smoke test. TLS negotiation and exported tracing
// need independent traffic assertions and are not qualified by proxy readiness.
func testIstioProxyMetrics(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceEndpointSession(ctx, client, opts,
		"vllm-semantic-router-system", "semantic-router", "15090")
	if err != nil {
		return err
	}
	defer session.Close()
	if err := checkIstioProxyMetrics(ctx, &http.Client{Timeout: 30 * time.Second}, session.BaseURL()+"/stats/prometheus"); err != nil {
		return err
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"endpoint": "istio-proxy:15090/stats/prometheus", "envoy_server_live": 1})
	}
	return nil
}

func checkIstioProxyMetrics(ctx context.Context, client *http.Client, url string) error {
	response, err := fixtures.DoGETRequest(ctx, client, url)
	if err != nil {
		return err
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("istio proxy metrics returned HTTP %d", response.StatusCode)
	}
	// Envoy emits braces even for an unlabelled gauge: envoy_server_live{} 1.
	metrics, err := modelruntime.ParseMetrics(string(response.Body))
	if err != nil {
		return fmt.Errorf("parse istio proxy metrics: %w", err)
	}
	found := false
	for _, sample := range metrics.Samples {
		if sample.Name != "envoy_server_live" {
			continue
		}
		found = true
		if sample.Value != 1 {
			return fmt.Errorf("istio proxy live gauge is %g, want 1", sample.Value)
		}
	}
	if !found {
		return fmt.Errorf("istio proxy metrics omitted the envoy_server_live gauge; response prefix: %.512q", response.Body)
	}
	return nil
}
