package testcases

import (
	"bufio"
	"context"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
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
		return fmt.Errorf("Istio proxy metrics returned HTTP %d", response.StatusCode)
	}
	scanner := bufio.NewScanner(strings.NewReader(string(response.Body)))
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 2 || fields[0] != "envoy_server_live" {
			continue
		}
		value, err := strconv.ParseFloat(fields[1], 64)
		if err != nil || value != 1 {
			return fmt.Errorf("Istio proxy live gauge is %q, want 1", fields[1])
		}
		return nil
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read Istio proxy metrics: %w", err)
	}
	return fmt.Errorf("Istio proxy metrics omitted the envoy_server_live gauge")
}
