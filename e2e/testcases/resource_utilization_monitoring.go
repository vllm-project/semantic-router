package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("resource-utilization-monitoring", pkgtestcases.TestCase{
		Description: "Verify Prometheus is scraping metrics for vLLM/router (non-empty series)",
		Tags:        []string{"observability"},
		Fn:          testResourceUtilizationMonitoring,
	})
}

func testResourceUtilizationMonitoring(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	// Port-forward to Prometheus service in default namespace
	stop, err := startPrometheusPF(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	// Query a basic metric that should exist for pods (container CPU usage)
	// Note: Using a short range vector to confirm series existence
	query := `sum(rate(container_cpu_usage_seconds_total{namespace=~"default|vllm-semantic-router-system"}[1m]))`
	ok, err := promHasNonEmptyResult(fmt.Sprintf("http://localhost:%d/api/v1/query?query=%s", 9090, url.QueryEscape(query)))
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("prometheus returned empty result for CPU usage query")
	}
	return nil
}

func startPrometheusPF(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) (func(), error) {
	stop, err := helpers.StartPortForward(ctx, client, opts.RestConfig, "default", "prometheus", "9090:9090", opts.Verbose)
	if err != nil {
		return nil, fmt.Errorf("failed to port-forward to prometheus: %w", err)
	}
	// settle
	time.Sleep(2 * time.Second)
	return stop, nil
}

func promHasNonEmptyResult(urlStr string) (bool, error) {
	httpClient := &http.Client{Timeout: 10 * time.Second}
	resp, err := httpClient.Get(urlStr)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		return false, fmt.Errorf("prometheus query status %d: %s", resp.StatusCode, string(b))
	}
	var out struct {
		Status string `json:"status"`
		Data   struct {
			Result []interface{} `json:"result"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return false, err
	}
	return len(out.Data.Result) > 0, nil
}
