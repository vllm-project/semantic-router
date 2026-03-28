package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("dashboard-health", pkgtestcases.TestCase{
		Description: "Verify dashboard /healthz returns healthy status",
		Tags:        []string{"dashboard", "health"},
		Fn:          testDashboardHealth,
	})
}

func testDashboardHealth(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	url := fmt.Sprintf("http://localhost:%s/healthz", localPort)
	if opts.Verbose {
		fmt.Printf("[Dashboard] GET %s\n", url)
	}

	httpClient := &http.Client{Timeout: 10 * time.Second}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("health check request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected 200, got %d: %s", resp.StatusCode, truncateString(string(body), 200))
	}

	var result map[string]interface{}
	if err := json.Unmarshal(body, &result); err != nil {
		return fmt.Errorf("response is not valid JSON: %w (body: %s)", err, truncateString(string(body), 200))
	}

	status, ok := result["status"].(string)
	if !ok || status != "healthy" {
		return fmt.Errorf("expected status=healthy, got: %v", result["status"])
	}

	if opts.Verbose {
		fmt.Printf("[Dashboard] health OK: status=%s\n", status)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"status": status})
	}

	return nil
}
