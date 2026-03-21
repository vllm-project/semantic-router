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
	pkgtestcases.Register("dashboard-status", pkgtestcases.TestCase{
		Description: "Verify dashboard /api/status reports overall system status with services listed",
		Tags:        []string{"dashboard", "status"},
		Fn:          testDashboardStatus,
	})
}

func testDashboardStatus(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	url := fmt.Sprintf("http://localhost:%s/api/status", localPort)
	if opts.Verbose {
		fmt.Printf("[Dashboard] GET %s\n", url)
	}

	httpClient := &http.Client{Timeout: 15 * time.Second}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("status request failed: %w", err)
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

	overall, ok := result["overall"].(string)
	if !ok || overall == "" {
		return fmt.Errorf("expected non-empty 'overall' field in status response, got: %v", result)
	}

	services, ok := result["services"].([]interface{})
	if !ok {
		return fmt.Errorf("expected 'services' array in status response, got: %v", result["services"])
	}

	if opts.Verbose {
		fmt.Printf("[Dashboard] status OK: overall=%s, services=%d\n", overall, len(services))
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"overall":       overall,
			"service_count": len(services),
		})
	}

	return nil
}
