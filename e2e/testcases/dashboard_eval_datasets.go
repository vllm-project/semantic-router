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
	pkgtestcases.Register("dashboard-eval-datasets", pkgtestcases.TestCase{
		Description: "Verify dashboard /api/evaluation/datasets returns a non-empty dataset list",
		Tags:        []string{"dashboard", "evaluation"},
		Fn:          testDashboardEvalDatasets,
	})
}

func testDashboardEvalDatasets(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	url := fmt.Sprintf("http://localhost:%s/api/evaluation/datasets", localPort)
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
		return fmt.Errorf("evaluation/datasets request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected 200, got %d: %s", resp.StatusCode, truncateString(string(body), 200))
	}

	// Response is a map of dimension -> []DatasetInfo
	var datasets map[string]interface{}
	if err := json.Unmarshal(body, &datasets); err != nil {
		return fmt.Errorf("evaluation/datasets response is not valid JSON: %w", err)
	}

	if len(datasets) == 0 {
		return fmt.Errorf("evaluation/datasets returned empty map — expected at least one dimension")
	}

	if opts.Verbose {
		fmt.Printf("[Dashboard] eval/datasets OK: dimensions=%d\n", len(datasets))
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"dimension_count": len(datasets)})
	}

	return nil
}
