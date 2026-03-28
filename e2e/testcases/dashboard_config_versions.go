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
	pkgtestcases.Register("dashboard-config-versions", pkgtestcases.TestCase{
		Description: "Verify dashboard config/versions returns a JSON array (may be empty on a fresh environment)",
		Tags:        []string{"dashboard", "deploy"},
		Fn:          testDashboardConfigVersions,
	})
}

func testDashboardConfigVersions(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	url := fmt.Sprintf("http://localhost:%s/api/router/config/versions", localPort)
	if opts.Verbose {
		fmt.Printf("[Dashboard] GET %s\n", url)
	}

	httpClient := &http.Client{Timeout: 10 * time.Second}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("create versions request: %w", err)
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("config/versions request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("config/versions: expected 200, got %d: %s", resp.StatusCode, truncateString(string(body), 200))
	}

	var versions []interface{}
	if err := json.Unmarshal(body, &versions); err != nil {
		return fmt.Errorf("config/versions response is not a JSON array: %w (body: %s)", err, truncateString(string(body), 200))
	}

	if opts.Verbose {
		fmt.Printf("[Dashboard] config/versions OK: count=%d\n", len(versions))
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"version_count": len(versions)})
	}

	return nil
}
