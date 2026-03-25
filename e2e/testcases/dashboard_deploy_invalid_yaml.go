package testcases

import (
	"bytes"
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
	pkgtestcases.Register("dashboard-deploy-invalid-yaml", pkgtestcases.TestCase{
		Description: "Verify dashboard deploy/preview rejects malformed YAML with a 400 error",
		Tags:        []string{"dashboard", "deploy", "validation"},
		Fn:          testDashboardDeployInvalidYAML,
	})
}

func testDashboardDeployInvalidYAML(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	payload := map[string]string{"yaml": "invalid: yaml: [unclosed bracket"}
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal payload: %w", err)
	}

	url := fmt.Sprintf("http://localhost:%s/api/router/config/deploy/preview", localPort)
	if opts.Verbose {
		fmt.Printf("[Dashboard] POST %s (invalid YAML — expecting 400)\n", url)
	}

	httpClient := &http.Client{Timeout: 10 * time.Second}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(body))
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusBadRequest {
		return fmt.Errorf("expected 400 for invalid YAML, got %d: %s", resp.StatusCode, truncateString(string(respBody), 200))
	}

	if opts.Verbose {
		fmt.Printf("[Dashboard] deploy-invalid-yaml OK: got expected 400\n")
	}

	return nil
}

// dashboardKeys returns the map keys as a slice for error messages.
func dashboardKeys(m map[string]interface{}) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}
