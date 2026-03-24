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
	pkgtestcases.Register("dashboard-deploy-preview", pkgtestcases.TestCase{
		Description: "Verify dashboard deploy/preview returns current and preview YAML for diff display",
		Tags:        []string{"dashboard", "deploy"},
		Fn:          testDashboardDeployPreview,
	})
}

func testDashboardDeployPreview(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	payload := map[string]string{"yaml": "default_model: \"MoM\"\n"}
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal preview payload: %w", err)
	}

	url := fmt.Sprintf("http://localhost:%s/api/router/config/deploy/preview", localPort)
	if opts.Verbose {
		fmt.Printf("[Dashboard] POST %s\n", url)
	}

	httpClient := &http.Client{Timeout: 15 * time.Second}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(body))
	if err != nil {
		return fmt.Errorf("create deploy/preview request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("deploy/preview request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("deploy/preview: expected 200, got %d: %s", resp.StatusCode, truncateString(string(respBody), 300))
	}

	var result map[string]interface{}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return fmt.Errorf("deploy/preview response is not valid JSON: %w", err)
	}

	if _, ok := result["current"]; !ok {
		return fmt.Errorf("deploy/preview: expected 'current' field, got keys: %v", dashboardKeys(result))
	}
	if _, ok := result["preview"]; !ok {
		return fmt.Errorf("deploy/preview: expected 'preview' field, got keys: %v", dashboardKeys(result))
	}

	if opts.Verbose {
		fmt.Printf("[Dashboard] deploy/preview OK: current and preview fields present\n")
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"has_current": true, "has_preview": true})
	}

	return nil
}
