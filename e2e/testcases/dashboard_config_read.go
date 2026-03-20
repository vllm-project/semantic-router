package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("dashboard-config-read", pkgtestcases.TestCase{
		Description: "Verify dashboard config endpoints return the router config as JSON and YAML",
		Tags:        []string{"dashboard", "config"},
		Fn:          testDashboardConfigRead,
	})
}

func testDashboardConfigRead(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	httpClient := &http.Client{Timeout: 15 * time.Second}
	baseURL := fmt.Sprintf("http://localhost:%s", localPort)

	configJSON, err := fetchDashboardJSONConfig(ctx, httpClient, baseURL, opts.Verbose)
	if err != nil {
		return err
	}

	yamlSize, err := fetchDashboardYAMLConfig(ctx, httpClient, baseURL, opts.Verbose)
	if err != nil {
		return err
	}

	if opts.Verbose {
		fmt.Printf("[Dashboard] config-read OK: JSON keys=%d, YAML bytes=%d\n", len(configJSON), yamlSize)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"config_json_keys": len(configJSON),
			"config_yaml_size": yamlSize,
		})
	}

	return nil
}

func fetchDashboardJSONConfig(ctx context.Context, client *http.Client, baseURL string, verbose bool) (map[string]interface{}, error) {
	url := baseURL + "/api/router/config/all"
	if verbose {
		fmt.Printf("[Dashboard] GET %s\n", url)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("create JSON config request: %w", err)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("config/all request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("config/all: expected 200, got %d: %s", resp.StatusCode, truncateString(string(body), 200))
	}

	var result map[string]interface{}
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("config/all response is not valid JSON: %w", err)
	}

	if len(result) == 0 {
		return nil, fmt.Errorf("config/all returned empty JSON object")
	}

	return result, nil
}

func fetchDashboardYAMLConfig(ctx context.Context, client *http.Client, baseURL string, verbose bool) (int, error) {
	url := baseURL + "/api/router/config/yaml"
	if verbose {
		fmt.Printf("[Dashboard] GET %s\n", url)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return 0, fmt.Errorf("create YAML config request: %w", err)
	}

	resp, err := client.Do(req)
	if err != nil {
		return 0, fmt.Errorf("config/yaml request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("config/yaml: expected 200, got %d: %s", resp.StatusCode, truncateString(string(body), 200))
	}

	contentType := resp.Header.Get("Content-Type")
	if !strings.Contains(contentType, "yaml") {
		return 0, fmt.Errorf("config/yaml: expected Content-Type to contain 'yaml', got %q", contentType)
	}

	if len(strings.TrimSpace(string(body))) == 0 {
		return 0, fmt.Errorf("config/yaml returned empty body")
	}

	return len(body), nil
}
