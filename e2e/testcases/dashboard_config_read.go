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
	"sigs.k8s.io/yaml"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("dashboard-config-read", pkgtestcases.TestCase{
		Description: "Verify dashboard config endpoints serve the deployed router config as JSON and YAML",
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
	token, err := dashboardAuthToken(ctx, httpClient, baseURL, opts.Verbose)
	if err != nil {
		return err
	}

	configJSON, err := fetchDashboardJSONConfig(ctx, httpClient, baseURL, token, opts.Verbose)
	if err != nil {
		return err
	}
	if err = assertCanonicalConfigIdentity("config/all", configJSON); err != nil {
		return err
	}

	yamlBody, err := fetchDashboardYAMLConfig(ctx, httpClient, baseURL, token, opts.Verbose)
	if err != nil {
		return err
	}
	var configYAML map[string]interface{}
	if err = yaml.Unmarshal(yamlBody, &configYAML); err != nil {
		return fmt.Errorf("config/yaml response is not valid YAML: %w", err)
	}
	if err = assertCanonicalConfigIdentity("config/yaml", configYAML); err != nil {
		return err
	}

	if opts.Verbose {
		fmt.Printf("[Dashboard] config-read OK: JSON keys=%d, YAML bytes=%d, base-model and other_decision present on both reads\n", len(configJSON), len(yamlBody))
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"config_json_keys": len(configJSON),
			"config_yaml_size": len(yamlBody),
		})
	}

	return nil
}

// assertCanonicalConfigIdentity checks that a served config document is the
// deployed dashboard fixture (e2e/profiles/dashboard/values.yaml) rather than
// chart placeholders: the provider surface must declare base-model and the
// routing surface must declare other_decision. Nonempty-body checks alone
// cannot tell those apart, because the chart-default document is also a
// nonempty canonical config.
func assertCanonicalConfigIdentity(source string, doc map[string]interface{}) error {
	if err := assertNamedEntry(doc, "providers", "models", "base-model"); err != nil {
		return fmt.Errorf("%s: %w", source, err)
	}
	if err := assertNamedEntry(doc, "routing", "decisions", "other_decision"); err != nil {
		return fmt.Errorf("%s: %w", source, err)
	}
	return nil
}

// assertNamedEntry asserts doc[section][list] contains an entry whose name
// field equals want. Both read paths decode into JSON-shaped maps, so one
// walker covers the JSON and YAML responses.
func assertNamedEntry(doc map[string]interface{}, section, list, want string) error {
	sectionMap, ok := doc[section].(map[string]interface{})
	if !ok {
		return fmt.Errorf("config has no %q object", section)
	}
	entries, ok := sectionMap[list].([]interface{})
	if !ok {
		return fmt.Errorf("config %s has no %q list", section, list)
	}
	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		entryMap, ok := entry.(map[string]interface{})
		if !ok {
			continue
		}
		name, _ := entryMap["name"].(string)
		if name == want {
			return nil
		}
		names = append(names, name)
	}
	return fmt.Errorf("%s.%s does not declare %q, got %v", section, list, want, names)
}

func fetchDashboardJSONConfig(ctx context.Context, client *http.Client, baseURL, token string, verbose bool) (map[string]interface{}, error) {
	url := baseURL + "/api/router/config/all"
	if verbose {
		fmt.Printf("[Dashboard] GET %s\n", url)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("create JSON config request: %w", err)
	}
	setDashboardAuth(req, token)

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

func fetchDashboardYAMLConfig(ctx context.Context, client *http.Client, baseURL, token string, verbose bool) ([]byte, error) {
	url := baseURL + "/api/router/config/yaml"
	if verbose {
		fmt.Printf("[Dashboard] GET %s\n", url)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("create YAML config request: %w", err)
	}
	setDashboardAuth(req, token)

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("config/yaml request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("config/yaml: expected 200, got %d: %s", resp.StatusCode, truncateString(string(body), 200))
	}

	contentType := resp.Header.Get("Content-Type")
	if !strings.Contains(contentType, "yaml") {
		return nil, fmt.Errorf("config/yaml: expected Content-Type to contain 'yaml', got %q", contentType)
	}

	if len(strings.TrimSpace(string(body))) == 0 {
		return nil, fmt.Errorf("config/yaml returned empty body")
	}

	return body, nil
}
