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

	yamlSize, err := fetchDashboardYAMLConfig(ctx, httpClient, baseURL, token, opts.Verbose)
	if err != nil {
		return err
	}

	if opts.Verbose {
		fmt.Printf("[Dashboard] config-read OK: JSON keys=%d, YAML bytes=%d, base-model and other_decision present on both reads\n", len(configJSON), yamlSize)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"config_json_keys": len(configJSON),
			"config_yaml_size": yamlSize,
		})
	}

	return nil
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

	if err := assertDashboardCanonicalConfig(result); err != nil {
		return nil, fmt.Errorf("config/all: %w", err)
	}
	if err := assertCanonicalConfigIdentity(result); err != nil {
		return nil, fmt.Errorf("config/all: %w", err)
	}

	return result, nil
}

func fetchDashboardYAMLConfig(ctx context.Context, client *http.Client, baseURL, token string, verbose bool) (int, error) {
	url := baseURL + "/api/router/config/yaml"
	if verbose {
		fmt.Printf("[Dashboard] GET %s\n", url)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return 0, fmt.Errorf("create YAML config request: %w", err)
	}
	setDashboardAuth(req, token)

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

	var document map[string]interface{}
	if err := yaml.Unmarshal(body, &document); err != nil {
		return 0, fmt.Errorf("config/yaml is not valid YAML: %w", err)
	}
	if err := assertDashboardCanonicalConfig(document); err != nil {
		return 0, fmt.Errorf("config/yaml: %w", err)
	}
	if err := assertCanonicalConfigIdentity(document); err != nil {
		return 0, fmt.Errorf("config/yaml: %w", err)
	}
	return len(body), nil
}

// Check the deployed document, not just whether the file endpoint returned bytes.
// The Router's strict parser owns full validation; this assertion proves the
// Dashboard is reading that canonical provider/routing document.
func assertDashboardCanonicalConfig(document map[string]interface{}) error {
	if document["version"] != "v0.3" {
		return fmt.Errorf("expected canonical version v0.3, got %v", document["version"])
	}
	providers, ok := document["providers"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("canonical config must declare providers")
	}
	models, ok := providers["models"].([]interface{})
	if !ok || len(models) == 0 {
		return fmt.Errorf("canonical config must declare provider models")
	}
	routing, ok := document["routing"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("canonical config must declare routing")
	}
	decisions, ok := routing["decisions"].([]interface{})
	if !ok || len(decisions) == 0 {
		return fmt.Errorf("canonical config must declare routing decisions")
	}
	return nil
}

// assertCanonicalConfigIdentity checks that a served config document is the
// deployed dashboard fixture (e2e/profiles/dashboard/values.yaml) rather than
// chart placeholders: the provider surface must declare base-model and the
// routing surface must declare other_decision. Canonical-shape checks alone
// cannot tell those apart, because a placeholder document can also be a
// canonical v0.3 config with nonempty models and decisions.
func assertCanonicalConfigIdentity(doc map[string]interface{}) error {
	if err := assertNamedEntry(doc, "providers", "models", "base-model"); err != nil {
		return err
	}
	if err := assertNamedEntry(doc, "routing", "decisions", "other_decision"); err != nil {
		return err
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
