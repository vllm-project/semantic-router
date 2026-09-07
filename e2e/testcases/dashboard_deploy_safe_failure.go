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
	pkgtestcases.Register("dashboard-deploy-safe-failure", pkgtestcases.TestCase{
		Description: "Verify a semantically invalid deploy is rejected without disturbing the active config (issue #3233)",
		Tags:        []string{"dashboard", "deploy", "validation", "safe-failure"},
		Fn:          testDashboardDeploySafeFailure,
	})
}

// safeFailureFragment is syntactically valid YAML that cannot validate: it
// routes to a model nothing defines. It clears the parse stage, so the only
// thing that can reject it is config validation.
//
// The fragment carries its own modelCards deliberately. The unknown-model check
// in the router config validator only runs when the merged document declares a
// model surface, and the dashboard profile's router config
// (e2e/profiles/dashboard/values.yaml) still uses the pre-v0.3 schema and
// declares none. Without these modelCards the merged document validates, the
// deploy succeeds, and this case would silently rewrite the active config
// instead of asserting anything.
const safeFailureFragment = `routing:
  modelCards:
    - name: base-model
  decisions:
    - name: e2e-safe-failure
      description: Deploy probe that must be rejected by config validation
      priority: 3
      rules:
        operator: OR
        conditions:
          - type: domain
            name: other
      modelRefs:
        - model: e2e-nonexistent-model
          use_reasoning: false
`

// testDashboardDeploySafeFailure covers the "safe failure" journey from #3233.
// The existing negative case asserts only that malformed YAML returns 400
// (e2e/testcases/dashboard_deploy_invalid_yaml.go:63-65). Nothing asserts what
// the operator actually depends on: that a rejected deploy leaves the config
// that is currently serving completely untouched.
//
// deployDirectWrite validates the merged document before it takes a backup or
// writes (dashboard/backend/handlers/deploy.go:254-271), so a rejected deploy
// must leave both the active config and the backup history unchanged.
func testDashboardDeploySafeFailure(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	baseURL := fmt.Sprintf("http://localhost:%s", localPort)
	httpClient := &http.Client{Timeout: 30 * time.Second}

	token, err := dashboardAuthToken(ctx, httpClient, baseURL, opts.Verbose)
	if err != nil {
		return err
	}

	configBefore, err := fetchDashboardConfigYAML(ctx, httpClient, baseURL, token)
	if err != nil {
		return fmt.Errorf("read config before deploy: %w", err)
	}
	versionsBefore, err := countDashboardConfigVersions(ctx, httpClient, baseURL, token)
	if err != nil {
		return fmt.Errorf("read versions before deploy: %w", err)
	}

	payload, err := json.Marshal(map[string]string{
		"yaml": safeFailureFragment,
		"mode": "merge",
	})
	if err != nil {
		return err
	}

	rejectionError, err := postRejectedDashboardDeploy(ctx, httpClient, baseURL, token, payload)
	if err != nil {
		return err
	}

	if err := assertDashboardConfigUnchanged(ctx, httpClient, baseURL, token, configBefore, versionsBefore); err != nil {
		return err
	}

	anonStatus, err := postAnonymousDashboardDeploy(ctx, httpClient, baseURL, payload)
	if err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"rejection_error":               rejectionError,
			"config_bytes":                  len(configBefore),
			"config_unchanged":              true,
			"versions":                      versionsBefore,
			"unauthenticated_deploy_status": anonStatus,
		})
	}
	if opts.Verbose {
		fmt.Printf("[Dashboard] safe-failure OK: rejected with %s, config and %d versions unchanged\n",
			rejectionError, versionsBefore)
	}
	return nil
}

// postRejectedDashboardDeploy sends the invalid fragment and returns the
// rejection's error code.
func postRejectedDashboardDeploy(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	token string,
	payload []byte,
) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/api/router/config/deploy", bytes.NewReader(payload))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	setDashboardAuth(req, token)

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("deploy request failed: %w", err)
	}
	body, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		return "", fmt.Errorf("expected 400 for a semantically invalid deploy, got %d: %s",
			resp.StatusCode, truncateString(string(body), 300))
	}

	var rejection struct {
		Error   string `json:"error"`
		Message string `json:"message"`
	}
	if decodeErr := json.Unmarshal(body, &rejection); decodeErr != nil {
		return "", fmt.Errorf("deploy rejection is not JSON: %w (body: %s)", decodeErr, truncateString(string(body), 300))
	}
	// The distinction is the point of the case. A yaml_parse_error would mean
	// the fragment was rejected for syntax and validation never ran, which
	// would make the unchanged-config assertions vacuous.
	if rejection.Error != "config_validation_error" {
		return "", fmt.Errorf("expected error=config_validation_error, got %q (message: %s)",
			rejection.Error, truncateString(rejection.Message, 300))
	}
	return rejection.Error, nil
}

// assertDashboardConfigUnchanged is the invariant the case exists to protect.
func assertDashboardConfigUnchanged(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	token string,
	configBefore []byte,
	versionsBefore int,
) error {
	configAfter, err := fetchDashboardConfigYAML(ctx, client, baseURL, token)
	if err != nil {
		return fmt.Errorf("read config after deploy: %w", err)
	}
	if !bytes.Equal(configBefore, configAfter) {
		return fmt.Errorf("a rejected deploy must not modify the active config (before %d bytes, after %d bytes)",
			len(configBefore), len(configAfter))
	}

	versionsAfter, err := countDashboardConfigVersions(ctx, client, baseURL, token)
	if err != nil {
		return fmt.Errorf("read versions after deploy: %w", err)
	}
	if versionsAfter != versionsBefore {
		return fmt.Errorf("a rejected deploy must not create a backup version (before %d, after %d)",
			versionsBefore, versionsAfter)
	}
	return nil
}

// postAnonymousDashboardDeploy asserts the mutating surface is closed to
// anonymous callers.
func postAnonymousDashboardDeploy(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	payload []byte,
) (int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/api/router/config/deploy", bytes.NewReader(payload))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return 0, fmt.Errorf("unauthenticated deploy request failed: %w", err)
	}
	body, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()

	if resp.StatusCode != http.StatusUnauthorized {
		return resp.StatusCode, fmt.Errorf("expected 401 for an unauthenticated deploy, got %d: %s",
			resp.StatusCode, truncateString(string(body), 200))
	}
	return resp.StatusCode, nil
}

func fetchDashboardConfigYAML(ctx context.Context, client *http.Client, baseURL, token string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/api/router/config/yaml", nil)
	if err != nil {
		return nil, err
	}
	setDashboardAuth(req, token)

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("expected 200, got %d: %s", resp.StatusCode, truncateString(string(body), 200))
	}
	return body, nil
}

func countDashboardConfigVersions(ctx context.Context, client *http.Client, baseURL, token string) (int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/api/router/config/versions", nil)
	if err != nil {
		return 0, err
	}
	setDashboardAuth(req, token)

	resp, err := client.Do(req)
	if err != nil {
		return 0, err
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, err
	}
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("expected 200, got %d: %s", resp.StatusCode, truncateString(string(body), 200))
	}

	var versions []map[string]interface{}
	if err := json.Unmarshal(body, &versions); err != nil {
		return 0, fmt.Errorf("versions response is not a JSON array: %w (body: %s)", err, truncateString(string(body), 200))
	}
	return len(versions), nil
}
