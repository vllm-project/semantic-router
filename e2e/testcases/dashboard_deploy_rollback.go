package testcases

import (
	"bytes"
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
	pkgtestcases.Register("dashboard-deploy-rollback", pkgtestcases.TestCase{
		Description: "Verify rollback restores the exact pre-deploy config and the version ledger records both operations (issue #3233)",
		Tags:        []string{"dashboard", "deploy", "rollback"},
		Fn:          testDashboardDeployRollback,
	})
}

// rollbackProbeDecision names the temporary decision the journey deploys.
// Its presence in the served YAML proves the deploy took effect, and the
// byte-for-byte restore assertion proves rollback removed every trace of it.
const rollbackProbeDecision = "e2e-rollback-probe"

// rollbackProbeFragment merges into the dashboard profile fixture
// (e2e/profiles/dashboard/values.yaml). Sequence nodes merge by wholesale
// replacement (dashboard/backend/handlers/canonical_transport.go:205-226), so
// routing.decisions consists of only this probe for the deployed interval.
// The modelRef points at the fixture's base-model, so the merged document
// passes config validation. No routed traffic flows while the probe is
// active: the journey only calls the dashboard API, and it restores the
// original document before finishing.
const rollbackProbeFragment = `routing:
  decisions:
    - name: e2e-rollback-probe
      description: Temporary decision the rollback journey deploys and reverts
      priority: 2
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: base-model
          use_reasoning: false
`

// rollbackMissingVersion is a well-formed version id that never exists: the
// backup namer only mints ids at request time (config_backups.go).
const rollbackMissingVersion = "19700101-000000"

// dashboardDeployResult is the JSON both mutating endpoints answer with
// (DeployResponse in dashboard/backend/handlers/deploy.go).
type dashboardDeployResult struct {
	Status  string `json:"status"`
	Version string `json:"version"`
	Error   string `json:"error"`
	Message string `json:"message"`
}

// testDashboardDeployRollback covers the deploy-then-rollback journey from
// #3233. The safe-failure case proved a rejected deploy changes nothing; this
// case proves an accepted deploy is fully reversible. The operator-facing
// invariant is byte-for-byte: after rolling back to the version a deploy
// returned, GET /api/router/config/yaml serves exactly the pre-deploy bytes.
// The ledger assertions compare version-id sets rather than counts: the
// deploy adds exactly its own backup id, and rollback leaves every existing
// id in place, so the target stays available for a second rollback.
//
// The negative probes run first, against pristine state: rollback is closed
// to anonymous callers, and a rollback to an unknown version must fail
// without disturbing the active config. rollbackDirectWrite rejects the
// unknown version before it snapshots or writes, and this case pins that
// order from the outside.
func testDashboardDeployRollback(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
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
		return fmt.Errorf("read config before the journey: %w", err)
	}
	versionsBefore, err := listDashboardConfigVersionIDs(ctx, httpClient, baseURL, token)
	if err != nil {
		return fmt.Errorf("read versions before the journey: %w", err)
	}

	anonStatus, err := postAnonymousDashboardRollback(ctx, httpClient, baseURL)
	if err != nil {
		return err
	}

	if rejectionErr := assertRollbackToMissingVersionRejected(ctx, httpClient, baseURL, token, configBefore); rejectionErr != nil {
		return rejectionErr
	}

	deployVersion, err := postDashboardDeploy(ctx, httpClient, baseURL, token, rollbackProbeFragment)
	if err != nil {
		return err
	}
	if containsVersionID(versionsBefore, deployVersion) {
		return fmt.Errorf("deploy returned version %s, which already existed before the journey; the ledger assertions below cannot be trusted", deployVersion)
	}

	// From here the active config carries the probe. If any later step fails,
	// restore best-effort so the rest of the suite does not run against the
	// journey's leftovers.
	restored := false
	defer func() {
		if restored {
			return
		}
		if _, _, rollbackErr := postDashboardRollback(ctx, httpClient, baseURL, token, deployVersion); rollbackErr != nil && opts.Verbose {
			fmt.Printf("[Dashboard] cleanup rollback to %s failed: %v\n", deployVersion, rollbackErr)
		}
	}()

	configDeployed, err := fetchDashboardConfigYAML(ctx, httpClient, baseURL, token)
	if err != nil {
		return fmt.Errorf("read config after deploy: %w", err)
	}
	if bytes.Equal(configDeployed, configBefore) {
		return fmt.Errorf("deploy returned success (version %s) but the served config is unchanged, so the rollback assertions below would be vacuous", deployVersion)
	}
	if !strings.Contains(string(configDeployed), rollbackProbeDecision) {
		return fmt.Errorf("deployed config does not contain the %s decision: %s", rollbackProbeDecision, truncateString(string(configDeployed), 400))
	}

	versionsAfterDeploy, err := listDashboardConfigVersionIDs(ctx, httpClient, baseURL, token)
	if err != nil {
		return fmt.Errorf("read versions after deploy: %w", err)
	}
	if !containsVersionID(versionsAfterDeploy, deployVersion) {
		return fmt.Errorf("deploy returned version %s but the version list does not offer it: %v", deployVersion, versionsAfterDeploy)
	}
	if newAfterDeploy := versionIDsNotIn(versionsAfterDeploy, versionsBefore); len(newAfterDeploy) != 1 {
		return fmt.Errorf("deploy must add exactly one version id (its backup %s), got new ids %v", deployVersion, newAfterDeploy)
	}
	// Deploy trims the ledger to its retention bound after writing the
	// backup. The bound's value is the handler's business; the journey pins
	// the shape of the trim instead: only the oldest backups may drop out.
	// Ids sort chronologically as strings (config_backups.go names them
	// 20060102-150405), the same order cleanupBackups prunes by.
	for _, pruned := range versionIDsNotIn(versionsBefore, versionsAfterDeploy) {
		for _, kept := range versionsAfterDeploy {
			if kept != deployVersion && pruned > kept {
				return fmt.Errorf("deploy pruned backup %s while keeping older backup %s; retention must drop the oldest first", pruned, kept)
			}
		}
	}

	rollbackStatus, rollbackResult, err := postDashboardRollback(ctx, httpClient, baseURL, token, deployVersion)
	if err != nil {
		return err
	}
	if rollbackStatus != http.StatusOK || rollbackResult.Status != "success" {
		return fmt.Errorf("rollback to %s: expected 200 success, got %d %q (message: %s)",
			deployVersion, rollbackStatus, rollbackResult.Status, truncateString(rollbackResult.Message, 300))
	}
	if rollbackResult.Version != deployVersion {
		return fmt.Errorf("rollback answered with version %q, expected %q", rollbackResult.Version, deployVersion)
	}

	configRestored, err := fetchDashboardConfigYAML(ctx, httpClient, baseURL, token)
	if err != nil {
		return fmt.Errorf("read config after rollback: %w", err)
	}
	if !bytes.Equal(configRestored, configBefore) {
		return fmt.Errorf("rollback must restore the pre-deploy config byte for byte (before %d bytes, after %d bytes)",
			len(configBefore), len(configRestored))
	}
	restored = true

	versionsAfterRollback, err := listDashboardConfigVersionIDs(ctx, httpClient, baseURL, token)
	if err != nil {
		return fmt.Errorf("read versions after rollback: %w", err)
	}
	// Rollback snapshots the outgoing config and never prunes. Backup ids
	// have second resolution (config_backups.go names files
	// config.20060102-150405.yaml), so a snapshot landing in the same second
	// as the deploy reuses the deploy backup's id and the id set is
	// unchanged; in a later second it appears as exactly one new id. The set
	// comparison holds either way, so the journey never waits out a second
	// boundary.
	if pruned := versionIDsNotIn(versionsAfterDeploy, versionsAfterRollback); len(pruned) > 0 {
		return fmt.Errorf("rollback removed %v from the version list; rolling back twice to the same version relies on every backup surviving", pruned)
	}
	newAfterRollback := versionIDsNotIn(versionsAfterRollback, versionsAfterDeploy)
	if len(newAfterRollback) > 1 {
		return fmt.Errorf("rollback may add at most its pre-rollback snapshot to the version list, got new ids %v", newAfterRollback)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"deploy_version":                  deployVersion,
			"config_bytes":                    len(configBefore),
			"config_restored_byte_for_byte":   true,
			"versions_before":                 len(versionsBefore),
			"versions_after_deploy":           len(versionsAfterDeploy),
			"versions_after_rollback":         len(versionsAfterRollback),
			"pre_rollback_snapshot_reused_id": len(newAfterRollback) == 0,
			"unauthenticated_rollback_status": anonStatus,
		})
	}
	if opts.Verbose {
		fmt.Printf("[Dashboard] deploy-rollback OK: version %s deployed and reverted, %d -> %d -> %d versions\n",
			deployVersion, len(versionsBefore), len(versionsAfterDeploy), len(versionsAfterRollback))
	}
	return nil
}

// postDashboardDeploy deploys the fragment in merge mode and returns the
// backup version the dashboard minted for the outgoing config.
func postDashboardDeploy(ctx context.Context, client *http.Client, baseURL, token, fragment string) (string, error) {
	payload, err := json.Marshal(map[string]string{
		"yaml": fragment,
		"mode": "merge",
	})
	if err != nil {
		return "", err
	}

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

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("expected 200 for the probe deploy, got %d: %s", resp.StatusCode, truncateString(string(body), 300))
	}
	var result dashboardDeployResult
	if decodeErr := json.Unmarshal(body, &result); decodeErr != nil {
		return "", fmt.Errorf("deploy response is not JSON: %w (body: %s)", decodeErr, truncateString(string(body), 300))
	}
	if result.Status != "success" || result.Version == "" {
		return "", fmt.Errorf("expected a success response naming the backup version, got status %q version %q (message: %s)",
			result.Status, result.Version, truncateString(result.Message, 300))
	}
	return result.Version, nil
}

// postDashboardRollback requests a rollback and returns the HTTP status with
// the decoded response so callers can assert both outcomes.
func postDashboardRollback(ctx context.Context, client *http.Client, baseURL, token, version string) (int, dashboardDeployResult, error) {
	payload, err := json.Marshal(map[string]string{"version": version})
	if err != nil {
		return 0, dashboardDeployResult{}, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/api/router/config/rollback", bytes.NewReader(payload))
	if err != nil {
		return 0, dashboardDeployResult{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	setDashboardAuth(req, token)

	resp, err := client.Do(req)
	if err != nil {
		return 0, dashboardDeployResult{}, fmt.Errorf("rollback request failed: %w", err)
	}
	body, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()

	var result dashboardDeployResult
	if decodeErr := json.Unmarshal(body, &result); decodeErr != nil {
		return resp.StatusCode, dashboardDeployResult{}, fmt.Errorf("rollback response is not JSON: %w (body: %s)",
			decodeErr, truncateString(string(body), 300))
	}
	return resp.StatusCode, result, nil
}

// assertRollbackToMissingVersionRejected pins the failure mode of a rollback
// to a version that does not exist: 404 version_not_found, active config
// untouched.
func assertRollbackToMissingVersionRejected(ctx context.Context, client *http.Client, baseURL, token string, configBefore []byte) error {
	status, result, err := postDashboardRollback(ctx, client, baseURL, token, rollbackMissingVersion)
	if err != nil {
		return err
	}
	if status != http.StatusNotFound {
		return fmt.Errorf("expected 404 for a rollback to missing version %s, got %d", rollbackMissingVersion, status)
	}
	if result.Error != "version_not_found" {
		return fmt.Errorf("expected error=version_not_found, got %q (message: %s)", result.Error, truncateString(result.Message, 300))
	}

	configAfter, err := fetchDashboardConfigYAML(ctx, client, baseURL, token)
	if err != nil {
		return fmt.Errorf("read config after the rejected rollback: %w", err)
	}
	if !bytes.Equal(configBefore, configAfter) {
		return fmt.Errorf("a rejected rollback must not modify the active config (before %d bytes, after %d bytes)",
			len(configBefore), len(configAfter))
	}
	return nil
}

// postAnonymousDashboardRollback asserts the mutating surface is closed to
// anonymous callers.
func postAnonymousDashboardRollback(ctx context.Context, client *http.Client, baseURL string) (int, error) {
	payload, err := json.Marshal(map[string]string{"version": rollbackMissingVersion})
	if err != nil {
		return 0, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/api/router/config/rollback", bytes.NewReader(payload))
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return 0, fmt.Errorf("unauthenticated rollback request failed: %w", err)
	}
	body, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()

	if resp.StatusCode != http.StatusUnauthorized {
		return resp.StatusCode, fmt.Errorf("expected 401 for an unauthenticated rollback, got %d: %s",
			resp.StatusCode, truncateString(string(body), 200))
	}
	return resp.StatusCode, nil
}

// listDashboardConfigVersionIDs returns the backup version ids the dashboard
// currently offers, so callers can assert membership as well as count.
func listDashboardConfigVersionIDs(ctx context.Context, client *http.Client, baseURL, token string) ([]string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/api/router/config/versions", nil)
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

	var versions []struct {
		Version string `json:"version"`
	}
	if err := json.Unmarshal(body, &versions); err != nil {
		return nil, fmt.Errorf("versions response is not a JSON array: %w (body: %s)", err, truncateString(string(body), 200))
	}
	ids := make([]string, 0, len(versions))
	for _, version := range versions {
		ids = append(ids, version.Version)
	}
	return ids, nil
}

func containsVersionID(ids []string, id string) bool {
	for _, candidate := range ids {
		if candidate == id {
			return true
		}
	}
	return false
}

// versionIDsNotIn returns the ids absent from baseline, in the order the
// dashboard listed them.
func versionIDsNotIn(ids []string, baseline []string) []string {
	var diff []string
	for _, id := range ids {
		if !containsVersionID(baseline, id) {
			diff = append(diff, id)
		}
	}
	return diff
}
