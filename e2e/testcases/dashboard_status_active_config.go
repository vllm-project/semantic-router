package testcases

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("dashboard-status-active-config", pkgtestcases.TestCase{
		Description: "Verify the active config projection names the served config by version and content hash after a deploy and after its rollback (issue #3233)",
		Tags:        []string{"dashboard", "status", "deploy", "rollback"},
		Fn:          testDashboardStatusActiveConfig,
	})
}

// statusProbeDecision names the temporary decision this journey deploys. It
// differs from the rollback journey's probe so neither case can mistake the
// other's leftovers for its own effect.
const statusProbeDecision = "e2e-status-probe"

// statusProbeFragment merges into the dashboard profile fixture
// (e2e/profiles/dashboard/values.yaml) the same way the rollback journey's
// fragment does. Sequence nodes merge by wholesale replacement
// (dashboard/backend/handlers/canonical_transport.go:205-226), so
// routing.decisions holds only this probe while it is deployed, and the
// modelRef resolves against the fixture's base-model so the merged document
// validates. The journey restores the original document before finishing.
const statusProbeFragment = `routing:
  decisions:
    - name: e2e-status-probe
      description: Temporary decision the status journey deploys and reverts
      priority: 2
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: base-model
          use_reasoning: false
`

// The projection refresh runs in its own goroutine after a deploy or a
// rollback has answered (dashboard/backend/handlers/config_projection_deps.go),
// so the journey waits for the projection to catch up instead of reading it
// once and racing the refresh.
const (
	activeProjectionWait     = 20 * time.Second
	activeProjectionInterval = 500 * time.Millisecond
)

// dashboardActiveProjection is the JSON /api/router/config/active-projection
// answers with (ActiveProjectionResponse in dashboard/backend/configprojection).
type dashboardActiveProjection struct {
	ActiveVersion string                     `json:"active_version"`
	Status        string                     `json:"status"`
	LastError     string                     `json:"last_error"`
	Deployment    *dashboardDeploymentRecord `json:"deployment"`
}

// dashboardDeploymentRecord is the part of a persisted deployment projection
// the journey reasons about.
type dashboardDeploymentRecord struct {
	Version    string `json:"version"`
	Source     string `json:"source"`
	YAMLHash   string `json:"yaml_hash"`
	Validation struct {
		Status string `json:"status"`
	} `json:"validation"`
}

// dashboardDeploymentSummary is one entry of /api/router/config/deployments.
type dashboardDeploymentSummary struct {
	Version  string `json:"version"`
	Source   string `json:"source"`
	YAMLHash string `json:"yaml_hash"`
}

// dashboardStatusShape is the part of /api/status a routing deploy must not
// move: which stack the Dashboard believes it runs in and which services it
// reports on. Health flags come from live probes and are recorded in the
// details, not asserted, because they can change for reasons outside this
// journey.
type dashboardStatusShape struct {
	Overall        string
	DeploymentType string
	Services       []string
}

// testDashboardStatusActiveConfig covers the status journey from #3233 as
// behavior. The existing dashboard-status case accepts any non-empty overall
// value, and /api/status carries no config identity at all. What an operator
// needs after a deploy is a status surface that names the config now being
// served, and the Dashboard keeps that in the active projection: a version, a
// source, and the sha256 of the bytes it wrote
// (dashboard/backend/configprojection/builder.go, hashYAML). This case ties
// those claims to the served document on both sides of a rollback.
//
// A deploy mints its own projection version, while a rollback records the
// backup id it restored as the active version (the refreshConfigProjection
// calls in dashboard/backend/handlers/deploy.go), so after the rollback the
// projection must point at exactly the version the operator asked for.
func testDashboardStatusActiveConfig(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
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
	hashBefore := sha256Hex(configBefore)

	statusBefore, err := fetchDashboardStatusShape(ctx, httpClient, baseURL)
	if err != nil {
		return fmt.Errorf("read status before the journey: %w", err)
	}
	projectionBefore, err := fetchDashboardActiveProjection(ctx, httpClient, baseURL, token)
	if err != nil {
		return fmt.Errorf("read active projection before the journey: %w", err)
	}

	deployVersion, err := postDashboardDeploy(ctx, httpClient, baseURL, token, statusProbeFragment)
	if err != nil {
		return err
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
	if !strings.Contains(string(configDeployed), statusProbeDecision) {
		return fmt.Errorf("deployed config does not contain the %s decision: %s", statusProbeDecision, truncateString(string(configDeployed), 400))
	}
	hashDeployed := sha256Hex(configDeployed)
	if hashDeployed == hashBefore {
		return fmt.Errorf("deploy returned success (version %s) but the served config hash is unchanged, so the projection assertions below would be vacuous", deployVersion)
	}

	projectionDeployed, err := awaitDashboardActiveProjection(ctx, httpClient, baseURL, token, func(projection dashboardActiveProjection) bool {
		return projection.Status == "ok" && projection.ActiveVersion != projectionBefore.ActiveVersion
	})
	if err != nil {
		return fmt.Errorf("active projection after deploy: %w", err)
	}
	statusDeployed, err := fetchDashboardStatusShape(ctx, httpClient, baseURL)
	if err != nil {
		return fmt.Errorf("read status after deploy: %w", err)
	}
	if shapeErr := assertDashboardStatusShapeKept(statusBefore, statusDeployed); shapeErr != nil {
		return fmt.Errorf("after deploy: %w", shapeErr)
	}
	if assertErr := assertActiveProjectionNamesConfig(projectionDeployed, hashDeployed); assertErr != nil {
		return fmt.Errorf("after deploy: %w", assertErr)
	}
	if listErr := assertDashboardDeploymentListed(ctx, httpClient, baseURL, token, projectionDeployed.ActiveVersion, hashDeployed); listErr != nil {
		return fmt.Errorf("after deploy: %w", listErr)
	}

	rollbackStatus, rollbackResult, err := postDashboardRollback(ctx, httpClient, baseURL, token, deployVersion)
	if err != nil {
		return err
	}
	if rollbackStatus != http.StatusOK || rollbackResult.Status != "success" {
		return fmt.Errorf("rollback to %s: expected 200 success, got %d %q (message: %s)",
			deployVersion, rollbackStatus, rollbackResult.Status, truncateString(rollbackResult.Message, 300))
	}
	configRestored, err := fetchDashboardConfigYAML(ctx, httpClient, baseURL, token)
	if err != nil {
		return fmt.Errorf("read config after rollback: %w", err)
	}
	// dashboard-deploy-rollback owns the byte-for-byte restore contract. It is
	// re-checked here only because the hash claim below says nothing if the
	// restore itself went wrong.
	if sha256Hex(configRestored) != hashBefore {
		return fmt.Errorf("rollback did not restore the pre-deploy config (before %d bytes, after %d bytes)", len(configBefore), len(configRestored))
	}
	restored = true

	projectionRestored, err := awaitDashboardActiveProjection(ctx, httpClient, baseURL, token, func(projection dashboardActiveProjection) bool {
		return projection.Status == "ok" && projection.ActiveVersion == deployVersion
	})
	if err != nil {
		return fmt.Errorf("active projection after rollback must name the restored version %s: %w", deployVersion, err)
	}
	if assertErr := assertActiveProjectionNamesConfig(projectionRestored, hashBefore); assertErr != nil {
		return fmt.Errorf("after rollback: %w", assertErr)
	}
	if projectionRestored.Deployment.Source != "rollback" {
		return fmt.Errorf("after rollback: expected the active deployment source to be rollback, got %q", projectionRestored.Deployment.Source)
	}
	if listErr := assertDashboardDeploymentListed(ctx, httpClient, baseURL, token, deployVersion, hashBefore); listErr != nil {
		return fmt.Errorf("after rollback: %w", listErr)
	}

	statusAfter, err := fetchDashboardStatusShape(ctx, httpClient, baseURL)
	if err != nil {
		return fmt.Errorf("read status after rollback: %w", err)
	}
	if shapeErr := assertDashboardStatusShapeKept(statusBefore, statusAfter); shapeErr != nil {
		return fmt.Errorf("after rollback: %w", shapeErr)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"deploy_version":                 deployVersion,
			"projection_version_before":      projectionBefore.ActiveVersion,
			"projection_version_deployed":    projectionDeployed.ActiveVersion,
			"projection_version_restored":    projectionRestored.ActiveVersion,
			"projection_source_deployed":     projectionDeployed.Deployment.Source,
			"projection_source_restored":     projectionRestored.Deployment.Source,
			"config_hash_before":             hashBefore,
			"config_hash_deployed":           hashDeployed,
			"config_hash_matches_projection": true,
			"status_overall_before":          statusBefore.Overall,
			"status_overall_deployed":        statusDeployed.Overall,
			"status_overall_after":           statusAfter.Overall,
			"status_services":                statusAfter.Services,
		})
	}
	if opts.Verbose {
		fmt.Printf("[Dashboard] status-active-config OK: projection %s -> %s (deploy) -> %s (rollback), hashes verified\n",
			projectionBefore.ActiveVersion, projectionDeployed.ActiveVersion, projectionRestored.ActiveVersion)
	}
	return nil
}

// assertActiveProjectionNamesConfig checks that the projection points at a
// deployment record whose hash is the served document's and whose validation
// passed. A projection with no deployment record, or one hashing a different
// document, would tell the operator the wrong config is active.
func assertActiveProjectionNamesConfig(projection dashboardActiveProjection, wantHash string) error {
	if projection.LastError != "" {
		return fmt.Errorf("active projection reports last_error %q", projection.LastError)
	}
	if projection.Deployment == nil {
		return fmt.Errorf("active projection %s has no deployment record", projection.ActiveVersion)
	}
	if projection.Deployment.Version != projection.ActiveVersion {
		return fmt.Errorf("active projection %s carries deployment record %s", projection.ActiveVersion, projection.Deployment.Version)
	}
	if projection.Deployment.YAMLHash != wantHash {
		return fmt.Errorf("active projection %s hashes %s, the served config hashes %s", projection.ActiveVersion, projection.Deployment.YAMLHash, wantHash)
	}
	if projection.Deployment.Validation.Status != "ok" {
		return fmt.Errorf("active projection %s records validation status %q, want ok", projection.ActiveVersion, projection.Deployment.Validation.Status)
	}
	return nil
}

// assertDashboardStatusShapeKept checks that /api/status still describes the
// same stack and the same services. Health flags are deliberately left out.
func assertDashboardStatusShapeKept(before, after dashboardStatusShape) error {
	if after.DeploymentType != before.DeploymentType || !equalStringSets(after.Services, before.Services) {
		return fmt.Errorf("status shape changed: deployment_type %q -> %q, services %v -> %v",
			before.DeploymentType, after.DeploymentType, before.Services, after.Services)
	}
	return nil
}

// assertDashboardDeploymentListed checks that the deployment history offers
// the version with the same hash the active projection reports. The list
// orders by creation time at second resolution, so membership is asserted
// rather than position.
func assertDashboardDeploymentListed(ctx context.Context, client *http.Client, baseURL, token, version, wantHash string) error {
	deployments, err := listDashboardConfigDeployments(ctx, client, baseURL, token)
	if err != nil {
		return err
	}
	for _, deployment := range deployments {
		if deployment.Version != version {
			continue
		}
		if deployment.YAMLHash != wantHash {
			return fmt.Errorf("deployment history lists %s with hash %s, the active projection reports %s", version, deployment.YAMLHash, wantHash)
		}
		return nil
	}
	return fmt.Errorf("deployment history does not list version %s (%d entries)", version, len(deployments))
}

// awaitDashboardActiveProjection polls the active projection until accept
// returns true, and reports the last projection it saw when it gives up so a
// failed or stale refresh names its error.
func awaitDashboardActiveProjection(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	token string,
	accept func(dashboardActiveProjection) bool,
) (dashboardActiveProjection, error) {
	deadline := time.Now().Add(activeProjectionWait)
	var last dashboardActiveProjection
	for {
		projection, err := fetchDashboardActiveProjection(ctx, client, baseURL, token)
		if err != nil {
			return dashboardActiveProjection{}, err
		}
		if accept(projection) {
			return projection, nil
		}
		last = projection
		if time.Now().After(deadline) {
			return last, fmt.Errorf("projection did not reach the expected state within %s (active_version %q, status %q, last_error %q)",
				activeProjectionWait, last.ActiveVersion, last.Status, last.LastError)
		}
		select {
		case <-ctx.Done():
			return last, ctx.Err()
		case <-time.After(activeProjectionInterval):
		}
	}
}

func fetchDashboardActiveProjection(ctx context.Context, client *http.Client, baseURL, token string) (dashboardActiveProjection, error) {
	body, err := getDashboardJSON(ctx, client, baseURL+"/api/router/config/active-projection", token)
	if err != nil {
		return dashboardActiveProjection{}, err
	}
	var projection dashboardActiveProjection
	if err := json.Unmarshal(body, &projection); err != nil {
		return dashboardActiveProjection{}, fmt.Errorf("active projection is not JSON: %w (body: %s)", err, truncateString(string(body), 300))
	}
	return projection, nil
}

func listDashboardConfigDeployments(ctx context.Context, client *http.Client, baseURL, token string) ([]dashboardDeploymentSummary, error) {
	body, err := getDashboardJSON(ctx, client, baseURL+"/api/router/config/deployments", token)
	if err != nil {
		return nil, err
	}
	var deployments []dashboardDeploymentSummary
	if err := json.Unmarshal(body, &deployments); err != nil {
		return nil, fmt.Errorf("deployments response is not a JSON array: %w (body: %s)", err, truncateString(string(body), 300))
	}
	return deployments, nil
}

// fetchDashboardStatusShape reads the public /api/status and keeps the parts
// the journey compares.
func fetchDashboardStatusShape(ctx context.Context, client *http.Client, baseURL string) (dashboardStatusShape, error) {
	body, err := getDashboardJSON(ctx, client, baseURL+"/api/status", "")
	if err != nil {
		return dashboardStatusShape{}, err
	}
	var status struct {
		Overall        string `json:"overall"`
		DeploymentType string `json:"deployment_type"`
		Services       []struct {
			Name string `json:"name"`
		} `json:"services"`
	}
	if err := json.Unmarshal(body, &status); err != nil {
		return dashboardStatusShape{}, fmt.Errorf("status response is not JSON: %w (body: %s)", err, truncateString(string(body), 300))
	}
	if status.Overall == "" {
		return dashboardStatusShape{}, fmt.Errorf("status response has no overall value: %s", truncateString(string(body), 300))
	}
	shape := dashboardStatusShape{Overall: status.Overall, DeploymentType: status.DeploymentType}
	for _, service := range status.Services {
		shape.Services = append(shape.Services, service.Name)
	}
	return shape, nil
}

// getDashboardJSON performs an authenticated GET when token is set and
// returns the body of a 200 answer.
func getDashboardJSON(ctx context.Context, client *http.Client, url, token string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	if token != "" {
		setDashboardAuth(req, token)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("GET %s failed: %w", url, err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GET %s: expected 200, got %d: %s", url, resp.StatusCode, truncateString(string(body), 200))
	}
	return body, nil
}

func sha256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func equalStringSets(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	sortedLeft := append([]string(nil), left...)
	sortedRight := append([]string(nil), right...)
	sort.Strings(sortedLeft)
	sort.Strings(sortedRight)
	for index := range sortedLeft {
		if sortedLeft[index] != sortedRight[index] {
			return false
		}
	}
	return true
}
