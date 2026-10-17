package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("dashboard-anonymous-access-denied", pkgtestcases.TestCase{
		Description: "Verify anonymous and forged sessions are rejected on /api/logs and on deploy, and a deployable fragment sent without a session leaves the served config, the version ledger, and the active projection untouched (issue #3233)",
		Tags:        []string{"dashboard", "auth", "security", "logs", "deploy"},
		Fn:          testDashboardAnonymousAccessDenied,
	})
}

// anonymousProbeFragment declares the e2e-anonymous-probe decision the
// anonymous deploy attempts to land. The fragment is deliberately
// deployable, routed to the fixture's base-model like the rollback journey's
// probe, so the only thing standing between it and the served config is the
// session check. The safe-failure case sends an invalid fragment
// anonymously, which validation would have stopped anyway; this one would
// have applied.
const anonymousProbeFragment = `routing:
  decisions:
    - name: e2e-anonymous-probe
      description: Deployable probe that must never land without a session
      priority: 2
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: base-model
          use_reasoning: false
`

// forgedDashboardSession is shaped like a session but was never issued by
// this Dashboard. Any value the Dashboard did not sign must be refused the
// same way as no session at all.
const forgedDashboardSession = "e2e.forged.session"

// anonymousProbe is one caller the Dashboard must turn away.
type anonymousProbe struct {
	name    string
	session string
}

var anonymousProbes = []anonymousProbe{
	{name: "no session"},
	{name: "forged session", session: forgedDashboardSession},
}

// testDashboardAnonymousAccessDenied covers the negative auth journey from
// #3233 as state, not only status codes. The safe-failure and rollback cases
// already expect 401 from an anonymous mutation; this case sends a fragment
// that would be accepted with a session, with no session and with a forged
// one, and then proves nothing moved: the served bytes, the set of backup
// version ids, and the active projection are all as before. The logs
// surface, which the safe-failure slice left uncovered, is closed the same
// way, and an anonymous caller does not even reach input validation there:
// a bad component is still a 401, not a 400.
func testDashboardAnonymousAccessDenied(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	baseURL := fmt.Sprintf("http://localhost:%s", localPort)
	httpClient := &http.Client{Timeout: 30 * time.Second}

	// The admin session is used only to read state on both sides of the
	// anonymous attempts, never to send them.
	token, err := dashboardAuthToken(ctx, httpClient, baseURL, opts.Verbose)
	if err != nil {
		return err
	}

	configBefore, err := fetchDashboardConfigYAML(ctx, httpClient, baseURL, token)
	if err != nil {
		return fmt.Errorf("read config before the anonymous attempts: %w", err)
	}
	versionsBefore, err := listDashboardConfigVersionIDs(ctx, httpClient, baseURL, token)
	if err != nil {
		return fmt.Errorf("read versions before the anonymous attempts: %w", err)
	}
	projectionBefore, err := fetchDashboardActiveProjection(ctx, httpClient, baseURL, token)
	if err != nil {
		return fmt.Errorf("read active projection before the anonymous attempts: %w", err)
	}

	payload, err := json.Marshal(map[string]string{
		"yaml": anonymousProbeFragment,
		"mode": "merge",
	})
	if err != nil {
		return err
	}

	for _, probe := range anonymousProbes {
		for _, path := range []string{"/api/logs", "/api/logs?component=e2e-not-a-component"} {
			if logsErr := assertDashboardRejectsAnonymous(ctx, httpClient, baseURL, http.MethodGet, path, probe, nil); logsErr != nil {
				return logsErr
			}
		}
		if deployErr := assertDashboardRejectsAnonymous(ctx, httpClient, baseURL, http.MethodPost, "/api/router/config/deploy", probe, payload); deployErr != nil {
			// A deploy that landed must be reverted before the case reports,
			// so the rest of the suite does not run against the probe.
			var landed *anonymousDeployLanded
			if errors.As(deployErr, &landed) {
				if _, _, rollbackErr := postDashboardRollback(ctx, httpClient, baseURL, token, landed.version); rollbackErr != nil && opts.Verbose {
					fmt.Printf("[Dashboard] cleanup rollback to %s failed: %v\n", landed.version, rollbackErr)
				}
			}
			return deployErr
		}
	}

	configAfter, err := fetchDashboardConfigYAML(ctx, httpClient, baseURL, token)
	if err != nil {
		return fmt.Errorf("read config after the anonymous attempts: %w", err)
	}
	if !bytes.Equal(configBefore, configAfter) {
		return fmt.Errorf("anonymous deploy attempts must not modify the active config (before %d bytes, after %d bytes)", len(configBefore), len(configAfter))
	}
	versionsAfter, err := listDashboardConfigVersionIDs(ctx, httpClient, baseURL, token)
	if err != nil {
		return fmt.Errorf("read versions after the anonymous attempts: %w", err)
	}
	if added := versionIDsNotIn(versionsAfter, versionsBefore); len(added) > 0 {
		return fmt.Errorf("anonymous deploy attempts must not create backup versions, got new ids %v", added)
	}
	if removed := versionIDsNotIn(versionsBefore, versionsAfter); len(removed) > 0 {
		return fmt.Errorf("anonymous deploy attempts must not prune backup versions, lost ids %v", removed)
	}
	projectionAfter, err := fetchDashboardActiveProjection(ctx, httpClient, baseURL, token)
	if err != nil {
		return fmt.Errorf("read active projection after the anonymous attempts: %w", err)
	}
	if projectionErr := assertActiveProjectionUnchanged(projectionBefore, projectionAfter); projectionErr != nil {
		return fmt.Errorf("anonymous deploy attempts: %w", projectionErr)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"probes":             len(anonymousProbes),
			"logs_status":        http.StatusUnauthorized,
			"deploy_status":      http.StatusUnauthorized,
			"config_unchanged":   true,
			"versions":           len(versionsBefore),
			"projection_version": projectionAfter.ActiveVersion,
		})
	}
	if opts.Verbose {
		fmt.Printf("[Dashboard] anonymous-access-denied OK: %d callers rejected with 401 on logs and deploy, config, %d versions and projection unchanged\n",
			len(anonymousProbes), len(versionsBefore))
	}
	return nil
}

// anonymousDeployLanded reports a deploy the Dashboard accepted without a
// valid session, carrying the version it minted so the caller can revert.
type anonymousDeployLanded struct {
	probe   string
	version string
}

func (e *anonymousDeployLanded) Error() string {
	return fmt.Sprintf("deploy with %s was accepted and minted version %s; the mutating surface is open", e.probe, e.version)
}

// assertDashboardRejectsAnonymous sends one request as the probe describes
// and requires a 401. It reports a deploy that landed as
// anonymousDeployLanded so the journey can revert it.
func assertDashboardRejectsAnonymous(ctx context.Context, client *http.Client, baseURL, method, path string, probe anonymousProbe, payload []byte) error {
	var body io.Reader
	if payload != nil {
		body = bytes.NewReader(payload)
	}
	req, err := http.NewRequestWithContext(ctx, method, baseURL+path, body)
	if err != nil {
		return err
	}
	if payload != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if probe.session != "" {
		setDashboardAuth(req, probe.session)
	}

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("%s %s with %s failed: %w", method, path, probe.name, err)
	}
	respBody, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()

	if resp.StatusCode == http.StatusUnauthorized {
		return nil
	}
	if method == http.MethodPost && resp.StatusCode == http.StatusOK {
		var result dashboardDeployResult
		if decodeErr := json.Unmarshal(respBody, &result); decodeErr == nil && result.Version != "" {
			return &anonymousDeployLanded{probe: probe.name, version: result.Version}
		}
	}
	return fmt.Errorf("expected 401 for %s %s with %s, got %d: %s", method, path, probe.name, resp.StatusCode, truncateString(string(respBody), 200))
}
