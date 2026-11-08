package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("dashboard-logs-diagnostics", pkgtestcases.TestCase{
		Description: "Verify an operator can diagnose a rejected deploy from the Dashboard alone: /api/logs states its deployment contract, the audit trail records the rejection, and the served config and active projection stay put (issue #3233)",
		Tags:        []string{"dashboard", "logs", "diagnostics", "validation"},
		Fn:          testDashboardLogsDiagnostics,
	})
}

// The audit row is written after the handler answers, in the same request
// goroutine (dashboard/backend/auth/policy_middleware.go, serveWithRouteAudit),
// so it normally exists before the client reads the response. The journey
// still waits a bounded interval rather than assuming the ordering.
const (
	auditTrailWait     = 10 * time.Second
	auditTrailInterval = 250 * time.Millisecond
)

// dashboardLogsResponse is the bounded log API contract (LogsResponse in
// dashboard/backend/handlers/logs.go).
type dashboardLogsResponse struct {
	DeploymentType string            `json:"deployment_type"`
	Service        string            `json:"service"`
	Supported      bool              `json:"supported"`
	Logs           []json.RawMessage `json:"logs"`
	Count          int               `json:"count"`
	Error          string            `json:"error"`
	Message        string            `json:"message"`
}

// dashboardAuditEntry is one row of /api/admin/audit-logs (AuditLog in
// dashboard/backend/auth/audit_store.go).
type dashboardAuditEntry struct {
	Action     string `json:"action"`
	Method     string `json:"method"`
	Path       string `json:"path"`
	StatusCode int    `json:"statusCode"`
}

// testDashboardLogsDiagnostics covers the diagnostics journey from #3233 as
// behavior rather than HTTP success. The dashboard profile runs on
// Kubernetes, where no bounded log spool is mounted: only the `vllm-sr serve`
// split runtime relays container output into one, and the Kubernetes
// manifests do not. /api/logs must therefore say so explicitly, with
// supported=false and a message, instead of answering an empty success that
// reads as "nothing happened". Input validation still runs ahead of that
// answer: an unknown component is the caller's mistake and is reported as
// such, not hidden behind the unsupported notice.
//
// A rejected deploy writes no log line and no backup; the Dashboard records
// it only in the deploy response and in the audit trail. The second half of
// the case proves that trail is enough to diagnose from: the rejection
// appears as one more client-error row for config.deploy, no success row
// appears, and the served config, the version ledger, and the active
// projection all stay exactly as they were.
func testDashboardLogsDiagnostics(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
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

	logsDefault, err := assertDashboardLogsUnsupported(ctx, httpClient, baseURL, token, "", "router")
	if err != nil {
		return err
	}
	if _, allErr := assertDashboardLogsUnsupported(ctx, httpClient, baseURL, token, "all", "all"); allErr != nil {
		return allErr
	}
	if componentErr := assertDashboardLogsRejectsComponent(ctx, httpClient, baseURL, token, "e2e-not-a-component"); componentErr != nil {
		return componentErr
	}

	configBefore, err := fetchDashboardConfigYAML(ctx, httpClient, baseURL, token)
	if err != nil {
		return fmt.Errorf("read config before the rejected deploy: %w", err)
	}
	versionsBefore, err := countDashboardConfigVersions(ctx, httpClient, baseURL, token)
	if err != nil {
		return fmt.Errorf("read versions before the rejected deploy: %w", err)
	}
	projectionBefore, err := fetchDashboardActiveProjection(ctx, httpClient, baseURL, token)
	if err != nil {
		return fmt.Errorf("read active projection before the rejected deploy: %w", err)
	}
	failuresBefore, err := countDashboardAuditEntries(ctx, httpClient, baseURL, token, "config.deploy", "4xx")
	if err != nil {
		return fmt.Errorf("count rejected deploys in the audit trail before: %w", err)
	}
	successesBefore, err := countDashboardAuditEntries(ctx, httpClient, baseURL, token, "config.deploy", "success")
	if err != nil {
		return fmt.Errorf("count accepted deploys in the audit trail before: %w", err)
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

	newest, err := awaitDashboardAuditFailure(ctx, httpClient, baseURL, token, failuresBefore)
	if err != nil {
		return err
	}
	if newest.Method != http.MethodPost || newest.Path != "/api/router/config/deploy" || newest.StatusCode != http.StatusBadRequest {
		return fmt.Errorf("newest rejected config.deploy audit row is %s %s with status %d, want POST /api/router/config/deploy with 400",
			newest.Method, newest.Path, newest.StatusCode)
	}
	successesAfter, err := countDashboardAuditEntries(ctx, httpClient, baseURL, token, "config.deploy", "success")
	if err != nil {
		return fmt.Errorf("count accepted deploys in the audit trail after: %w", err)
	}
	if successesAfter != successesBefore {
		return fmt.Errorf("a rejected deploy must not add a success row to the audit trail (before %d, after %d)", successesBefore, successesAfter)
	}

	if unchangedErr := assertDashboardConfigUnchanged(ctx, httpClient, baseURL, token, configBefore, versionsBefore); unchangedErr != nil {
		return unchangedErr
	}
	projectionAfter, err := fetchDashboardActiveProjection(ctx, httpClient, baseURL, token)
	if err != nil {
		return fmt.Errorf("read active projection after the rejected deploy: %w", err)
	}
	if projectionErr := assertActiveProjectionUnchanged(projectionBefore, projectionAfter); projectionErr != nil {
		return projectionErr
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"logs_deployment_type":      logsDefault.DeploymentType,
			"logs_supported":            logsDefault.Supported,
			"logs_message":              logsDefault.Message,
			"rejection_error":           rejectionError,
			"audit_rejections_before":   failuresBefore,
			"audit_rejections_after":    failuresBefore + 1,
			"audit_successes_unchanged": successesBefore,
			"config_unchanged":          true,
			"projection_version":        projectionAfter.ActiveVersion,
		})
	}
	if opts.Verbose {
		fmt.Printf("[Dashboard] logs-diagnostics OK: logs %s/supported=%t, rejection %s audited as %d, config and projection unchanged\n",
			logsDefault.DeploymentType, logsDefault.Supported, rejectionError, newest.StatusCode)
	}
	return nil
}

// assertDashboardLogsUnsupported reads /api/logs for the given component and
// pins the explicit unsupported answer this profile must give: a 200 with
// supported=false, the unsupported deployment type, no entries, no error,
// and a message that tells the operator why. The exact wording belongs to
// the handler and is recorded in the details rather than asserted.
func assertDashboardLogsUnsupported(ctx context.Context, client *http.Client, baseURL, token, component, wantService string) (dashboardLogsResponse, error) {
	target := baseURL + "/api/logs"
	if component != "" {
		target += "?component=" + url.QueryEscape(component)
	}
	body, err := getDashboardJSON(ctx, client, target, token)
	if err != nil {
		return dashboardLogsResponse{}, err
	}
	var logs dashboardLogsResponse
	if decodeErr := json.Unmarshal(body, &logs); decodeErr != nil {
		return dashboardLogsResponse{}, fmt.Errorf("logs response is not JSON: %w (body: %s)", decodeErr, truncateString(string(body), 300))
	}
	if logs.Supported || logs.DeploymentType != "unsupported" {
		return logs, fmt.Errorf("logs for component %q must state they are unsupported on this profile, got supported=%t deployment_type=%q",
			wantService, logs.Supported, logs.DeploymentType)
	}
	if logs.Message == "" || logs.Error != "" {
		return logs, fmt.Errorf("unsupported logs must carry a message and no error, got message=%q error=%q", logs.Message, logs.Error)
	}
	if logs.Logs == nil || len(logs.Logs) != 0 || logs.Count != 0 {
		return logs, fmt.Errorf("unsupported logs must answer an empty entry list with count 0, got %d entries (count %d, list present %t)",
			len(logs.Logs), logs.Count, logs.Logs != nil)
	}
	if logs.Service != wantService {
		return logs, fmt.Errorf("logs response names service %q, want %q", logs.Service, wantService)
	}
	return logs, nil
}

// assertDashboardLogsRejectsComponent pins that an unknown component is
// answered as the caller's error, ahead of the spool check.
func assertDashboardLogsRejectsComponent(ctx context.Context, client *http.Client, baseURL, token, component string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/api/logs?component="+url.QueryEscape(component), nil)
	if err != nil {
		return err
	}
	setDashboardAuth(req, token)

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("logs request with component %q failed: %w", component, err)
	}
	body, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		return fmt.Errorf("expected 400 for logs component %q, got %d: %s", component, resp.StatusCode, truncateString(string(body), 200))
	}
	var rejection struct {
		Error string `json:"error"`
	}
	if decodeErr := json.Unmarshal(body, &rejection); decodeErr != nil {
		return fmt.Errorf("logs rejection is not JSON: %w (body: %s)", decodeErr, truncateString(string(body), 200))
	}
	if rejection.Error != "invalid_component" {
		return fmt.Errorf("expected error=invalid_component for logs component %q, got %q", component, rejection.Error)
	}
	return nil
}

// awaitDashboardAuditFailure waits until the audit trail holds exactly one
// more client-error row for config.deploy than before and returns the newest
// such row. The count is the proof that the row is this journey's; row
// timestamps are the pod's clock at second resolution and are not compared
// with the runner's.
func awaitDashboardAuditFailure(ctx context.Context, client *http.Client, baseURL, token string, failuresBefore int) (dashboardAuditEntry, error) {
	deadline := time.Now().Add(auditTrailWait)
	for {
		entries, total, err := queryDashboardAuditTrail(ctx, client, baseURL, token, "config.deploy", "4xx", 1)
		if err != nil {
			return dashboardAuditEntry{}, fmt.Errorf("read rejected deploys from the audit trail: %w", err)
		}
		if total == failuresBefore+1 && len(entries) == 1 {
			return entries[0], nil
		}
		if total > failuresBefore+1 {
			return dashboardAuditEntry{}, fmt.Errorf("one rejected deploy must add exactly one client-error audit row (before %d, after %d)", failuresBefore, total)
		}
		if time.Now().After(deadline) {
			return dashboardAuditEntry{}, fmt.Errorf("the rejected deploy did not appear in the audit trail within %s (client-error rows before %d, now %d)",
				auditTrailWait, failuresBefore, total)
		}
		select {
		case <-ctx.Done():
			return dashboardAuditEntry{}, ctx.Err()
		case <-time.After(auditTrailInterval):
		}
	}
}

func countDashboardAuditEntries(ctx context.Context, client *http.Client, baseURL, token, action, status string) (int, error) {
	_, total, err := queryDashboardAuditTrail(ctx, client, baseURL, token, action, status, 1)
	return total, err
}

// queryDashboardAuditTrail reads the newest rows for one action and outcome
// class through the paginated audit API and returns them with the total
// match count (AuditLogPageResponse in dashboard/backend/auth).
func queryDashboardAuditTrail(ctx context.Context, client *http.Client, baseURL, token, action, status string, limit int) ([]dashboardAuditEntry, int, error) {
	query := url.Values{}
	query.Set("action", action)
	query.Set("status", status)
	query.Set("sort", "createdAt")
	query.Set("order", "desc")
	query.Set("limit", fmt.Sprintf("%d", limit))
	query.Set("page", "1")

	body, err := getDashboardJSON(ctx, client, baseURL+"/api/admin/audit-logs?"+query.Encode(), token)
	if err != nil {
		return nil, 0, err
	}
	var page struct {
		Logs  []dashboardAuditEntry `json:"logs"`
		Total int                   `json:"total"`
	}
	if decodeErr := json.Unmarshal(body, &page); decodeErr != nil {
		return nil, 0, fmt.Errorf("audit trail response is not JSON: %w (body: %s)", decodeErr, truncateString(string(body), 300))
	}
	return page.Logs, page.Total, nil
}

// assertActiveProjectionUnchanged compares the fields of the active
// projection an operator reads to learn what is being served.
func assertActiveProjectionUnchanged(before, after dashboardActiveProjection) error {
	if after.ActiveVersion != before.ActiveVersion || after.Status != before.Status {
		return fmt.Errorf("a rejected deploy must not move the active projection (version %q/%s -> %q/%s)",
			before.ActiveVersion, before.Status, after.ActiveVersion, after.Status)
	}
	beforeHash, afterHash := "", ""
	if before.Deployment != nil {
		beforeHash = before.Deployment.YAMLHash
	}
	if after.Deployment != nil {
		afterHash = after.Deployment.YAMLHash
	}
	if beforeHash != afterHash {
		return fmt.Errorf("a rejected deploy must not change the active deployment hash (%s -> %s)", beforeHash, afterHash)
	}
	return nil
}
