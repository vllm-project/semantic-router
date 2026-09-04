package testcases

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const (
	evaluationSchemaVersion = "evaluation.v1"
	evaluationGateContract  = "evaluation-release-gates.v1"
	evaluationSmokeSuite    = "evaluation-smoke"
	evaluationFixtureTarget = "fixture"
	evaluationPollTimeout   = 2 * time.Minute
)

var evaluationTrackIDs = []string{
	"routing",
	"model_pool",
	"joint",
	"agentic",
	"multimodal",
	"preference",
	"safety",
	"capacity",
}

type dashboardEvaluationRun struct {
	ID       string `json:"id"`
	Status   string `json:"status"`
	Progress struct {
		Percent int `json:"percent"`
	} `json:"progress"`
	Error string `json:"error"`
}

type dashboardEvaluationReport struct {
	SchemaVersion string `json:"schema_version"`
	Run           struct {
		ChangeProfile string `json:"change_profile"`
		EvidenceLevel string `json:"evidence_level"`
	} `json:"run"`
	Summary   evaluationReportSummary `json:"summary"`
	Tracks    []evaluationTrackReport `json:"tracks"`
	Gates     []evaluationGate        `json:"gates"`
	Artifacts []evaluationArtifact    `json:"artifacts"`
}

type evaluationReportSummary struct {
	Verdict          string   `json:"verdict"`
	QualityScore     *float64 `json:"quality_score"`
	FailedGates      int      `json:"failed_gates"`
	UnavailableGates int      `json:"unavailable_gates"`
	Coverage         struct {
		Fraction float64 `json:"fraction"`
	} `json:"coverage"`
}

type evaluationTrackReport struct {
	TrackID string `json:"track_id"`
	Status  string `json:"status"`
}

type evaluationGate struct {
	ID              string   `json:"id"`
	Verdict         string   `json:"verdict"`
	ChangeProfile   string   `json:"change_profile"`
	ContractVersion string   `json:"contract_version"`
	EvidenceRefs    []string `json:"evidence_refs"`
}

type evaluationArtifact struct {
	ID     string `json:"id"`
	Name   string `json:"name"`
	Digest string `json:"digest"`
}

type evaluationCatalogItem struct {
	ID string `json:"id"`
}

func init() {
	pkgtestcases.Register("dashboard-evaluation-plane", pkgtestcases.TestCase{
		Description: "Run all Evaluation Plane tracks and verify evidence, gates, reports, comparison, and cancellation",
		Tags:        []string{"dashboard", "evaluation"},
		Fn:          testDashboardEvaluationPlane,
	})
}

func testDashboardEvaluationPlane(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	baseURL := fmt.Sprintf("http://localhost:%s", localPort)
	httpClient := &http.Client{Timeout: 15 * time.Second}
	token, err := dashboardAuthToken(ctx, httpClient, baseURL, opts.Verbose)
	if err != nil {
		return err
	}

	if err := verifyEvaluationCatalog(ctx, httpClient, baseURL, token); err != nil {
		return err
	}
	if err := verifyEvaluationAPIGuards(ctx, httpClient, baseURL, token); err != nil {
		return err
	}

	baseline, baselineReport, err := executeVerifiedEvaluationBaseline(
		ctx, httpClient, baseURL, token,
	)
	if err != nil {
		return err
	}
	if err := verifySameRevisionComparisonGuard(ctx, httpClient, baseURL, token, baseline.ID); err != nil {
		return err
	}
	if err := verifyEvaluationCancellation(ctx, httpClient, baseURL, token); err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"schema_version": evaluationSchemaVersion,
			"track_count":    len(baselineReport.Tracks),
			"coverage":       baselineReport.Summary.Coverage.Fraction,
			"failed_gates":   baselineReport.Summary.FailedGates,
		})
	}
	return nil
}

func executeVerifiedEvaluationBaseline(
	ctx context.Context,
	client *http.Client,
	baseURL, token string,
) (dashboardEvaluationRun, dashboardEvaluationReport, error) {
	baseline, err := createEvaluationSmokeRun(ctx, client, baseURL, token, "baseline", 41, "")
	if err != nil {
		return baseline, dashboardEvaluationReport{}, err
	}
	baseline, err = waitForEvaluationRun(ctx, client, baseURL, token, baseline.ID)
	if err != nil {
		return baseline, dashboardEvaluationReport{}, err
	}
	report, err := fetchEvaluationReport(ctx, client, baseURL, token, baseline.ID)
	if err != nil {
		return baseline, report, err
	}
	if err := verifyEvaluationReport(report); err != nil {
		return baseline, report, err
	}
	if err := verifyEvaluationArtifactDownload(
		ctx, client, baseURL, token, baseline.ID, report,
	); err != nil {
		return baseline, report, err
	}
	return baseline, report, nil
}

func verifySameRevisionComparisonGuard(
	ctx context.Context,
	client *http.Client,
	baseURL, token, baselineID string,
) error {
	candidate, err := createEvaluationSmokeRun(
		ctx, client, baseURL, token, "candidate", 41, baselineID,
	)
	if err != nil {
		return err
	}
	candidate, err = waitForEvaluationRun(ctx, client, baseURL, token, candidate.ID)
	if err != nil {
		return err
	}
	url := fmt.Sprintf("%s/api/evaluation/v1/compare?baseline_run_id=%s&candidate_run_id=%s", baseURL, baselineID, candidate.ID)
	var response struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := evaluationJSON(ctx, client, http.MethodGet, url, token, nil, &response, http.StatusBadRequest); err != nil {
		return err
	}
	if !strings.Contains(response.Error.Message, "source code revision treatment to change") {
		return fmt.Errorf("same-revision comparison was rejected for the wrong reason: %s", response.Error.Message)
	}
	return nil
}

func verifyEvaluationCancellation(
	ctx context.Context,
	client *http.Client,
	baseURL, token string,
) error {
	pending, err := createEvaluationRun(ctx, client, baseURL, token, "cancel-contract", 43, "")
	if err != nil {
		return err
	}
	cancelled, err := mutateEvaluationRun(ctx, client, baseURL, token, pending.ID, "cancel")
	if err != nil {
		return err
	}
	if cancelled.Status != "cancelled" {
		return fmt.Errorf("cancelled run status = %q, want cancelled", cancelled.Status)
	}
	return nil
}

func verifyEvaluationAPIGuards(ctx context.Context, client *http.Client, baseURL, token string) error {
	invalid := map[string]interface{}{
		"name": "must-not-start", "description": "RBAC boundary",
		"suite_ids": []string{evaluationSmokeSuite}, "track_ids": evaluationTrackIDs,
		"mode": "replay", "target_id": evaluationFixtureTarget,
		"change_profile": "schema_adapter",
		"sample_limit":   4, "concurrency": 1, "seed": 1, "auto_start": true,
	}
	if err := evaluationJSON(ctx, client, http.MethodPost, baseURL+"/api/evaluation/v1/runs", token, invalid, nil, http.StatusBadRequest); err != nil {
		return fmt.Errorf("auto_start authorization guard: %w", err)
	}
	invalid["auto_start"] = false
	invalid["unexpected_field"] = "schema drift"
	if err := evaluationJSON(ctx, client, http.MethodPost, baseURL+"/api/evaluation/v1/runs", token, invalid, nil, http.StatusBadRequest); err != nil {
		return fmt.Errorf("strict evaluation request guard: %w", err)
	}
	for _, legacyPath := range []string{"/api/evaluation/tasks", "/api/evaluation/run", "/api/evaluation/datasets"} {
		if err := evaluationJSON(ctx, client, http.MethodGet, baseURL+legacyPath, token, nil, nil, http.StatusNotFound); err != nil {
			return fmt.Errorf("legacy endpoint %s is still reachable: %w", legacyPath, err)
		}
	}
	return nil
}

func verifyEvaluationCatalog(ctx context.Context, client *http.Client, baseURL, token string) error {
	var catalog struct {
		SchemaVersion       string                  `json:"schema_version"`
		GateContractVersion string                  `json:"gate_contract_version"`
		ChangeProfiles      []evaluationCatalogItem `json:"change_profiles"`
		Tracks              []evaluationCatalogItem `json:"tracks"`
		Suites              []evaluationCatalogItem `json:"suites"`
		Targets             []evaluationCatalogItem `json:"targets"`
	}
	if err := evaluationJSON(ctx, client, http.MethodGet, baseURL+"/api/evaluation/v1/catalog", token, nil, &catalog, http.StatusOK); err != nil {
		return err
	}
	if catalog.SchemaVersion != evaluationSchemaVersion {
		return fmt.Errorf("evaluation schema = %q, want %q", catalog.SchemaVersion, evaluationSchemaVersion)
	}
	if catalog.GateContractVersion != evaluationGateContract {
		return fmt.Errorf("evaluation gate contract = %q, want %q", catalog.GateContractVersion, evaluationGateContract)
	}
	if !containsEvaluationID(catalog.ChangeProfiles, "schema_adapter") {
		return fmt.Errorf("evaluation catalog is missing schema_adapter change profile")
	}
	if !containsEveryEvaluationID(catalog.Tracks, evaluationTrackIDs) {
		return fmt.Errorf("evaluation catalog does not expose all required tracks")
	}
	if !containsEvaluationID(catalog.Suites, evaluationSmokeSuite) {
		return fmt.Errorf("evaluation catalog is missing suite %q", evaluationSmokeSuite)
	}
	if !containsEvaluationID(catalog.Targets, evaluationFixtureTarget) {
		return fmt.Errorf("evaluation catalog is missing target %q", evaluationFixtureTarget)
	}
	return nil
}

func createEvaluationSmokeRun(
	ctx context.Context,
	client *http.Client,
	baseURL, token, name string,
	seed int,
	baselineID string,
) (dashboardEvaluationRun, error) {
	run, err := createEvaluationRun(ctx, client, baseURL, token, name, seed, baselineID)
	if err != nil {
		return run, err
	}
	if run.Status != "pending" {
		return run, fmt.Errorf("new evaluation run status = %q, want pending", run.Status)
	}
	return mutateEvaluationRun(ctx, client, baseURL, token, run.ID, "start")
}

func createEvaluationRun(
	ctx context.Context,
	client *http.Client,
	baseURL, token, name string,
	seed int,
	baselineID string,
) (dashboardEvaluationRun, error) {
	payload := map[string]interface{}{
		"name":           "E2E " + name,
		"description":    "Deterministic Evaluation Plane acceptance run",
		"suite_ids":      []string{evaluationSmokeSuite},
		"track_ids":      evaluationTrackIDs,
		"mode":           "replay",
		"target_id":      evaluationFixtureTarget,
		"change_profile": "schema_adapter",
		"sample_limit":   8,
		"concurrency":    2,
		"seed":           seed,
		"auto_start":     false,
	}
	if baselineID != "" {
		payload["baseline_run_id"] = baselineID
	}
	var run dashboardEvaluationRun
	err := evaluationJSON(ctx, client, http.MethodPost, baseURL+"/api/evaluation/v1/runs", token, payload, &run, http.StatusCreated)
	return run, err
}

func waitForEvaluationRun(
	ctx context.Context,
	client *http.Client,
	baseURL, token, runID string,
) (dashboardEvaluationRun, error) {
	deadline := time.Now().Add(evaluationPollTimeout)
	for time.Now().Before(deadline) {
		var run dashboardEvaluationRun
		if err := evaluationJSON(ctx, client, http.MethodGet, baseURL+"/api/evaluation/v1/runs/"+runID, token, nil, &run, http.StatusOK); err != nil {
			return run, err
		}
		switch run.Status {
		case "completed":
			if run.Progress.Percent != 100 {
				return run, fmt.Errorf("completed run progress = %d, want 100", run.Progress.Percent)
			}
			return run, nil
		case "failed", "cancelled":
			return run, fmt.Errorf("evaluation run %s ended %s: %s", runID, run.Status, run.Error)
		}
		select {
		case <-ctx.Done():
			return run, ctx.Err()
		case <-time.After(500 * time.Millisecond):
		}
	}
	return dashboardEvaluationRun{}, fmt.Errorf("evaluation run %s did not finish within %s", runID, evaluationPollTimeout)
}

func fetchEvaluationReport(ctx context.Context, client *http.Client, baseURL, token, runID string) (dashboardEvaluationReport, error) {
	var report dashboardEvaluationReport
	err := evaluationJSON(ctx, client, http.MethodGet, baseURL+"/api/evaluation/v1/runs/"+runID+"/report", token, nil, &report, http.StatusOK)
	return report, err
}

func verifyEvaluationReport(report dashboardEvaluationReport) error {
	for _, verify := range []func(dashboardEvaluationReport) error{
		verifyEvaluationReportIdentity,
		verifyEvaluationReportSummary,
		verifyEvaluationReportTracks,
		verifyEvaluationReportGates,
		verifyEvaluationReportArtifacts,
	} {
		if err := verify(report); err != nil {
			return err
		}
	}
	raw, _ := json.Marshal(report)
	if bytes.Contains(bytes.ToLower(raw), []byte("case_grading")) {
		return fmt.Errorf("public report leaked hidden grading data")
	}
	return nil
}

func verifyEvaluationReportIdentity(report dashboardEvaluationReport) error {
	if report.SchemaVersion != evaluationSchemaVersion {
		return fmt.Errorf("report schema = %q, want %q", report.SchemaVersion, evaluationSchemaVersion)
	}
	if report.Run.ChangeProfile != "schema_adapter" {
		return fmt.Errorf("report change profile = %q, want schema_adapter", report.Run.ChangeProfile)
	}
	if report.Run.EvidenceLevel != "E0" || report.Summary.Verdict != "unavailable" {
		return fmt.Errorf("fixture evidence/verdict = %s/%s, want E0/unavailable", report.Run.EvidenceLevel, report.Summary.Verdict)
	}
	return nil
}

func verifyEvaluationReportSummary(report dashboardEvaluationReport) error {
	if report.Summary.QualityScore != nil {
		return fmt.Errorf("E0 fixture quality score = %v, want unavailable", *report.Summary.QualityScore)
	}
	if report.Summary.Coverage.Fraction < 0.95 {
		return fmt.Errorf("fixture coverage = %.3f, want >= 0.95", report.Summary.Coverage.Fraction)
	}
	if report.Summary.FailedGates != 0 {
		return fmt.Errorf("fixture report has %d failed gates", report.Summary.FailedGates)
	}
	if report.Summary.UnavailableGates == 0 {
		return fmt.Errorf("fixture report must mark production online evidence unavailable")
	}
	return nil
}

func verifyEvaluationReportTracks(report dashboardEvaluationReport) error {
	seenTracks := make(map[string]bool, len(report.Tracks))
	for _, track := range report.Tracks {
		if track.Status != "completed" {
			return fmt.Errorf("track %s status = %q, want completed", track.TrackID, track.Status)
		}
		seenTracks[track.TrackID] = true
	}
	for _, trackID := range evaluationTrackIDs {
		if !seenTracks[trackID] {
			return fmt.Errorf("report is missing track %s", trackID)
		}
	}
	return nil
}

func verifyEvaluationReportGates(report dashboardEvaluationReport) error {
	if len(report.Gates) == 0 {
		return fmt.Errorf("report must include gates")
	}
	for _, gate := range report.Gates {
		if gate.ChangeProfile != report.Run.ChangeProfile || gate.ContractVersion != evaluationGateContract || len(gate.EvidenceRefs) == 0 {
			return fmt.Errorf("gate %s is missing its profile-qualified evidence contract", gate.ID)
		}
	}
	return nil
}

func verifyEvaluationReportArtifacts(report dashboardEvaluationReport) error {
	if len(report.Artifacts) == 0 {
		return fmt.Errorf("report must include artifacts")
	}
	for _, artifact := range report.Artifacts {
		if strings.TrimSpace(artifact.Digest) == "" {
			return fmt.Errorf("artifact %q is missing a content digest", artifact.Name)
		}
	}
	return nil
}

func verifyEvaluationArtifactDownload(
	ctx context.Context,
	client *http.Client,
	baseURL, token, runID string,
	report dashboardEvaluationReport,
) error {
	publicSummary := findEvaluationArtifact(report.Artifacts, "failure-summary.json")
	if publicSummary == nil || publicSummary.ID == "" {
		return fmt.Errorf("evaluation report is missing downloadable failure-summary.json identity")
	}
	data, status, err := fetchEvaluationArtifact(ctx, client, baseURL, token, runID, publicSummary.ID)
	if err != nil {
		return err
	}
	if status != http.StatusOK {
		return fmt.Errorf("failure-summary.json artifact returned HTTP %d", status)
	}
	digest := fmt.Sprintf("sha256:%x", sha256.Sum256(data))
	if digest != publicSummary.Digest {
		return fmt.Errorf("failure-summary.json digest mismatch: got %s, want %s", digest, publicSummary.Digest)
	}
	return verifyProtectedEvaluationArtifacts(ctx, client, baseURL, token, runID, report)
}

func findEvaluationArtifact(
	artifacts []evaluationArtifact,
	name string,
) *evaluationArtifact {
	for index := range artifacts {
		if artifacts[index].Name == name {
			return &artifacts[index]
		}
	}
	return nil
}

func verifyProtectedEvaluationArtifacts(
	ctx context.Context,
	client *http.Client,
	baseURL, token, runID string,
	report dashboardEvaluationReport,
) error {
	for _, protected := range []struct {
		name string
		id   string
	}{
		{name: "report.html", id: "report-html"},
		{name: "cases.jsonl", id: "cases-jsonl"},
		{name: "run-manifest.json", id: "run-manifest-json"},
	} {
		if findEvaluationArtifact(report.Artifacts, protected.name) != nil {
			return fmt.Errorf("private artifact %s leaked into the public report", protected.name)
		}
		_, status, err := fetchEvaluationArtifact(ctx, client, baseURL, token, runID, protected.id)
		if err != nil {
			return err
		}
		if status != http.StatusBadRequest && status != http.StatusNotFound {
			return fmt.Errorf("private artifact %s returned HTTP %d, want protected", protected.name, status)
		}
	}
	return nil
}

func fetchEvaluationArtifact(
	ctx context.Context,
	client *http.Client,
	baseURL, token, runID, artifactID string,
) ([]byte, int, error) {
	url := fmt.Sprintf("%s/api/evaluation/v1/runs/%s/artifacts/%s", baseURL, runID, artifactID)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, 0, fmt.Errorf("create evaluation artifact request: %w", err)
	}
	setDashboardAuth(req, token)
	resp, err := client.Do(req)
	if err != nil {
		return nil, 0, fmt.Errorf("download evaluation artifact: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	if err != nil {
		return nil, resp.StatusCode, fmt.Errorf("read evaluation artifact: %w", err)
	}
	return data, resp.StatusCode, nil
}

func mutateEvaluationRun(ctx context.Context, client *http.Client, baseURL, token, runID, action string) (dashboardEvaluationRun, error) {
	var run dashboardEvaluationRun
	err := evaluationJSON(ctx, client, http.MethodPost, fmt.Sprintf("%s/api/evaluation/v1/runs/%s/%s", baseURL, runID, action), token, map[string]interface{}{}, &run, http.StatusOK)
	return run, err
}

func evaluationJSON(
	ctx context.Context,
	client *http.Client,
	method, url, token string,
	payload interface{},
	destination interface{},
	wantStatus int,
) error {
	var body io.Reader
	if payload != nil {
		raw, err := json.Marshal(payload)
		if err != nil {
			return fmt.Errorf("marshal evaluation request: %w", err)
		}
		body = bytes.NewReader(raw)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		return fmt.Errorf("create evaluation request: %w", err)
	}
	setDashboardAuth(req, token)
	if payload != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("evaluation request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != wantStatus {
		return fmt.Errorf("%s %s: expected %d, got %d: %s", method, url, wantStatus, resp.StatusCode, truncateString(string(raw), 300))
	}
	if destination != nil && len(raw) > 0 {
		if err := json.Unmarshal(raw, destination); err != nil {
			return fmt.Errorf("decode evaluation response: %w", err)
		}
	}
	return nil
}

func containsEvaluationID(items []evaluationCatalogItem, wanted string) bool {
	for _, item := range items {
		if item.ID == wanted {
			return true
		}
	}
	return false
}

func containsEveryEvaluationID(items []evaluationCatalogItem, wanted []string) bool {
	for _, wantedID := range wanted {
		if !containsEvaluationID(items, wantedID) {
			return false
		}
	}
	return true
}
