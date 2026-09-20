package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/google/uuid"
	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const srBenchAPI = "/api/sr-bench/v1"

type dashboardBenchRun struct {
	ID       string `json:"id"`
	Status   string `json:"status"`
	Manifest struct {
		PlanSHA string `json:"plan_sha256"`
	} `json:"manifest"`
	Progress struct {
		Total     int `json:"total"`
		Completed int `json:"completed"`
		Failed    int `json:"failed"`
	} `json:"progress"`
}

func init() {
	pkgtestcases.Register("dashboard-sr-bench", pkgtestcases.TestCase{
		Description: "sr-bench authenticates, freezes, executes and accounts synthetic HTTP cases through its independent worker",
		Tags:        []string{"dashboard", "functional", "sr-bench"},
		Fn:          testDashboardSrBench,
	})
}

func testDashboardSrBench(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	port, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()
	base := "http://localhost:" + port
	httpClient := &http.Client{Timeout: 30 * time.Second}
	if err = srBenchJSON(ctx, httpClient, http.MethodGet, base+srBenchAPI+"/catalog", "", nil, nil, http.StatusUnauthorized); err != nil {
		return fmt.Errorf("anonymous benchmark access: %w", err)
	}
	token, err := dashboardAuthToken(ctx, httpClient, base, opts.Verbose)
	if err != nil {
		return err
	}
	var catalog struct {
		Version    string           `json:"version"`
		Benchmarks []map[string]any `json:"benchmarks"`
	}
	if err = srBenchJSON(ctx, httpClient, http.MethodGet, base+srBenchAPI+"/catalog", token, nil, &catalog, http.StatusOK); err != nil {
		return err
	}
	if catalog.Version != "sr-bench-1.0" || len(catalog.Benchmarks) != 9 {
		return fmt.Errorf("unexpected sr-bench catalog: version=%q benchmarks=%d", catalog.Version, len(catalog.Benchmarks))
	}
	run, evidence, err := executeVerifiedSrBench(ctx, httpClient, base, token)
	if err != nil {
		return err
	}
	if err = verifySrBenchCancellation(ctx, httpClient, base, token); err != nil {
		return err
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]any{
			"run_id": run.ID, "plan_sha256": run.Manifest.PlanSHA,
			"evidence_sha256": evidence, "completed_cases": run.Progress.Completed,
			"synthetic_transport_only": true, "final_channel_and_four_bucket_cost_verified": true,
			"idempotency_verified": true, "cancellation_verified": true,
		})
	}
	return nil
}

func srBenchManifest(target string, count int) map[string]any {
	cases := make([]map[string]any, count)
	for i := range cases {
		cases[i] = map[string]any{
			"id": fmt.Sprintf("synthetic-%d", i), "benchmark": "mmlu-pro",
			"messages": []map[string]string{{"role": "user", "content": "Synthetic HTTP fixture: return A."}},
			"answer":   "A",
		}
	}
	return map[string]any{
		"version": "sr-bench-1.0", "name": "Kubernetes synthetic transport acceptance", "profile": "smoke",
		"targets": []map[string]string{{"id": target}}, "cases": cases,
		"limits": map[string]any{
			"concurrency": 1, "max_output_tokens": 16, "total_timeout_s": 60,
			"idle_timeout_s": 10, "max_run_seconds": 180, "max_cost_usd": 1,
		},
		"limitations": []string{"Synthetic CI fixture; not a model capability or upstream benchmark score."},
	}
}

func executeVerifiedSrBench(ctx context.Context, client *http.Client, base, token string) (dashboardBenchRun, string, error) {
	manifest := srBenchManifest("fixture", 2)
	var planned struct {
		PlanSHA string `json:"plan_sha256"`
		Total   int    `json:"total"`
		Status  string `json:"status"`
	}
	if err := srBenchJSON(ctx, client, http.MethodPost, base+srBenchAPI+"/plans", token, map[string]any{"manifest": manifest}, &planned, http.StatusOK); err != nil {
		return dashboardBenchRun{}, "", err
	}
	if planned.Status != "validated" || planned.Total != 2 || len(planned.PlanSHA) != 64 {
		return dashboardBenchRun{}, "", fmt.Errorf("plan did not freeze exactly two cases: %+v", planned)
	}
	body := map[string]any{"manifest": manifest, "idempotency_key": uuid.NewString()}
	var run dashboardBenchRun
	if err := srBenchJSON(ctx, client, http.MethodPost, base+srBenchAPI+"/runs", token, body, &run, http.StatusCreated); err != nil {
		return run, "", err
	}
	run, err := waitSrBenchTerminal(ctx, client, base, token, run.ID)
	if err != nil {
		return run, "", err
	}
	if run.Status != "completed" || run.Progress.Total != 2 || run.Progress.Completed != 2 || run.Progress.Failed != 0 || run.Manifest.PlanSHA != planned.PlanSHA {
		return run, "", fmt.Errorf("run did not complete its frozen two-case plan: %+v", run)
	}
	var duplicate dashboardBenchRun
	if err = srBenchJSON(ctx, client, http.MethodPost, base+srBenchAPI+"/runs", token, body, &duplicate, http.StatusCreated); err != nil {
		return run, "", err
	}
	if duplicate.ID != run.ID || duplicate.Status != "completed" {
		return run, "", fmt.Errorf("idempotent submission changed completed run identity/status")
	}
	evidence, err := srBenchEvidenceDigest(ctx, client, base, token, run)
	return run, evidence, err
}

func waitSrBenchTerminal(ctx context.Context, client *http.Client, base, token, id string) (dashboardBenchRun, error) {
	deadline := time.NewTimer(90 * time.Second)
	defer deadline.Stop()
	ticker := time.NewTicker(250 * time.Millisecond)
	defer ticker.Stop()
	for {
		var run dashboardBenchRun
		if err := srBenchJSON(ctx, client, http.MethodGet, base+srBenchAPI+"/runs/"+id, token, nil, &run, http.StatusOK); err != nil {
			return run, err
		}
		switch run.Status {
		case "completed", "cancelled", "failed", "interrupted":
			return run, nil
		}
		select {
		case <-ctx.Done():
			return run, ctx.Err()
		case <-deadline.C:
			return run, fmt.Errorf("sr-bench run %s did not terminate", id)
		case <-ticker.C:
		}
	}
}

func verifySrBenchCancellation(ctx context.Context, client *http.Client, base, token string) error {
	var run dashboardBenchRun
	body := map[string]any{"manifest": srBenchManifest("fixture-slow", 3), "idempotency_key": uuid.NewString()}
	if err := srBenchJSON(ctx, client, http.MethodPost, base+srBenchAPI+"/runs", token, body, &run, http.StatusCreated); err != nil {
		return err
	}
	if err := srBenchJSON(ctx, client, http.MethodPost, base+srBenchAPI+"/runs/"+run.ID+"/cancel", token, map[string]any{}, nil, http.StatusOK); err != nil {
		return err
	}
	terminal, err := waitSrBenchTerminal(ctx, client, base, token, run.ID)
	if err != nil {
		return err
	}
	if terminal.Status != "cancelled" || terminal.Progress.Total != 3 || terminal.Progress.Completed != 0 {
		return fmt.Errorf("cancellation lost the planned denominator or completed a slow fixture: %+v", terminal)
	}
	var calls struct {
		Total int `json:"total"`
	}
	if err := srBenchJSON(ctx, client, http.MethodGet, base+srBenchAPI+"/runs/"+run.ID+"/calls", token, nil, &calls, http.StatusOK); err != nil {
		return err
	}
	if calls.Total > 1 {
		return fmt.Errorf("cancelled serial run dispatched queued cases: %d calls", calls.Total)
	}
	return nil
}

func srBenchJSON(ctx context.Context, client *http.Client, method, url, token string, body, result any, status int) error {
	var data []byte
	if body != nil {
		var err error
		data, err = json.Marshal(body)
		if err != nil {
			return err
		}
	}
	req, err := http.NewRequestWithContext(ctx, method, url, bytes.NewReader(data))
	if err != nil {
		return err
	}
	if token != "" {
		setDashboardAuth(req, token)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	data, err = io.ReadAll(io.LimitReader(resp.Body, 2*1024*1024+1))
	if err != nil {
		return err
	}
	if len(data) > 2*1024*1024 {
		return fmt.Errorf("sr-bench response exceeds 2 MiB")
	}
	if resp.StatusCode != status {
		return fmt.Errorf("%s %s: expected %d, got %d: %s", method, req.URL.Path, status, resp.StatusCode, truncateString(string(data), 200))
	}
	if result != nil {
		return json.Unmarshal(data, result)
	}
	return nil
}
