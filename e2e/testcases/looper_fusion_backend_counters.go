package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// fusionBackendCounters reads the looper fake backend's test-only dispatch
// counters, the only way to see that a call did not happen: a regression that
// calls the judge and discards its output leaves the response unchanged.
//
// The counters are process-global in the backend, so cases using them must run
// sequentially. That is the framework default; see the parallel-mode warning in
// e2e/README.md.
type fusionBackendCounters struct {
	baseURL string
	close   func()
}

func openFusionBackendCounters(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) (*fusionBackendCounters, error) {
	backendOpts := opts
	backendOpts.ServiceConfig = pkgtestcases.ServiceConfig{
		Namespace:   "default",
		Name:        "looper-fake-backend",
		ServicePort: "8000",
	}
	session, err := fixtures.OpenServiceSession(ctx, client, backendOpts)
	if err != nil {
		return nil, fmt.Errorf("open looper fake-backend session: %w", err)
	}
	return &fusionBackendCounters{
		baseURL: "http://localhost:" + session.LocalPort(),
		close:   session.Close,
	}, nil
}

func (c *fusionBackendCounters) reset(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/test/reset", nil)
	if err != nil {
		return fmt.Errorf("create counter reset request: %w", err)
	}
	resp, err := (&http.Client{Timeout: 10 * time.Second}).Do(req)
	if err != nil {
		return fmt.Errorf("reset backend counters: %w", err)
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("reset backend counters: status %d", resp.StatusCode)
	}
	return nil
}

func (c *fusionBackendCounters) snapshot(ctx context.Context) (map[string]int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/test/calls", nil)
	if err != nil {
		return nil, fmt.Errorf("create counter snapshot request: %w", err)
	}
	resp, err := (&http.Client{Timeout: 10 * time.Second}).Do(req)
	if err != nil {
		return nil, fmt.Errorf("read backend counters: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read backend counter body: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("read backend counters: status %d: %s", resp.StatusCode, string(body))
	}
	var decoded struct {
		Calls map[string]int `json:"calls"`
	}
	if err := json.Unmarshal(body, &decoded); err != nil {
		return nil, fmt.Errorf("decode backend counters: %w: %s", err, string(body))
	}
	if decoded.Calls == nil {
		decoded.Calls = map[string]int{}
	}
	return decoded.Calls, nil
}

// waitForDispatch blocks until the backend has been asked to serve model at
// least once. Tests use it as a synchronization barrier instead of a fixed
// sleep, which would be flaky in both directions.
func (c *fusionBackendCounters) waitForDispatch(
	ctx context.Context,
	model string,
	timeout time.Duration,
) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		calls, err := c.snapshot(ctx)
		if err != nil {
			return err
		}
		if calls[model] > 0 {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(50 * time.Millisecond):
		}
	}
	return fmt.Errorf("model %q was never dispatched within %s", model, timeout)
}

// requireCounts asserts exact dispatch counts for the named models only, so a
// case pins its own contract without enumerating every panel member.
func (c *fusionBackendCounters) requireCounts(
	ctx context.Context,
	testName string,
	want map[string]int,
) error {
	calls, err := c.snapshot(ctx)
	if err != nil {
		return err
	}
	models := make([]string, 0, len(want))
	for model := range want {
		models = append(models, model)
	}
	sort.Strings(models)

	mismatches := make([]string, 0, len(models))
	for _, model := range models {
		if got := calls[model]; got != want[model] {
			mismatches = append(mismatches, fmt.Sprintf("%s: got %d, want %d", model, got, want[model]))
		}
	}
	if len(mismatches) > 0 {
		return fmt.Errorf("%s: backend dispatch counts wrong (%s); full snapshot: %v",
			testName, strings.Join(mismatches, "; "), calls)
	}
	return nil
}

// openFusionCounters opens the counter session and clears it, so a case asserts
// only the dispatches its own request caused.
func openFusionCounters(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) (*fusionBackendCounters, error) {
	counters, err := openFusionBackendCounters(ctx, client, opts)
	if err != nil {
		return nil, err
	}
	if err := counters.reset(ctx); err != nil {
		counters.close()
		return nil, err
	}
	return counters, nil
}
