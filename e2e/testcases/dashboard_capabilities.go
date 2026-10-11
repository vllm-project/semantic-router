package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("dashboard-capabilities", pkgtestcases.TestCase{
		Description: "Verify the Dashboard reports per-caller capability states with cached dependency probes",
		Tags:        []string{"dashboard", "api"},
		Fn:          testDashboardCapabilities,
	})
}

type dashboardCapability struct {
	ID         string `json:"id"`
	Enabled    bool   `json:"enabled"`
	Configured bool   `json:"configured"`
	Reachable  bool   `json:"reachable"`
	Authorized bool   `json:"authorized"`
	Degraded   bool   `json:"degraded"`
	State      string `json:"state"`
	Reason     string `json:"reason"`
}

type dashboardCapabilities struct {
	Capabilities []dashboardCapability `json:"capabilities"`
}

var (
	dashboardCapabilityIDs    = []string{"setup", "config", "evaluation", "mlPipeline", "mcp"}
	dashboardCapabilityStates = []string{"disabled", "unconfigured", "unauthorized", "unreachable", "degraded", "available"}
)

// The prober refreshes every 30 s, so a Router that became ready after the
// Dashboard started is visible within one interval.
const dashboardCapabilityReachableTimeout = 75 * time.Second

func testDashboardCapabilities(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	httpClient := &http.Client{Timeout: 15 * time.Second}
	baseURL := fmt.Sprintf("http://localhost:%s", localPort)
	url := baseURL + "/api/v1/capabilities"

	status, _, err := getDashboardOpenAPI(ctx, httpClient, url, "")
	if err != nil {
		return err
	}
	if status != http.StatusUnauthorized {
		return fmt.Errorf("anonymous GET /api/v1/capabilities = %d, want %d", status, http.StatusUnauthorized)
	}

	token, err := dashboardAuthToken(ctx, httpClient, baseURL, opts.Verbose)
	if err != nil {
		return err
	}

	deadline := time.Now().Add(dashboardCapabilityReachableTimeout)
	for {
		status, body, err := getDashboardOpenAPI(ctx, httpClient, url, token)
		if err != nil {
			return err
		}
		if status != http.StatusOK {
			return fmt.Errorf("GET /api/v1/capabilities = %d: %s", status, truncateString(string(body), 200))
		}
		var response dashboardCapabilities
		if decodeErr := json.Unmarshal(body, &response); decodeErr != nil {
			return fmt.Errorf("decode capabilities: %w", decodeErr)
		}
		config, checkErr := checkDashboardCapabilities(response)
		if checkErr != nil {
			return checkErr
		}
		if config.Reachable {
			if opts.Verbose {
				fmt.Printf("[Dashboard] capabilities OK: config state=%s\n", config.State)
			}
			if opts.SetDetails != nil {
				opts.SetDetails(map[string]interface{}{"config_state": config.State})
			}
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("config capability stayed unreachable (reason %q) for %s", config.Reason, dashboardCapabilityReachableTimeout)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(5 * time.Second):
		}
	}
}

// checkDashboardCapabilities validates the response shape for an admin caller
// and returns the config capability.
func checkDashboardCapabilities(response dashboardCapabilities) (dashboardCapability, error) {
	byID := map[string]dashboardCapability{}
	for _, capability := range response.Capabilities {
		if !slices.Contains(dashboardCapabilityStates, capability.State) {
			return dashboardCapability{}, fmt.Errorf("capability %s has unknown state %q", capability.ID, capability.State)
		}
		if (capability.State == "available") != (capability.Reason == "") {
			return dashboardCapability{}, fmt.Errorf("capability %s state %q disagrees with reason %q", capability.ID, capability.State, capability.Reason)
		}
		byID[capability.ID] = capability
	}
	for _, id := range dashboardCapabilityIDs {
		capability, ok := byID[id]
		if !ok {
			return dashboardCapability{}, fmt.Errorf("capabilities omit %q", id)
		}
		if !capability.Authorized {
			return dashboardCapability{}, fmt.Errorf("admin is not authorized for %s", id)
		}
	}
	return byID["config"], nil
}
