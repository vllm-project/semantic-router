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
	ID         string  `json:"id"`
	Enabled    bool    `json:"enabled"`
	Configured bool    `json:"configured"`
	Reachable  bool    `json:"reachable"`
	Authorized bool    `json:"authorized"`
	Degraded   bool    `json:"degraded"`
	State      string  `json:"state"`
	Reason     string  `json:"reason"`
	CheckedAt  *string `json:"checkedAt"`
}

type dashboardCapabilities struct {
	Capabilities []dashboardCapability `json:"capabilities"`
}

var (
	dashboardCapabilityIDs    = []string{"setup", "config", "evaluation", "mlPipeline", "mcp"}
	dashboardCapabilityStates = []string{"disabled", "unconfigured", "unauthorized", "unreachable", "degraded", "available"}
)

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

	// An /api/status request publishes the Router readiness it observed, so
	// the config capability must then report a completed check, whether or
	// not this deployment exposes the Router management API to the Dashboard.
	status, body, err := getDashboardOpenAPI(ctx, httpClient, baseURL+"/api/status", token)
	if err != nil {
		return err
	}
	if status != http.StatusOK {
		return fmt.Errorf("GET /api/status = %d: %s", status, truncateString(string(body), 200))
	}
	status, body, err = getDashboardOpenAPI(ctx, httpClient, url, token)
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
	config, err := checkDashboardCapabilities(response)
	if err != nil {
		return err
	}
	if opts.Verbose {
		fmt.Printf("[Dashboard] capabilities OK: config state=%s reason=%s\n", config.State, config.Reason)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"config_state": config.State, "config_reason": config.Reason})
	}
	return nil
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
	config := byID["config"]
	if config.CheckedAt == nil || config.Reason == "probe_pending" {
		return dashboardCapability{}, fmt.Errorf("config capability has no completed Router check after /api/status: %+v", config)
	}
	return config, nil
}
