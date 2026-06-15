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

	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

const (
	dashboardRestartNamespace  = "vllm-semantic-router-system"
	dashboardRestartDeployment = "semantic-router-dashboard"
	dashboardRestartPodLabel   = "app=semantic-router-dashboard"

	dashboardRestartRecoveryTimeout  = 5 * time.Minute
	dashboardRestartRecoveryInterval = 5 * time.Second

	dashboardRestartMCPServerID   = "e2e00001-0000-4000-8000-000000000001"
	dashboardRestartMCPServerName = "E2E Restart Recovery MCP"
)

func init() {
	pkgtestcases.Register("dashboard-restart-recovery", pkgtestcases.TestCase{
		Description: "Dashboard workflow SQLite state survives a dashboard pod restart",
		Tags:        []string{"dashboard", "functional", "restart"},
		Fn:          testDashboardRestartRecovery,
	})
}

func testDashboardRestartRecovery(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Dashboard: restart recovery (workflow SQLite persistence)")
	}

	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}

	baseURL := fmt.Sprintf("http://localhost:%s", localPort)
	httpClient := &http.Client{Timeout: 30 * time.Second}

	if err := seedDashboardWorkflowState(ctx, httpClient, baseURL, opts.Verbose); err != nil {
		stop()
		return err
	}
	stop()

	if err := deleteDashboardPod(ctx, client, opts); err != nil {
		return err
	}

	if err := waitForDashboardReady(ctx, client, opts); err != nil {
		return err
	}

	localPort, stop, err = setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()

	return verifyDashboardWorkflowStateAfterRestart(
		ctx,
		&http.Client{Timeout: 30 * time.Second},
		fmt.Sprintf("http://localhost:%s", localPort),
		opts,
	)
}

func seedDashboardWorkflowState(ctx context.Context, client *http.Client, baseURL string, verbose bool) error {
	token, err := dashboardAuthToken(ctx, client, baseURL, verbose)
	if err != nil {
		return fmt.Errorf("pre-restart login: %w", err)
	}

	if err := createRestartTestMCPServer(ctx, client, baseURL, token, verbose); err != nil {
		return err
	}

	return assertMCPServerPresent(ctx, client, baseURL, token, verbose, "before restart")
}

func createRestartTestMCPServer(ctx context.Context, client *http.Client, baseURL, token string, verbose bool) error {
	payload := map[string]interface{}{
		"id":          dashboardRestartMCPServerID,
		"name":        dashboardRestartMCPServerName,
		"description": "Created by dashboard-restart-recovery E2E",
		"transport":   "stdio",
		"connection": map[string]string{
			"command": "echo",
		},
		"enabled": false,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal MCP server payload: %w", err)
	}

	url := strings.TrimRight(baseURL, "/") + "/api/mcp/servers"
	if verbose {
		fmt.Printf("[Dashboard] POST %s (seed MCP server)\n", url)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("create MCP server request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	setDashboardAuth(req, token)

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("create MCP server request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusCreated {
		return fmt.Errorf("create MCP server: expected 201, got %d: %s", resp.StatusCode, truncateString(string(respBody), 200))
	}

	if verbose {
		fmt.Printf("[Dashboard] MCP server %s seeded\n", dashboardRestartMCPServerID)
	}
	return nil
}

func verifyDashboardWorkflowStateAfterRestart(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	opts pkgtestcases.TestCaseOptions,
) error {
	const verifyTimeout = 90 * time.Second
	deadline := time.Now().Add(verifyTimeout)
	var lastErr error

	for time.Now().Before(deadline) {
		token, err := dashboardAuthToken(ctx, client, baseURL, opts.Verbose)
		if err != nil {
			lastErr = err
			time.Sleep(3 * time.Second)
			continue
		}

		if err := assertMCPServerPresent(ctx, client, baseURL, token, opts.Verbose, "after restart"); err != nil {
			lastErr = err
			time.Sleep(3 * time.Second)
			continue
		}

		if err := assertWorkflowHealthOK(ctx, client, baseURL, token, opts.Verbose); err != nil {
			lastErr = err
			time.Sleep(3 * time.Second)
			continue
		}

		if opts.Verbose {
			fmt.Printf("[Test] Dashboard workflow state survived restart (mcp_server_id=%s)\n", dashboardRestartMCPServerID)
		}
		if opts.SetDetails != nil {
			opts.SetDetails(map[string]interface{}{
				"mcp_server_id":   dashboardRestartMCPServerID,
				"mcp_server_name": dashboardRestartMCPServerName,
				"survived":        true,
			})
		}
		return nil
	}

	return fmt.Errorf("dashboard workflow state not recoverable after %s: %w", verifyTimeout, lastErr)
}

func assertMCPServerPresent(ctx context.Context, client *http.Client, baseURL, token string, verbose bool, phase string) error {
	url := strings.TrimRight(baseURL, "/") + "/api/mcp/servers"
	if verbose {
		fmt.Printf("[Dashboard] GET %s (%s)\n", url, phase)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("create MCP list request: %w", err)
	}
	setDashboardAuth(req, token)

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("list MCP servers failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("list MCP servers: expected 200, got %d: %s", resp.StatusCode, truncateString(string(body), 200))
	}

	var result struct {
		Servers []struct {
			Config struct {
				ID   string `json:"id"`
				Name string `json:"name"`
			} `json:"config"`
		} `json:"servers"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return fmt.Errorf("MCP servers response is not valid JSON: %w", err)
	}

	for _, server := range result.Servers {
		if server.Config.ID == dashboardRestartMCPServerID {
			if server.Config.Name != dashboardRestartMCPServerName {
				return fmt.Errorf("MCP server name mismatch: got %q, expected %q", server.Config.Name, dashboardRestartMCPServerName)
			}
			return nil
		}
	}

	return fmt.Errorf("MCP server %s not found in list (%s)", dashboardRestartMCPServerID, phase)
}

func assertWorkflowHealthOK(ctx context.Context, client *http.Client, baseURL, token string, verbose bool) error {
	url := strings.TrimRight(baseURL, "/") + "/api/workflows/health"
	if verbose {
		fmt.Printf("[Dashboard] GET %s (workflow health)\n", url)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("create workflow health request: %w", err)
	}
	setDashboardAuth(req, token)

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("workflow health request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("workflow health: expected 200, got %d: %s", resp.StatusCode, truncateString(string(body), 200))
	}

	var health struct {
		Store string `json:"store"`
	}
	if err := json.Unmarshal(body, &health); err != nil {
		return fmt.Errorf("workflow health response is not valid JSON: %w", err)
	}
	if strings.TrimSpace(health.Store) != "ok" {
		return fmt.Errorf("workflow health store: got %q, expected ok", health.Store)
	}
	return nil
}

func deleteDashboardPod(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	pods, err := client.CoreV1().Pods(dashboardRestartNamespace).List(ctx, metav1.ListOptions{
		LabelSelector: dashboardRestartPodLabel,
	})
	if err != nil {
		return fmt.Errorf("failed to list dashboard pods: %w", err)
	}
	if len(pods.Items) == 0 {
		return fmt.Errorf("no dashboard pods found in %s", dashboardRestartNamespace)
	}

	podName := pods.Items[0].Name
	if opts.Verbose {
		fmt.Printf("[Test] Deleting dashboard pod %s to simulate restart\n", podName)
	}

	if err := client.CoreV1().Pods(dashboardRestartNamespace).Delete(ctx, podName, metav1.DeleteOptions{}); err != nil {
		return fmt.Errorf("failed to delete dashboard pod %s: %w", podName, err)
	}

	return waitForOldDashboardPodTerminated(ctx, client, podName, opts)
}

func waitForOldDashboardPodTerminated(ctx context.Context, client *kubernetes.Clientset, podName string, opts pkgtestcases.TestCaseOptions) error {
	deadline := time.Now().Add(2 * time.Minute)
	for time.Now().Before(deadline) {
		_, err := client.CoreV1().Pods(dashboardRestartNamespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			if opts.Verbose {
				fmt.Printf("[Test] Old dashboard pod %s terminated\n", podName)
			}
			return nil
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("old dashboard pod %s still exists after 2m", podName)
}

func waitForDashboardReady(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Printf("[Test] Waiting for dashboard deployment to recover (timeout=%s)\n", dashboardRestartRecoveryTimeout)
	}

	return helpers.WaitForDeploymentReady(
		ctx, client,
		dashboardRestartNamespace, dashboardRestartDeployment,
		dashboardRestartRecoveryTimeout, dashboardRestartRecoveryInterval,
		opts.Verbose,
	)
}
