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
	dashboardManagementMediaType     = "application/vnd.vllm-semantic-router.management.v1+json"
	dashboardManagementNamespace     = "VLLM-SR-Namespace"
	dashboardManagementIdempotency   = "Idempotency-Key"
	dashboardRestartAgentMessage     = "Keep this Router Agent context across the Dashboard restart."
)

type dashboardRestartAgentState struct {
	namespaceID string
	profileID   string
	profileName string
	sessionID   string
	turnID      string
	targetKind  string
	targetID    string
}

func init() {
	pkgtestcases.Register("dashboard-restart-recovery", pkgtestcases.TestCase{
		Description: "Router Agent session and transcript survive a dashboard pod restart",
		Tags:        []string{"dashboard", "functional", "restart"},
		Fn:          testDashboardRestartRecovery,
	})
}

func testDashboardRestartRecovery(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Dashboard: Router Agent continuity across a Dashboard restart")
	}

	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}

	baseURL := fmt.Sprintf("http://localhost:%s", localPort)
	httpClient := &http.Client{Timeout: 30 * time.Second}

	state, err := seedDashboardAgentState(ctx, httpClient, baseURL, opts.Verbose)
	if err != nil {
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

	return verifyDashboardAgentStateAfterRestart(
		ctx,
		&http.Client{Timeout: 30 * time.Second},
		fmt.Sprintf("http://localhost:%s", localPort),
		state,
		opts,
	)
}

func seedDashboardAgentState(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	verbose bool,
) (dashboardRestartAgentState, error) {
	token, err := dashboardAuthToken(ctx, client, baseURL, verbose)
	if err != nil {
		return dashboardRestartAgentState{}, fmt.Errorf("pre-restart login: %w", err)
	}
	namespaceID, userID, err := dashboardAgentIdentity(ctx, client, baseURL, token, verbose)
	if err != nil {
		return dashboardRestartAgentState{}, err
	}
	targetKind, targetID, err := ensureDashboardAgentFixture(
		ctx, client, baseURL, token, namespaceID, userID, verbose,
	)
	if err != nil {
		return dashboardRestartAgentState{}, err
	}
	seed := fmt.Sprintf("dashboard-restart-%d", time.Now().UTC().UnixNano())
	profileName := "Dashboard restart " + strings.TrimPrefix(seed, "dashboard-restart-")
	profileReceipt, err := dashboardManagementMutation(
		ctx, client, baseURL, token, namespaceID,
		http.MethodPost, "/agent-profiles", seed+"-profile",
		map[string]interface{}{
			"name":           profileName,
			"description":    "E2E Agent continuity contract",
			"supportedModes": []string{"chat"},
			"toolPolicy": map[string]interface{}{
				"allow": []string{"router.catalog.describe"},
			},
			"approvalPolicy":     "required",
			"maximumTurnSeconds": 300,
			"maximumToolSteps":   4,
			"contextTokenBudget": 8192,
		},
		verbose,
	)
	if err != nil {
		return dashboardRestartAgentState{}, fmt.Errorf("create pre-restart Agent Profile: %w", err)
	}
	if profileReceipt.Resource == nil || profileReceipt.Resource.ID == "" {
		return dashboardRestartAgentState{}, fmt.Errorf("create Agent Profile returned no resource")
	}
	sessionReceipt, err := dashboardManagementMutation(
		ctx, client, baseURL, token, namespaceID,
		http.MethodPost, "/agent-sessions", seed+"-session",
		map[string]interface{}{
			"profileId": profileReceipt.Resource.ID,
			"mode":      "chat",
			"target": map[string]string{
				"kind": targetKind,
				"id":   targetID,
			},
			"title": "Dashboard restart recovery",
		},
		verbose,
	)
	if err != nil {
		return dashboardRestartAgentState{}, fmt.Errorf("create pre-restart Agent Session: %w", err)
	}
	if sessionReceipt.Resource == nil || sessionReceipt.Resource.ID == "" {
		return dashboardRestartAgentState{}, fmt.Errorf("create Agent Session returned no resource")
	}
	turnReceipt, err := dashboardManagementMutation(
		ctx, client, baseURL, token, namespaceID,
		http.MethodPost, "/agent-sessions/"+sessionReceipt.Resource.ID+"/turns", seed+"-turn",
		map[string]interface{}{
			"input": map[string]interface{}{
				"content": []map[string]string{{"type": "text", "text": dashboardRestartAgentMessage}},
			},
		},
		verbose,
	)
	if err != nil {
		return dashboardRestartAgentState{}, fmt.Errorf("create pre-restart Agent Turn: %w", err)
	}
	if turnReceipt.Resource == nil || turnReceipt.Resource.ID == "" {
		return dashboardRestartAgentState{}, fmt.Errorf("create Agent Turn returned no resource")
	}
	state := dashboardRestartAgentState{
		namespaceID: namespaceID,
		profileID:   profileReceipt.Resource.ID,
		profileName: profileName,
		sessionID:   sessionReceipt.Resource.ID,
		turnID:      turnReceipt.Resource.ID,
		targetKind:  targetKind,
		targetID:    targetID,
	}
	if err := waitForDashboardAgentState(
		ctx, client, baseURL, token, state, verbose, "before restart", 90*time.Second,
	); err != nil {
		return dashboardRestartAgentState{}, err
	}
	return state, nil
}

func waitForDashboardAgentState(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	token string,
	state dashboardRestartAgentState,
	verbose bool,
	phase string,
	timeout time.Duration,
) error {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for time.Now().Before(deadline) {
		if err := assertDashboardAgentState(ctx, client, baseURL, token, state, verbose, phase); err == nil {
			return nil
		} else {
			lastErr = err
		}
		timer := time.NewTimer(time.Second)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
	return fmt.Errorf("Router Agent turn did not complete %s within %s: %w", phase, timeout, lastErr)
}

func verifyDashboardAgentStateAfterRestart(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	state dashboardRestartAgentState,
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

		if err := assertDashboardAgentState(
			ctx, client, baseURL, token, state, opts.Verbose, "after restart",
		); err != nil {
			lastErr = err
			time.Sleep(3 * time.Second)
			continue
		}
		if opts.Verbose {
			fmt.Printf("[Test] Router Agent state survived Dashboard restart (session_id=%s)\n", state.sessionID)
		}
		if opts.SetDetails != nil {
			opts.SetDetails(map[string]interface{}{
				"agent_profile_id": state.profileID,
				"agent_session_id": state.sessionID,
				"agent_turn_id":    state.turnID,
				"survived":         true,
			})
		}
		return nil
	}

	return fmt.Errorf("Router Agent state not recoverable through Dashboard after %s: %w", verifyTimeout, lastErr)
}

func assertDashboardAgentState(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	token string,
	state dashboardRestartAgentState,
	verbose bool,
	phase string,
) error {
	var profile struct {
		Data struct {
			ID   string `json:"id"`
			Name string `json:"name"`
		} `json:"data"`
	}
	if err := dashboardManagementJSON(
		ctx, client, baseURL, token, state.namespaceID, http.MethodGet,
		"/agent-profiles/"+state.profileID, "", nil, http.StatusOK, &profile, verbose,
	); err != nil {
		return fmt.Errorf("read Agent Profile %s: %w", phase, err)
	}
	if profile.Data.ID != state.profileID || profile.Data.Name != state.profileName {
		return fmt.Errorf("Agent Profile changed %s", phase)
	}

	var session struct {
		Data struct {
			ID     string `json:"id"`
			Mode   string `json:"mode"`
			Target struct {
				Kind string `json:"kind"`
				ID   string `json:"id"`
			} `json:"target"`
		} `json:"data"`
	}
	if err := dashboardManagementJSON(
		ctx, client, baseURL, token, state.namespaceID, http.MethodGet,
		"/agent-sessions/"+state.sessionID, "", nil, http.StatusOK, &session, verbose,
	); err != nil {
		return fmt.Errorf("read Agent Session %s: %w", phase, err)
	}
	if session.Data.ID != state.sessionID || session.Data.Mode != "chat" ||
		session.Data.Target.Kind != state.targetKind || session.Data.Target.ID != state.targetID {
		return fmt.Errorf("Agent Session identity or target changed %s", phase)
	}

	var events struct {
		Data []struct {
			TurnID  string          `json:"turnId"`
			Type    string          `json:"type"`
			Payload json.RawMessage `json:"payload"`
		} `json:"data"`
	}
	if err := dashboardManagementJSON(
		ctx, client, baseURL, token, state.namespaceID, http.MethodGet,
		"/agent-sessions/"+state.sessionID+"/events?pageSize=100", "", nil,
		http.StatusOK, &events, verbose,
	); err != nil {
		return fmt.Errorf("read Agent transcript %s: %w", phase, err)
	}
	foundUserInput := false
	foundAssistantOutput := false
	foundCompletedTerminal := false
	for _, event := range events.Data {
		if event.TurnID != state.turnID {
			continue
		}
		switch event.Type {
		case "user_input":
			var payload struct {
				Content []struct {
					Type string `json:"type"`
					Text string `json:"text"`
				} `json:"content"`
			}
			if err := json.Unmarshal(event.Payload, &payload); err != nil {
				return fmt.Errorf("Agent user event %s is invalid: %w", phase, err)
			}
			for _, block := range payload.Content {
				if block.Type == "text" && block.Text == dashboardRestartAgentMessage {
					foundUserInput = true
				}
			}
		case "assistant_delta":
			var payload struct {
				Delta struct {
					Kind string `json:"kind"`
					Text string `json:"text"`
				} `json:"delta"`
			}
			if err := json.Unmarshal(event.Payload, &payload); err != nil {
				return fmt.Errorf("Agent assistant event %s is invalid: %w", phase, err)
			}
			if payload.Delta.Kind == "text" && strings.TrimSpace(payload.Delta.Text) != "" {
				foundAssistantOutput = true
			}
		case "terminal":
			var payload struct {
				Status string `json:"status"`
			}
			if err := json.Unmarshal(event.Payload, &payload); err != nil {
				return fmt.Errorf("Agent terminal event %s is invalid: %w", phase, err)
			}
			foundCompletedTerminal = payload.Status == "completed"
		}
	}
	if !foundUserInput {
		return fmt.Errorf("Agent user input event is missing %s", phase)
	}
	if !foundAssistantOutput {
		return fmt.Errorf("Agent assistant output is missing %s", phase)
	}
	if !foundCompletedTerminal {
		return fmt.Errorf("Agent completed terminal event is missing %s", phase)
	}
	return nil
}

func dashboardAgentIdentity(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	token string,
	verbose bool,
) (string, string, error) {
	var me struct {
		Namespaces []struct {
			Namespace struct {
				ID string `json:"namespaceId"`
			} `json:"namespace"`
			Permissions []string `json:"permissions"`
			User        *struct {
				UserID string `json:"userId"`
			} `json:"user,omitempty"`
		} `json:"namespaces"`
	}
	if err := dashboardManagementJSON(
		ctx, client, baseURL, token, "", http.MethodGet, "/me", "", nil,
		http.StatusOK, &me, verbose,
	); err != nil {
		return "", "", fmt.Errorf("read Management identity: %w", err)
	}
	for _, scope := range me.Namespaces {
		if scope.Namespace.ID != "" && scope.User != nil && scope.User.UserID != "" &&
			hasDashboardAgentPermissions(scope.Permissions) {
			return scope.Namespace.ID, scope.User.UserID, nil
		}
	}
	return "", "", fmt.Errorf("Dashboard test principal has no linked User and namespace with Agent and routing authority")
}

func hasDashboardAgentPermissions(permissions []string) bool {
	required := map[string]bool{
		"agent.read": false, "agent.use": false, "agent.manage": false, "routing.read": false,
	}
	for _, permission := range permissions {
		if _, tracked := required[permission]; tracked {
			required[permission] = true
		}
	}
	for _, present := range required {
		if !present {
			return false
		}
	}
	return true
}

type dashboardMutationReceipt struct {
	Resource *struct {
		Kind     string `json:"kind"`
		ID       string `json:"id"`
		Revision int64  `json:"revision"`
	} `json:"resource,omitempty"`
}

func dashboardManagementMutation(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	token string,
	namespaceID string,
	method string,
	path string,
	idempotencyKey string,
	payload interface{},
	verbose bool,
) (dashboardMutationReceipt, error) {
	var receipt dashboardMutationReceipt
	err := dashboardManagementJSON(
		ctx, client, baseURL, token, namespaceID, method, path,
		idempotencyKey, payload, http.StatusCreated, &receipt, verbose,
	)
	return receipt, err
}

func dashboardManagementJSON(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	token string,
	namespaceID string,
	method string,
	path string,
	idempotencyKey string,
	payload interface{},
	expectedStatus int,
	result interface{},
	verbose bool,
) error {
	_, err := dashboardManagementJSONWithHeaders(
		ctx, client, baseURL, token, namespaceID, method, path, idempotencyKey,
		payload, expectedStatus, result, nil, verbose,
	)
	return err
}

func dashboardManagementJSONWithHeaders(
	ctx context.Context,
	client *http.Client,
	baseURL string,
	token string,
	namespaceID string,
	method string,
	path string,
	idempotencyKey string,
	payload interface{},
	expectedStatus int,
	result interface{},
	requestHeaders http.Header,
	verbose bool,
) (http.Header, error) {
	var requestBody io.Reader
	if payload != nil {
		body, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("marshal Management request: %w", err)
		}
		requestBody = bytes.NewReader(body)
	}
	url := strings.TrimRight(baseURL, "/") + "/api/router/management/v1" + path
	if verbose {
		fmt.Printf("[Dashboard] %s %s\n", method, url)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, requestBody)
	if err != nil {
		return nil, fmt.Errorf("create Management request: %w", err)
	}
	req.Header.Set("Accept", dashboardManagementMediaType)
	if payload != nil {
		req.Header.Set("Content-Type", dashboardManagementMediaType)
	}
	if namespaceID != "" {
		req.Header.Set(dashboardManagementNamespace, namespaceID)
	}
	if idempotencyKey != "" {
		req.Header.Set(dashboardManagementIdempotency, idempotencyKey)
	}
	for name, values := range requestHeaders {
		for _, value := range values {
			req.Header.Add(name, value)
		}
	}
	setDashboardAuth(req, token)
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("Management request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, _ := io.ReadAll(resp.Body)
	defer clear(body)
	if resp.StatusCode != expectedStatus {
		return nil, fmt.Errorf(
			"Management request: expected %d, got %d: %s",
			expectedStatus, resp.StatusCode, truncateString(string(body), 300),
		)
	}
	if result == nil {
		return resp.Header.Clone(), nil
	}
	if err := json.Unmarshal(body, result); err != nil {
		return nil, fmt.Errorf("Management response is not valid JSON: %w", err)
	}
	return resp.Header.Clone(), nil
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
