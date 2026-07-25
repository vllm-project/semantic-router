package testcases

import (
	"context"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("workflow-resume-restart-recovery", pkgtestcases.TestCase{
		Description: "Workflow states persist in Redis across Semantic Router pod restarts",
		Tags:        []string{"workflow", "functional", "redis", "restart"},
		Fn:          testWorkflowResumeRestartRecovery,
	})
}

func testWorkflowResumeRestartRecovery(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Workflow Resume: restart recovery (Redis persistence)")
	}

	// 1. Create a workflow response and get its ID before restart
	responseID, err := triggerWorkflowStateBeforeRestart(ctx, client, opts)
	if err != nil {
		return err
	}

	if opts.Verbose {
		fmt.Printf("[Test] Triggered workflow state %s. Restarting semantic-router pod...\n", responseID)
	}

	// 2. Kill the router pod
	if err := deleteSemanticRouterPod(ctx, client, opts); err != nil {
		return err
	}

	// 3. Wait for the new pod to spin up
	if err := waitForSemanticRouterReady(ctx, client, opts); err != nil {
		return err
	}

	if opts.Verbose {
		fmt.Println("[Test] Pod restarted. Validating workflow state persistence...")
	}

	// 4. Verify state persistence after restart
	return verifyWorkflowStateAfterRestart(ctx, client, opts, responseID)
}

func triggerWorkflowStateBeforeRestart(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) (string, error) {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return "", fmt.Errorf("open session for pre-restart workflow: %w", err)
	}
	defer session.Close()

	apiClient := fixtures.NewResponseAPIClient(session, 30*time.Second)
	
	// Issue a request with Store enabled to trigger workflow state tracking
	storeTrue := true
	apiResp, _, err := apiClient.Create(ctx, fixtures.ResponseAPIRequest{
		Model:        "openai/gpt-oss-20b",
		Input:        "Wait here for further instructions.",
		Instructions: "You are a helpful assistant.",
		Store:        &storeTrue,
	})
	if err != nil {
		return "", fmt.Errorf("workflow creation failed: %w", err)
	}

	if apiResp.ID == "" {
		return "", fmt.Errorf("workflow creation returned empty response ID")
	}

	// Ensure it persisted to Redis immediately
	if err := assertRedisResponseStored(ctx, client, apiResp.ID, opts); err != nil {
		return "", fmt.Errorf("workflow state not found in Redis before restart: %w", err)
	}

	// Extend the Redis key TTL so it survives the multi-minute restart cycle,
	// allowing us to keep the short default TTL in values.yaml.
	if err := extendRedisKeyTTL(ctx, client, apiResp.ID, 600, opts); err != nil {
		return "", fmt.Errorf("failed to extend Redis TTL for workflow restart test: %w", err)
	}

	return apiResp.ID, nil
}

func verifyWorkflowStateAfterRestart(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions, responseID string) error {
	const verifyTimeout = 90 * time.Second
	deadline := time.Now().Add(verifyTimeout)
	var lastErr error

	for time.Now().Before(deadline) {
		err := assertRedisResponseStored(ctx, client, responseID, opts)
		if err == nil {
			if opts.Verbose {
				fmt.Printf("[Test] Workflow state %s survived restart\n", responseID)
			}
			if opts.SetDetails != nil {
				opts.SetDetails(map[string]interface{}{
					"workflow_id": responseID,
					"survived":    true,
				})
			}
			return nil
		}
		lastErr = err
		time.Sleep(3 * time.Second)
	}

	return fmt.Errorf("workflow state %s not found in redis after %s: %w", responseID, verifyTimeout, lastErr)
}
