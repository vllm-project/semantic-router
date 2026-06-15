package testcases

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

const (
	semanticRouterNamespace  = "vllm-semantic-router-system"
	semanticRouterDeployment = "semantic-router"
	semanticRouterPodLabel   = "app=semantic-router"

	restartRecoveryTimeout  = 5 * time.Minute
	restartRecoveryInterval = 5 * time.Second
	restartKeyTTLSeconds    = 600
)

func init() {
	pkgtestcases.Register("response-api-restart-recovery", pkgtestcases.TestCase{
		Description: "Response API data stored in Redis survives a semantic-router pod restart",
		Tags:        []string{"response-api", "functional", "redis", "restart"},
		Fn:          testResponseAPIRestartRecovery,
	})
}

func testResponseAPIRestartRecovery(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Response API: restart recovery (Redis persistence)")
	}

	responseID, model, err := createResponseBeforeRestart(ctx, client, opts)
	if err != nil {
		return err
	}

	if err := deleteSemanticRouterPod(ctx, client, opts); err != nil {
		return err
	}

	if err := waitForSemanticRouterReady(ctx, client, opts); err != nil {
		return err
	}

	return verifyResponseAfterRestart(ctx, client, opts, responseID, model)
}

func createResponseBeforeRestart(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) (string, string, error) {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return "", "", fmt.Errorf("failed to open session for pre-restart create: %w", err)
	}
	defer session.Close()

	storeTrue := true
	apiClient := fixtures.NewResponseAPIClient(session, 30*time.Second)
	apiResp, _, err := apiClient.Create(ctx, fixtures.ResponseAPIRequest{
		Model:        "openai/gpt-oss-20b",
		Input:        "What is the speed of light?",
		Instructions: "You are a physics assistant.",
		Store:        &storeTrue,
		Metadata:     map[string]string{"test": "restart-recovery"},
	})
	if err != nil {
		return "", "", fmt.Errorf("failed to create response before restart: %w", err)
	}

	if apiResp.ID == "" || !strings.HasPrefix(apiResp.ID, "resp_") {
		return "", "", fmt.Errorf("invalid response ID: %s", apiResp.ID)
	}

	if err := assertRedisResponseStored(ctx, client, apiResp.ID, opts); err != nil {
		return "", "", fmt.Errorf("response not found in Redis before restart: %w", err)
	}

	// The profile may use a short TTL (e.g. 10s) for the TTL-expiry test.
	// Extend this key so it survives the multi-minute restart cycle.
	if err := extendRedisKeyTTL(ctx, client, apiResp.ID, restartKeyTTLSeconds, opts); err != nil {
		return "", "", fmt.Errorf("failed to extend Redis TTL for restart test: %w", err)
	}

	if opts.Verbose {
		fmt.Printf("[Test] Created response %s (model=%s) — stored in Redis with extended TTL\n", apiResp.ID, apiResp.Model)
	}

	return apiResp.ID, apiResp.Model, nil
}

func deleteSemanticRouterPod(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	pods, err := client.CoreV1().Pods(semanticRouterNamespace).List(ctx, metav1.ListOptions{
		LabelSelector: semanticRouterPodLabel,
	})
	if err != nil {
		return fmt.Errorf("failed to list semantic-router pods: %w", err)
	}
	if len(pods.Items) == 0 {
		return fmt.Errorf("no semantic-router pods found in %s", semanticRouterNamespace)
	}

	podName := pods.Items[0].Name
	if opts.Verbose {
		fmt.Printf("[Test] Deleting semantic-router pod %s to simulate restart\n", podName)
	}

	if err := client.CoreV1().Pods(semanticRouterNamespace).Delete(ctx, podName, metav1.DeleteOptions{}); err != nil {
		return fmt.Errorf("failed to delete semantic-router pod %s: %w", podName, err)
	}

	return waitForOldPodTerminated(ctx, client, podName, opts)
}

// waitForOldPodTerminated blocks until the deleted pod is gone, so
// WaitForDeploymentReady won't return prematurely from stale status.
func waitForOldPodTerminated(ctx context.Context, client *kubernetes.Clientset, podName string, opts pkgtestcases.TestCaseOptions) error {
	deadline := time.Now().Add(2 * time.Minute)
	for time.Now().Before(deadline) {
		_, err := client.CoreV1().Pods(semanticRouterNamespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			if opts.Verbose {
				fmt.Printf("[Test] Old pod %s terminated\n", podName)
			}
			return nil
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("old pod %s still exists after 2m", podName)
}

func waitForSemanticRouterReady(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Printf("[Test] Waiting for semantic-router deployment to recover (timeout=%s)\n", restartRecoveryTimeout)
	}

	return helpers.WaitForDeploymentReady(
		ctx, client,
		semanticRouterNamespace, semanticRouterDeployment,
		restartRecoveryTimeout, restartRecoveryInterval,
		opts.Verbose,
	)
}

func verifyResponseAfterRestart(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions, responseID, expectedModel string) error {
	// The pod may pass readiness checks before the Response API / Redis
	// connection is fully initialized. Poll until the GET succeeds.
	const verifyTimeout = 90 * time.Second
	deadline := time.Now().Add(verifyTimeout)
	var lastErr error

	for time.Now().Before(deadline) {
		session, err := fixtures.OpenServiceSession(ctx, client, opts)
		if err != nil {
			lastErr = err
			time.Sleep(3 * time.Second)
			continue
		}

		apiClient := fixtures.NewResponseAPIClient(session, 30*time.Second)
		apiResp, _, err := apiClient.Get(ctx, responseID)
		session.Close()

		if err != nil {
			lastErr = err
			if opts.Verbose {
				fmt.Printf("[Test] GET %s not ready yet: %v — retrying\n", responseID, err)
			}
			time.Sleep(3 * time.Second)
			continue
		}

		if apiResp.ID != responseID {
			return fmt.Errorf("response ID mismatch after restart: got %s, expected %s", apiResp.ID, responseID)
		}
		if apiResp.Object != "response" {
			return fmt.Errorf("invalid object type after restart: %s", apiResp.Object)
		}

		if opts.Verbose {
			fmt.Printf("[Test] Response %s survived restart (model=%s, status=%s)\n", apiResp.ID, apiResp.Model, apiResp.Status)
		}
		if opts.SetDetails != nil {
			opts.SetDetails(map[string]interface{}{
				"response_id":    responseID,
				"survived":       true,
				"model":          apiResp.Model,
				"status":         apiResp.Status,
				"expected_model": expectedModel,
			})
		}
		return nil
	}

	return fmt.Errorf("response %s not retrievable after %s: %w", responseID, verifyTimeout, lastErr)
}

func extendRedisKeyTTL(ctx context.Context, client *kubernetes.Clientset, responseID string, ttlSeconds int, opts pkgtestcases.TestCaseOptions) error {
	podName, useCluster, found, err := getRedisPod(ctx, client)
	if err != nil {
		return err
	}
	if !found {
		if opts.Verbose {
			fmt.Println("[Test] Redis pod not found; skipping TTL extension")
		}
		return nil
	}

	key := redisResponseKeyPrefix + responseID
	output, err := execRedisCli(ctx, podName, useCluster, opts.Verbose,
		"EXPIRE", key, fmt.Sprintf("%d", ttlSeconds))
	if err != nil {
		return fmt.Errorf("EXPIRE failed for key %s: %w", key, err)
	}
	if strings.TrimSpace(output) != "1" {
		return fmt.Errorf("EXPIRE returned %q for key %s (expected 1)", output, key)
	}

	if opts.Verbose {
		fmt.Printf("[Test] Extended TTL on %s to %ds\n", key, ttlSeconds)
	}
	return nil
}
