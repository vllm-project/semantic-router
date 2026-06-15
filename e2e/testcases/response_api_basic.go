package testcases

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

const (
	redisNamespace         = "default"
	redisResponseKeyPrefix = "sr:response:"
)

func init() {
	pkgtestcases.Register("response-api-create", pkgtestcases.TestCase{
		Description: "POST /v1/responses - Create a new response",
		Tags:        []string{"response-api", "functional"},
		Fn:          testResponseAPICreate,
	})
	pkgtestcases.Register("response-api-get", pkgtestcases.TestCase{
		Description: "GET /v1/responses/{id} - Retrieve a response",
		Tags:        []string{"response-api", "functional"},
		Fn:          testResponseAPIGet,
	})
	pkgtestcases.Register("response-api-delete", pkgtestcases.TestCase{
		Description: "DELETE /v1/responses/{id} - Delete a response",
		Tags:        []string{"response-api", "functional"},
		Fn:          testResponseAPIDelete,
	})
	pkgtestcases.Register("response-api-input-items", pkgtestcases.TestCase{
		Description: "GET /v1/responses/{id}/input_items - List input items",
		Tags:        []string{"response-api", "functional"},
		Fn:          testResponseAPIInputItems,
	})
	pkgtestcases.Register("response-api-ttl-expiry", pkgtestcases.TestCase{
		Description: "Response API TTL expiry - Response should disappear after TTL",
		Tags:        []string{"response-api", "functional", "redis"},
		Fn:          testResponseAPITTLExpiry,
	})
}

func testResponseAPICreate(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Response API: POST /v1/responses")
	}

	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	storeTrue := true
	apiClient := fixtures.NewResponseAPIClient(session, 30*time.Second)
	apiResp, _, err := apiClient.Create(ctx, fixtures.ResponseAPIRequest{
		Model:        "openai/gpt-oss-20b",
		Input:        "What is 2 + 2?",
		Instructions: "You are a helpful math assistant.",
		Store:        &storeTrue,
		Metadata:     map[string]string{"test": "response-api-create"},
	})
	if err != nil {
		return err
	}

	if apiResp.ID == "" || !strings.HasPrefix(apiResp.ID, "resp_") {
		return fmt.Errorf("invalid response ID: %s (expected resp_xxx format)", apiResp.ID)
	}
	if apiResp.Object != "response" {
		return fmt.Errorf("invalid object type: %s (expected 'response')", apiResp.Object)
	}
	if apiResp.Status != "completed" && apiResp.Status != "in_progress" {
		return fmt.Errorf("unexpected status: %s", apiResp.Status)
	}
	if apiResp.CreatedAt == 0 {
		return fmt.Errorf("created_at should not be zero")
	}

	if err := assertRedisResponseStored(ctx, client, apiResp.ID, opts); err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"response_id": apiResp.ID,
			"status":      apiResp.Status,
			"model":       apiResp.Model,
		})
	}
	return nil
}

func testResponseAPIGet(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Response API: GET /v1/responses/{id}")
	}

	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	apiClient := fixtures.NewResponseAPIClient(session, 30*time.Second)
	responseID, err := createTestResponse(ctx, apiClient, opts.Verbose)
	if err != nil {
		return fmt.Errorf("failed to create test response: %w", err)
	}

	apiResp, _, err := apiClient.Get(ctx, responseID)
	if err != nil {
		return err
	}
	if apiResp.ID != responseID {
		return fmt.Errorf("response ID mismatch: got %s, expected %s", apiResp.ID, responseID)
	}
	if apiResp.Object != "response" {
		return fmt.Errorf("invalid object type: %s", apiResp.Object)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"response_id": apiResp.ID,
			"status":      apiResp.Status,
		})
	}
	return nil
}

func testResponseAPIDelete(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Response API: DELETE /v1/responses/{id}")
	}

	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	apiClient := fixtures.NewResponseAPIClient(session, 30*time.Second)
	responseID, err := createTestResponse(ctx, apiClient, opts.Verbose)
	if err != nil {
		return fmt.Errorf("failed to create test response: %w", err)
	}

	deleteResp, _, err := apiClient.Delete(ctx, responseID)
	if err != nil {
		return err
	}
	if deleteResp.ID != responseID {
		return fmt.Errorf("response ID mismatch: got %s, expected %s", deleteResp.ID, responseID)
	}
	if deleteResp.Object != "response.deleted" {
		return fmt.Errorf("invalid object type: %s (expected 'response.deleted')", deleteResp.Object)
	}
	if !deleteResp.Deleted {
		return fmt.Errorf("deleted should be true")
	}

	rawGet, err := apiClient.GetRaw(ctx, responseID)
	if err != nil {
		return fmt.Errorf("failed to verify deletion: %w", err)
	}
	if rawGet.StatusCode != 404 {
		return fmt.Errorf("expected 404 after deletion, got %d", rawGet.StatusCode)
	}

	if err := assertRedisResponseDeleted(ctx, client, responseID, opts); err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"deleted_id": responseID,
			"verified":   true,
		})
	}
	return nil
}

func testResponseAPIInputItems(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Response API: GET /v1/responses/{id}/input_items")
	}

	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	apiClient := fixtures.NewResponseAPIClient(session, 30*time.Second)
	responseID, err := createTestResponseWithInstructions(ctx, apiClient, opts.Verbose)
	if err != nil {
		return fmt.Errorf("failed to create test response: %w", err)
	}

	listResp, _, err := apiClient.InputItems(ctx, responseID)
	if err != nil {
		return err
	}
	if listResp.Object != "list" {
		return fmt.Errorf("invalid object type: %s (expected 'list')", listResp.Object)
	}
	if len(listResp.Data) == 0 {
		return fmt.Errorf("expected at least one input item")
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"response_id": responseID,
			"item_count":  len(listResp.Data),
			"has_more":    listResp.HasMore,
		})
	}
	return nil
}

func createTestResponse(ctx context.Context, apiClient *fixtures.ResponseAPIClient, verbose bool) (string, error) {
	storeTrue := true
	apiResp, _, err := apiClient.Create(ctx, fixtures.ResponseAPIRequest{
		Model: "openai/gpt-oss-20b",
		Input: "Hello, how are you?",
		Store: &storeTrue,
	})
	if err != nil {
		return "", err
	}
	if verbose {
		fmt.Printf("[Test] Created test response: %s\n", apiResp.ID)
	}
	return apiResp.ID, nil
}

func createTestResponseWithInstructions(ctx context.Context, apiClient *fixtures.ResponseAPIClient, verbose bool) (string, error) {
	storeTrue := true
	apiResp, _, err := apiClient.Create(ctx, fixtures.ResponseAPIRequest{
		Model:        "openai/gpt-oss-20b",
		Input:        "What is the capital of France?",
		Instructions: "You are a geography expert. Answer concisely.",
		Store:        &storeTrue,
	})
	if err != nil {
		return "", err
	}
	if verbose {
		fmt.Printf("[Test] Created test response with instructions: %s\n", apiResp.ID)
	}
	return apiResp.ID, nil
}

func testResponseAPITTLExpiry(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Response API: TTL expiry")
	}

	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	apiClient := fixtures.NewResponseAPIClient(session, 10*time.Second)
	responseID, err := createTestResponse(ctx, apiClient, opts.Verbose)
	if err != nil {
		return fmt.Errorf("failed to create test response: %w", err)
	}

	if err := assertRedisResponseTTLSet(ctx, client, responseID, opts); err != nil {
		return err
	}

	rawGet, err := apiClient.GetRaw(ctx, responseID)
	if err != nil {
		return fmt.Errorf("failed to confirm response existence: %w", err)
	}
	if rawGet.StatusCode != 200 {
		return fmt.Errorf("expected status 200 before TTL expiry, got %d", rawGet.StatusCode)
	}

	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		rawGet, err = apiClient.GetRaw(ctx, responseID)
		if err != nil {
			return fmt.Errorf("failed to poll response TTL: %w", err)
		}
		if rawGet.StatusCode == 404 {
			return nil
		}
		time.Sleep(1 * time.Second)
	}

	return fmt.Errorf("expected response to expire (404) within timeout, id=%s", responseID)
}

// -------- Redis persistence assertions (Response API E2E) --------
func assertRedisResponseStored(ctx context.Context, client *kubernetes.Clientset, responseID string, opts pkgtestcases.TestCaseOptions) error {
	podName, useCluster, found, err := getRedisPod(ctx, client)
	if err != nil {
		return err
	}
	if !found {
		if opts.Verbose {
			fmt.Println("[Test] Redis pod not found; skipping direct Redis checks")
		}
		return nil
	}

	key := redisResponseKeyPrefix + responseID
	output, err := execRedisCli(ctx, podName, useCluster, opts.Verbose, "GET", key)
	if err != nil {
		return err
	}
	if output == "" || output == "(nil)" {
		return fmt.Errorf("expected Redis key to exist, got empty response for key %s", key)
	}

	var stored map[string]interface{}
	raw := &fixtures.HTTPResponse{Body: []byte(output)}
	if err := raw.DecodeJSON(&stored); err != nil {
		return fmt.Errorf("failed to parse Redis value for key %s: %w", key, err)
	}
	if id, ok := stored["id"].(string); !ok || id != responseID {
		return fmt.Errorf("unexpected Redis response id: got %v, expected %s", stored["id"], responseID)
	}

	return nil
}

func assertRedisResponseDeleted(ctx context.Context, client *kubernetes.Clientset, responseID string, opts pkgtestcases.TestCaseOptions) error {
	podName, useCluster, found, err := getRedisPod(ctx, client)
	if err != nil {
		return err
	}
	if !found {
		if opts.Verbose {
			fmt.Println("[Test] Redis pod not found; skipping direct Redis checks")
		}
		return nil
	}

	key := redisResponseKeyPrefix + responseID
	output, err := execRedisCli(ctx, podName, useCluster, opts.Verbose, "EXISTS", key)
	if err != nil {
		return err
	}
	if strings.TrimSpace(output) != "0" {
		return fmt.Errorf("expected Redis key to be deleted, EXISTS returned %q for key %s", output, key)
	}

	return nil
}

func assertRedisResponseTTLSet(ctx context.Context, client *kubernetes.Clientset, responseID string, opts pkgtestcases.TestCaseOptions) error {
	podName, useCluster, found, err := getRedisPod(ctx, client)
	if err != nil {
		return err
	}
	if !found {
		if opts.Verbose {
			fmt.Println("[Test] Redis pod not found; skipping direct Redis checks")
		}
		return nil
	}

	key := redisResponseKeyPrefix + responseID
	output, err := execRedisCli(ctx, podName, useCluster, opts.Verbose, "TTL", key)
	if err != nil {
		return err
	}
	ttl, err := strconv.Atoi(strings.TrimSpace(output))
	if err != nil {
		return fmt.Errorf("unexpected TTL output %q for key %s: %w", output, key, err)
	}
	if ttl <= 0 {
		return fmt.Errorf("expected Redis TTL to be set for key %s, got %d", key, ttl)
	}

	return nil
}

func getRedisPod(ctx context.Context, client *kubernetes.Clientset) (podName string, useCluster bool, found bool, err error) {
	podName, err = findRedisPod(ctx, client, "app=redis-cluster")
	if err != nil {
		return "", false, false, err
	}
	if podName != "" {
		return podName, true, true, nil
	}

	podName, err = findRedisPod(ctx, client, "app=redis")
	if err != nil {
		return "", false, false, err
	}
	if podName != "" {
		return podName, false, true, nil
	}

	return "", false, false, nil
}

func findRedisPod(ctx context.Context, client *kubernetes.Clientset, labelSelector string) (string, error) {
	pods, err := client.CoreV1().Pods(redisNamespace).List(ctx, metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err != nil {
		return "", fmt.Errorf("failed to list pods for selector %q: %w", labelSelector, err)
	}
	for i := range pods.Items {
		pod := pods.Items[i]
		if pod.Status.Phase == corev1.PodRunning {
			return pod.Name, nil
		}
	}
	if len(pods.Items) > 0 {
		return pods.Items[0].Name, nil
	}
	return "", nil
}

func execRedisCli(ctx context.Context, podName string, useCluster bool, verbose bool, args ...string) (string, error) {
	cmdArgs := []string{"exec", "-n", redisNamespace, podName, "--", "redis-cli"}
	if useCluster {
		cmdArgs = append(cmdArgs, "-c")
	}
	cmdArgs = append(cmdArgs, args...)
	if verbose {
		fmt.Printf("[Test] Redis CLI: kubectl %s\n", strings.Join(cmdArgs, " "))
	}
	cmd := exec.CommandContext(ctx, "kubectl", cmdArgs...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("redis-cli failed: %w: %s", err, strings.TrimSpace(string(output)))
	}
	result := strings.TrimSpace(string(output))
	if verbose {
		fmt.Printf("[Test] Redis CLI output: %s\n", truncateString(result, 200))
	}
	return result, nil
}
