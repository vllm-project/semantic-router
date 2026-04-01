package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("router-replay-restart-recovery", pkgtestcases.TestCase{
		Description: "Router Replay records stored in Postgres survive a semantic-router pod restart",
		Tags:        []string{"router-replay", "functional", "postgres", "restart"},
		Fn:          testRouterReplayRestartRecovery,
	})
}

func testRouterReplayRestartRecovery(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Router Replay: restart recovery (Redis persistence)")
	}

	recordID, err := triggerReplayRecordBeforeRestart(ctx, client, opts)
	if err != nil {
		return err
	}

	if err := deleteSemanticRouterPod(ctx, client, opts); err != nil {
		return err
	}

	if err := waitForSemanticRouterReady(ctx, client, opts); err != nil {
		return err
	}

	return verifyReplayRecordAfterRestart(ctx, client, opts, recordID)
}

// triggerReplayRecordBeforeRestart sends a chat completion through the router,
// waits for the replay record to appear, and confirms it is persisted in Redis.
func triggerReplayRecordBeforeRestart(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) (string, error) {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return "", fmt.Errorf("open session for pre-restart chat: %w", err)
	}
	defer session.Close()

	chatClient := fixtures.NewChatCompletionsClient(session, 30*time.Second)
	resp, err := chatClient.Create(ctx, fixtures.ChatCompletionsRequest{
		Model: "auto",
		Messages: []fixtures.ChatMessage{
			{Role: "user", Content: "What is 2+2? Reply with just the number."},
		},
	}, nil)
	if err != nil {
		return "", fmt.Errorf("chat completion failed: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("chat completion returned status %d: %s", resp.StatusCode, string(resp.Body))
	}

	if opts.Verbose {
		fmt.Println("[Test] Chat completion succeeded — waiting for replay record")
	}
	time.Sleep(3 * time.Second)

	recordID, err := fetchFirstReplayRecordID(session, opts.Verbose)
	if err != nil {
		return "", err
	}

	if err := assertPostgresReplayRecordStored(ctx, client, recordID, opts); err != nil {
		return "", fmt.Errorf("replay record not confirmed in Postgres before restart: %w", err)
	}
	return recordID, nil
}

// replayListResponse mirrors the JSON shape returned by GET /v1/router_replay.
type replayListResponse struct {
	Object string          `json:"object"`
	Count  int             `json:"count"`
	Data   json.RawMessage `json:"data"`
}

// replayRecordSummary captures only the id field from a replay record.
type replayRecordSummary struct {
	ID string `json:"id"`
}

// fetchFirstReplayRecordID calls GET /v1/router_replay?limit=1 and returns the
// first record's ID. When verbose is true, prints the full JSON response.
func fetchFirstReplayRecordID(session *fixtures.ServiceSession, verbose bool) (string, error) {
	httpClient := session.HTTPClient(30 * time.Second)
	raw, err := fixtures.DoGETRequest(context.Background(), httpClient, session.BaseURL()+"/v1/router_replay?limit=1")
	if err != nil {
		return "", fmt.Errorf("GET /v1/router_replay failed: %w", err)
	}
	if raw.StatusCode != http.StatusOK {
		return "", fmt.Errorf("GET /v1/router_replay returned status %d: %s", raw.StatusCode, string(raw.Body))
	}

	if verbose {
		fmt.Printf("[Test] Replay list response (pre-restart):\n%s\n", prettyJSON(raw.Body))
	}

	var listResp replayListResponse
	if err := raw.DecodeJSON(&listResp); err != nil {
		return "", fmt.Errorf("decode replay list: %w", err)
	}
	if listResp.Count == 0 {
		return "", fmt.Errorf("no replay records found after chat completion")
	}

	var records []replayRecordSummary
	if err := json.Unmarshal(listResp.Data, &records); err != nil {
		return "", fmt.Errorf("decode replay records array: %w", err)
	}
	if len(records) == 0 || records[0].ID == "" {
		return "", fmt.Errorf("replay record has empty ID")
	}
	return records[0].ID, nil
}

func prettyJSON(data []byte) string {
	var buf json.RawMessage
	if err := json.Unmarshal(data, &buf); err != nil {
		return string(data)
	}
	pretty, err := json.MarshalIndent(buf, "  ", "  ")
	if err != nil {
		return string(data)
	}
	return string(pretty)
}

func assertPostgresReplayRecordStored(ctx context.Context, client *kubernetes.Clientset, recordID string, opts pkgtestcases.TestCaseOptions) error {
	podName, found, err := getPostgresPod(ctx, client)
	if err != nil {
		return err
	}
	if !found {
		return nil
	}

	tableName := "router_replay_default_decision"
	query := fmt.Sprintf("SELECT COUNT(*) FROM %s WHERE id = '%s'", tableName, recordID)
	output, err := execPsql(ctx, podName, opts.Verbose, query)
	if err != nil {
		return fmt.Errorf("psql query failed: %w", err)
	}
	if strings.TrimSpace(output) == "0" {
		return fmt.Errorf("replay record %s not found in Postgres", recordID)
	}
	if opts.Verbose {
		fmt.Printf("[Test] Replay record %s confirmed in Postgres\n", recordID)
	}
	return nil
}

func getPostgresPod(ctx context.Context, client *kubernetes.Clientset) (string, bool, error) {
	pods, err := client.CoreV1().Pods("default").List(ctx, metav1.ListOptions{
		LabelSelector: "app=postgres",
	})
	if err != nil {
		return "", false, fmt.Errorf("failed to list postgres pods: %w", err)
	}
	for i := range pods.Items {
		if pods.Items[i].Status.Phase == "Running" {
			return pods.Items[i].Name, true, nil
		}
	}
	if len(pods.Items) > 0 {
		return pods.Items[0].Name, true, nil
	}
	return "", false, nil
}

func execPsql(ctx context.Context, podName string, verbose bool, query string) (string, error) {
	cmdArgs := []string{
		"exec", "-n", "default", podName, "--",
		"psql", "-U", "router", "-d", "vsr", "-t", "-A", "-c", query,
	}
	if verbose {
		fmt.Printf("[Test] Postgres CLI: kubectl %s\n", strings.Join(cmdArgs, " "))
	}
	cmd := exec.CommandContext(ctx, "kubectl", cmdArgs...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("psql failed: %w: %s", err, strings.TrimSpace(string(output)))
	}
	result := strings.TrimSpace(string(output))
	if verbose {
		fmt.Printf("[Test] Postgres CLI output: %s\n", result)
	}
	return result, nil
}

// verifyReplayRecordAfterRestart polls GET /v1/router_replay/{id} until the
// record is accessible again after the pod restart.
func verifyReplayRecordAfterRestart(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions, recordID string) error {
	const verifyTimeout = 90 * time.Second
	deadline := time.Now().Add(verifyTimeout)
	var lastErr error

	for time.Now().Before(deadline) {
		err := fetchReplayRecordOnce(ctx, client, opts, recordID)
		if err == nil {
			return nil
		}
		lastErr = err
		time.Sleep(3 * time.Second)
	}

	return fmt.Errorf("replay record %s not retrievable after %s: %w", recordID, verifyTimeout, lastErr)
}

// fetchReplayRecordOnce tries a single GET /v1/router_replay/{id} and returns
// nil when the record is found and valid.
func fetchReplayRecordOnce(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions, recordID string) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	httpClient := session.HTTPClient(30 * time.Second)
	raw, err := fixtures.DoGETRequest(ctx, httpClient, session.BaseURL()+"/v1/router_replay/"+recordID)
	if err != nil {
		if opts.Verbose {
			fmt.Printf("[Test] GET replay %s not ready yet: %v — retrying\n", recordID, err)
		}
		return err
	}

	if raw.StatusCode != http.StatusOK {
		retryErr := fmt.Errorf("expected status 200, got %d: %s", raw.StatusCode, string(raw.Body))
		if opts.Verbose {
			fmt.Printf("[Test] GET replay %s returned %d — retrying\n", recordID, raw.StatusCode)
		}
		return retryErr
	}

	if opts.Verbose {
		fmt.Printf("[Test] Replay record response (post-restart):\n%s\n", prettyJSON(raw.Body))
	}

	var record replayRecordSummary
	if err := raw.DecodeJSON(&record); err != nil {
		return fmt.Errorf("decode replay record after restart: %w", err)
	}
	if record.ID != recordID {
		return fmt.Errorf("replay record ID mismatch: got %s, expected %s", record.ID, recordID)
	}

	if opts.Verbose {
		fmt.Printf("[Test] Replay record %s survived restart\n", recordID)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"record_id": recordID,
			"survived":  true,
		})
	}
	return nil
}
