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

const routerReplayTableName = "router_replay"

const routerReplayManagementToken = "router-replay-e2e-viewer-token"

func init() {
	pkgtestcases.Register("router-replay-restart-recovery", pkgtestcases.TestCase{
		Description: "Router Replay records stored in Postgres survive a semantic-router pod restart",
		Tags:        []string{"router-replay", "functional", "postgres", "restart"},
		Fn:          testRouterReplayRestartRecovery,
	})
}

func testRouterReplayRestartRecovery(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Router Replay: restart recovery (Postgres persistence)")
	}

	expected, err := triggerReplayRecordBeforeRestart(ctx, client, opts)
	if err != nil {
		return err
	}

	if err := deleteSemanticRouterPod(ctx, client, opts); err != nil {
		return err
	}

	if err := waitForSemanticRouterReady(ctx, client, opts); err != nil {
		return err
	}

	return verifyReplayRecordAfterRestart(ctx, client, opts, expected.RecordID, expected.PreparedDispatch)
}

type replayRecoveryExpectation struct {
	RecordID         string
	PreparedDispatch preparedDispatchReceipt
}

// triggerReplayRecordBeforeRestart sends a chat completion through the router,
// waits for the replay record to appear, and confirms it is persisted in Postgres.
func triggerReplayRecordBeforeRestart(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) (*replayRecoveryExpectation, error) {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return nil, fmt.Errorf("open session for pre-restart chat: %w", err)
	}
	defer session.Close()
	apiSession, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return nil, fmt.Errorf("open Router management API session: %w", err)
	}
	defer apiSession.Close()
	providerSession, err := fixtures.OpenServiceEndpointSession(ctx, client, opts, "default", "provider-mocker", "8000")
	if err != nil {
		return nil, fmt.Errorf("open provider-mocker observer session: %w", err)
	}
	defer providerSession.Close()

	providerObservationID := fmt.Sprintf("prepared-dispatch-%d", time.Now().UnixNano())
	chatClient := fixtures.NewChatCompletionsClient(session, 30*time.Second)
	resp, err := chatClient.Create(ctx, fixtures.ChatCompletionsRequest{
		Model: "auto",
		User:  "e2e-replay-user",
		Messages: []fixtures.ChatMessage{
			{Role: "user", Content: "What is 2+2? Reply with just the number."},
		},
	}, map[string]string{
		"x-authz-user-id":       "e2e-replay-user",
		"x-vsr-test-session-id": providerObservationID,
	})
	if err != nil {
		return nil, fmt.Errorf("chat completion failed: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("chat completion returned status %d: %s", resp.StatusCode, string(resp.Body))
	}

	if opts.Verbose {
		fmt.Println("[Test] Chat completion succeeded — waiting for replay record")
	}
	time.Sleep(3 * time.Second)

	recordID := strings.TrimSpace(resp.Headers.Get("x-vsr-replay-id"))
	if recordID == "" {
		return nil, fmt.Errorf("chat completion response is missing x-vsr-replay-id")
	}

	if err := assertPostgresReplayRecordStored(ctx, client, recordID, opts); err != nil {
		return nil, fmt.Errorf("replay record not confirmed in Postgres before restart: %w", err)
	}
	if err := assertReplayRecordHasSessionMetadata(apiSession, recordID, opts.Verbose); err != nil {
		return nil, err
	}
	receipt, err := assertPreparedDispatchMatchesProvider(ctx, apiSession, providerSession, recordID, providerObservationID, opts.Verbose)
	if err != nil {
		return nil, err
	}
	return &replayRecoveryExpectation{RecordID: recordID, PreparedDispatch: receipt}, nil
}

// replayListResponse mirrors the JSON shape returned by GET /api/v1/observability/replays.
type replayListResponse struct {
	Object string          `json:"object"`
	Count  int             `json:"count"`
	Data   json.RawMessage `json:"data"`
}

// replayRecordSummary captures fields read from replay API JSON.
type replayRecordSummary struct {
	ID               string `json:"id"`
	SessionID        string `json:"session_id"`
	TurnIndex        int    `json:"turn_index"`
	RouteDiagnostics struct {
		PreparedDispatch *preparedDispatchReceipt `json:"prepared_dispatch"`
	} `json:"route_diagnostics"`
}

type preparedDispatchReceipt struct {
	Version    int    `json:"version"`
	WireFormat string `json:"wire_format"`
	SHA256     string `json:"sha256"`
	ByteLength int    `json:"byte_length"`
}

// fetchFirstReplayRecordID returns the first record ID from a Replay list target.
// When verbose is true, it prints the full JSON response.
func fetchFirstReplayRecordID(managementSession *fixtures.ServiceSession, requestTarget string, verbose bool) (string, error) {
	raw, err := doRouterReplayManagementGET(context.Background(), managementSession, requestTarget)
	if err != nil {
		return "", fmt.Errorf("GET /api/v1/observability/replays failed: %w", err)
	}
	if raw.StatusCode != http.StatusOK {
		return "", fmt.Errorf("GET /api/v1/observability/replays returned status %d: %s", raw.StatusCode, string(raw.Body))
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

func assertReplayRecordHasSessionMetadata(managementSession *fixtures.ServiceSession, recordID string, verbose bool) error {
	raw, err := doRouterReplayManagementGET(context.Background(), managementSession, "/api/v1/observability/replays/"+recordID)
	if err != nil {
		return fmt.Errorf("GET replay record for session metadata: %w", err)
	}
	if raw.StatusCode != http.StatusOK {
		return fmt.Errorf("GET replay record returned status %d: %s", raw.StatusCode, string(raw.Body))
	}
	var rec replayRecordSummary
	if err := raw.DecodeJSON(&rec); err != nil {
		return fmt.Errorf("decode replay record: %w", err)
	}
	if rec.SessionID == "" {
		return fmt.Errorf("replay record %s missing session_id", recordID)
	}
	if verbose {
		fmt.Printf("[Test] Replay session_id=%q turn_index=%d\n", rec.SessionID, rec.TurnIndex)
	}
	return nil
}

func assertPreparedDispatchMatchesProvider(
	ctx context.Context,
	managementSession *fixtures.ServiceSession,
	providerSession *fixtures.ServiceSession,
	recordID string,
	providerObservationID string,
	verbose bool,
) (preparedDispatchReceipt, error) {
	observedBody, err := lastProviderSimulatorRequest(ctx, providerSession, providerObservationID)
	if err != nil {
		return preparedDispatchReceipt{}, fmt.Errorf("read provider-mocker request receipt: %w", err)
	}
	var observed struct {
		SHA256     string `json:"body_sha256"`
		ByteLength int    `json:"body_bytes"`
	}
	unmarshalErr := json.Unmarshal(observedBody, &observed)
	if unmarshalErr != nil {
		return preparedDispatchReceipt{}, fmt.Errorf("decode provider-mocker request receipt: %w", unmarshalErr)
	}
	if observed.SHA256 == "" || observed.ByteLength <= 0 {
		return preparedDispatchReceipt{}, fmt.Errorf("provider-mocker request receipt is incomplete: sha256=%q bytes=%d", observed.SHA256, observed.ByteLength)
	}

	raw, err := doRouterReplayManagementGETAs(
		ctx,
		managementSession,
		"/api/v1/observability/replays/"+recordID,
		routerReplayDetailToken,
	)
	if err != nil {
		return preparedDispatchReceipt{}, fmt.Errorf("GET replay record for prepared dispatch: %w", err)
	}
	if raw.StatusCode != http.StatusOK {
		return preparedDispatchReceipt{}, fmt.Errorf("GET replay record returned status %d: %s", raw.StatusCode, string(raw.Body))
	}
	var record replayRecordSummary
	if err := raw.DecodeJSON(&record); err != nil {
		return preparedDispatchReceipt{}, fmt.Errorf("decode prepared-dispatch Replay record: %w", err)
	}
	receipt := record.RouteDiagnostics.PreparedDispatch
	if receipt == nil {
		return preparedDispatchReceipt{}, fmt.Errorf("replay record %s is missing route_diagnostics.prepared_dispatch", recordID)
	}
	if receipt.Version != 1 || receipt.WireFormat != "openai.chat.v1" ||
		receipt.SHA256 != observed.SHA256 || receipt.ByteLength != observed.ByteLength {
		return preparedDispatchReceipt{}, fmt.Errorf("prepared dispatch does not match provider-mocker observation: replay=%+v provider=%+v", receipt, observed)
	}
	if verbose {
		fmt.Printf("[Test] Prepared dispatch matched provider-mocker: sha256=%s bytes=%d\n", receipt.SHA256, receipt.ByteLength)
	}
	return *receipt, nil
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

	tableName := routerReplayTableName
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

// verifyReplayRecordAfterRestart polls GET /api/v1/observability/replays/{id} until the
// record is accessible again after the pod restart.
func verifyReplayRecordAfterRestart(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	recordID string,
	expectedReceipt preparedDispatchReceipt,
) error {
	const verifyTimeout = 90 * time.Second
	deadline := time.Now().Add(verifyTimeout)
	var lastErr error

	for time.Now().Before(deadline) {
		err := fetchReplayRecordOnce(ctx, client, opts, recordID, expectedReceipt)
		if err == nil {
			return nil
		}
		lastErr = err
		time.Sleep(3 * time.Second)
	}

	return fmt.Errorf("replay record %s not retrievable after %s: %w", recordID, verifyTimeout, lastErr)
}

// fetchReplayRecordOnce tries a single GET /api/v1/observability/replays/{id} and returns
// nil when the record is found and valid.
func fetchReplayRecordOnce(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	recordID string,
	expectedReceipt preparedDispatchReceipt,
) error {
	managementSession, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer managementSession.Close()

	raw, err := doRouterReplayManagementGETAs(
		ctx,
		managementSession,
		"/api/v1/observability/replays/"+recordID,
		routerReplayDetailToken,
	)
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
	if record.SessionID == "" {
		return fmt.Errorf("replay record %s missing session_id after restart", recordID)
	}
	receipt := record.RouteDiagnostics.PreparedDispatch
	if receipt == nil {
		return fmt.Errorf("replay record %s missing prepared dispatch after restart", recordID)
	}
	if *receipt != expectedReceipt {
		return fmt.Errorf("prepared dispatch changed after restart: got=%+v expected=%+v", *receipt, expectedReceipt)
	}

	if opts.Verbose {
		fmt.Printf("[Test] Replay record %s and prepared dispatch survived restart\n", recordID)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"record_id": recordID,
			"survived":  true,
		})
	}
	return nil
}

func doRouterReplayManagementGET(
	ctx context.Context,
	managementSession *fixtures.ServiceSession,
	requestTarget string,
) (*fixtures.HTTPResponse, error) {
	return doRouterReplayManagementGETAs(ctx, managementSession, requestTarget, routerReplayManagementToken)
}

// doRouterReplayManagementGETAs issues a management GET with an explicit
// bearer token, so a case can pick the role its assertions need.
func doRouterReplayManagementGETAs(
	ctx context.Context,
	managementSession *fixtures.ServiceSession,
	requestTarget string,
	token string,
) (*fixtures.HTTPResponse, error) {
	return fixtures.DoGETRequestWithHeaders(
		ctx,
		managementSession.HTTPClient(30*time.Second),
		managementSession.BaseURL()+requestTarget,
		map[string]string{"Authorization": "Bearer " + token},
	)
}
