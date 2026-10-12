package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("history-reset-recovery", pkgtestcases.TestCase{
		Description: "An accepted history_reset stores the removed turn in Redis before dispatch and offers only its issued key to the model",
		Tags:        []string{"plugin", "context", "history-reset", "redis"},
		Fn:          testHistoryResetRecovery,
	})
}

const (
	historyResetRecoveryProbe       = "__HISTORY_RESET_RECOVERY_PROBE__"
	historyResetRecoveryEntryPrefix = "vsr:context-recovery:v1:entry:"
	historyResetEnvelopeVersion     = "vsr.history-reset.v1"

	// The response-api-redis profile's provider mocker.
	historyResetRecoveryBackendNamespace = "default"
	historyResetRecoveryBackendService   = "provider-mocker"
	historyResetRecoveryBackendPort      = "8000"
)

func historyResetCallerTool() map[string]any {
	return map[string]any{"type": "function", "function": map[string]any{
		"name": "lookup", "description": "Look up a value",
		"parameters": map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}},
	}}
}

func testHistoryResetRecovery(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	backend, err := fixtures.OpenServiceEndpointSession(ctx, client, opts,
		historyResetRecoveryBackendNamespace, historyResetRecoveryBackendService, historyResetRecoveryBackendPort)
	if err != nil {
		return err
	}
	defer backend.Close()
	apiSession, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("open Router management API session: %w", err)
	}
	defer apiSession.Close()

	// The marker appears only in the removed turn, so finding it in Redis
	// proves the stored payload is that turn, and finding it in the provider
	// request or in Replay would be a leak.
	removedMarker := fmt.Sprintf("zqremovedmarker%d", time.Now().UnixNano())
	live := "Unrelated question: how do I renew a passport? " + historyResetRecoveryProbe
	replaySession := historyResetObservationID("recovery-replay")
	observationID := historyResetObservationID("recovery")
	response, err := historyResetPost(ctx, session, "/v1/chat/completions", map[string]any{
		// An explicit provider model bypasses decision routing; the route
		// is reached through the router's auto model.
		"model": "vllm-sr/auto",
		"tools": []any{historyResetCallerTool()},
		"messages": []any{
			map[string]any{"role": "user", "content": "What is the weather in Paris today? " + removedMarker},
			map[string]any{"role": "assistant", "content": "It is sunny in Paris, 21C."},
			map[string]any{"role": "user", "content": live},
		},
	}, map[string]string{
		historyResetObservationHeader: observationID,
		"x-session-id":                replaySession,
		// Recovery is scoped to an authenticated user, so stored turns can
		// never be retrieved across users; an anonymous request is refused.
		"x-authz-user-id": "e2e-history-reset-user",
	})
	if err != nil {
		return err
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("status %d", response.StatusCode)
	}

	observed, dispatched, err := historyResetObserve(ctx, backend, observationID)
	if err != nil {
		return err
	}
	if !dispatched {
		return fmt.Errorf("the request never reached the provider")
	}
	key, err := assertHistoryResetRecoveryDispatch(observed, live, removedMarker)
	if err != nil {
		return err
	}
	scanned, err := assertHistoryResetRecoveryStored(ctx, client, key, removedMarker)
	if err != nil {
		return err
	}

	record, err := historyResetReplayRecord(ctx, apiSession, replaySession, "")
	if err != nil {
		return fmt.Errorf("replay: %w", err)
	}
	receipt, err := historyResetReceiptFrom(record)
	if err != nil {
		return err
	}
	if receipt.Outcome != "applied" || receipt.RemovedMessages != 2 || receipt.RemovedTurns != 1 ||
		receipt.RecoveryStatus != "stored" || receipt.RecoveryEntries != 1 {
		return fmt.Errorf("unexpected replay receipt %+v", receipt)
	}
	if paths := historyResetLeakPaths(record, key); len(paths) > 0 {
		return fmt.Errorf("the replay record contains the recovery key at %v", paths)
	}
	if paths := historyResetLeakPaths(record, removedMarker); len(paths) > 0 {
		return fmt.Errorf("the replay record contains removed content at %v", paths)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"provider_messages":     len(historyResetMessages(observed)),
			"recovery_entries_seen": scanned,
			"replay_recovery":       receipt.RecoveryStatus,
		})
	}
	return nil
}

// assertHistoryResetRecoveryDispatch checks the provider-bound Chat request:
// only the live turn remains, the caller's tool is unchanged, and the
// router-owned retrieval tool accepts exactly one issued key, which it returns.
func assertHistoryResetRecoveryDispatch(body map[string]any, live string, removedMarker string) (string, error) {
	messages := historyResetMessages(body)
	if len(messages) != 1 || !strings.Contains(historyResetMessageText(messages[0]), live) {
		return "", fmt.Errorf("provider received %d messages, want only the live turn", len(messages))
	}
	encoded, _ := json.Marshal(body)
	if strings.Contains(string(encoded), removedMarker) {
		return "", fmt.Errorf("the removed turn reached the provider")
	}
	tools, _ := body["tools"].([]any)
	var caller, retrieve map[string]any
	for _, item := range tools {
		tool, _ := item.(map[string]any)
		function, _ := tool["function"].(map[string]any)
		switch function["name"] {
		case "lookup":
			caller = function
		case historyResetRetrieveTool:
			retrieve = function
		}
	}
	if len(tools) != 2 || caller == nil || retrieve == nil {
		return "", fmt.Errorf("provider tools are %d, want the caller's tool and the retrieval tool", len(tools))
	}
	if !reflect.DeepEqual(caller, historyResetCallerTool()["function"]) {
		return "", fmt.Errorf("the caller's tool was changed")
	}
	parameters, _ := retrieve["parameters"].(map[string]any)
	properties, _ := parameters["properties"].(map[string]any)
	keySchema, _ := properties["key"].(map[string]any)
	keys, _ := keySchema["enum"].([]any)
	if len(keys) != 1 {
		return "", fmt.Errorf("the retrieval tool offers %d keys, want exactly one", len(keys))
	}
	key, _ := keys[0].(string)
	if key == "" {
		return "", fmt.Errorf("the retrieval tool offers an empty key")
	}
	return key, nil
}

// assertHistoryResetRecoveryStored finds the stored entry for the issued key
// and checks the envelope it holds. Entry names are derived from a hash of
// the key, so every entry is scanned and matched by its decoded key. Raw
// values, keys, and payloads are never printed.
func assertHistoryResetRecoveryStored(
	ctx context.Context,
	client *kubernetes.Clientset,
	key string,
	removedMarker string,
) (int, error) {
	pod, err := findRedisPod(ctx, client, "app=redis")
	if err != nil {
		return 0, fmt.Errorf("find the redis pod: %w", err)
	}
	if pod == "" {
		return 0, fmt.Errorf("no redis pod found")
	}
	// --scan follows the SCAN cursor until it returns to zero.
	listing, err := execRedisCli(ctx, pod, false, false, "--scan", "--pattern", historyResetRecoveryEntryPrefix+"*")
	if err != nil {
		return 0, err
	}
	names := strings.Fields(listing)
	for _, name := range names {
		value, err := execRedisCli(ctx, pod, false, false, "GET", name)
		if err != nil {
			return len(names), fmt.Errorf("read a recovery entry: %w", err)
		}
		var entry struct {
			Key     string `json:"key"`
			Content string `json:"content"`
		}
		if json.Unmarshal([]byte(value), &entry) != nil || entry.Key != key {
			continue
		}
		return len(names), assertHistoryResetEnvelope(entry.Content, removedMarker)
	}
	return len(names), fmt.Errorf("none of %d recovery entries holds the issued key", len(names))
}

func assertHistoryResetEnvelope(content string, removedMarker string) error {
	var envelope struct {
		Version  string `json:"version"`
		Turns    int    `json:"turns"`
		Messages int    `json:"messages"`
		Removed  []struct {
			ID      int `json:"id"`
			Message struct {
				Role    string `json:"Role"`
				Content []struct {
					Text string `json:"Text"`
				} `json:"Content"`
			} `json:"message"`
		} `json:"removed"`
	}
	if err := json.Unmarshal([]byte(content), &envelope); err != nil {
		return fmt.Errorf("the stored payload is not a history-reset envelope")
	}
	if envelope.Version != historyResetEnvelopeVersion || envelope.Turns != 1 ||
		envelope.Messages != 2 || len(envelope.Removed) != 2 {
		return fmt.Errorf("unexpected envelope: version=%q turns=%d messages=%d removed=%d",
			envelope.Version, envelope.Turns, envelope.Messages, len(envelope.Removed))
	}
	first, second := envelope.Removed[0], envelope.Removed[1]
	if first.ID >= second.ID || first.Message.Role != "user" || second.Message.Role != "assistant" {
		return fmt.Errorf("the stored turn is not in its original order")
	}
	if len(first.Message.Content) == 0 || !strings.Contains(first.Message.Content[0].Text, removedMarker) {
		return fmt.Errorf("the stored turn does not hold the removed content")
	}
	return nil
}
