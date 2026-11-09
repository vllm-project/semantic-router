package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
)

// historyResetObservationHeader asks the provider mocker to record the
// provider-bound request under this ID (tools/test/services/provider-mocker/
// provider_mocker/provider_boundary.py). The mocker keeps only the last
// request per ID, so every request in these cases uses its own.
const historyResetObservationHeader = "x-vsr-test-session-id"

// historyResetRetrieveTool mirrors contextcompression.RetrieveToolName; the
// e2e module does not depend on the router module, so it cannot import it.
const historyResetRetrieveTool = "vsr_context_retrieve"

func historyResetObservationID(name string) string {
	return fmt.Sprintf("history-reset-%s-%d", name, time.Now().UnixNano())
}

// historyResetPost sends one JSON request through the router.
func historyResetPost(
	ctx context.Context,
	session *fixtures.ServiceSession,
	path string,
	payload any,
	headers map[string]string,
) (*fixtures.HTTPResponse, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, session.URL(path), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	for name, value := range headers {
		request.Header.Set(name, value)
	}
	response, err := session.HTTPClient(60 * time.Second).Do(request)
	if err != nil {
		return nil, err
	}
	defer func() { _ = response.Body.Close() }()
	raw, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, err
	}
	return &fixtures.HTTPResponse{StatusCode: response.StatusCode, Headers: response.Header, Body: raw}, nil
}

// historyResetObserve returns the request body the provider received for the
// observation ID, or found=false when the provider was never called. A 404 is
// the provider mocker's own statement that it recorded nothing for the ID,
// which is how these cases prove a rejection happened before dispatch.
func historyResetObserve(
	ctx context.Context,
	backend *fixtures.ServiceSession,
	observationID string,
) (map[string]any, bool, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, backend.URL("/debug/last-request"), nil)
	if err != nil {
		return nil, false, err
	}
	request.Header.Set(historyResetObservationHeader, observationID)
	response, err := backend.HTTPClient(30 * time.Second).Do(request)
	if err != nil {
		return nil, false, fmt.Errorf("provider observation request failed: %w", err)
	}
	defer func() { _ = response.Body.Close() }()
	raw, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, false, err
	}
	switch response.StatusCode {
	case http.StatusNotFound:
		return nil, false, nil
	case http.StatusOK:
		var observation struct {
			Body map[string]any `json:"body"`
		}
		if err := json.Unmarshal(raw, &observation); err != nil {
			return nil, false, fmt.Errorf("provider observation is not valid JSON: %w", err)
		}
		return observation.Body, true, nil
	default:
		return nil, false, fmt.Errorf("unexpected provider observation status %d", response.StatusCode)
	}
}

// historyResetMessages returns the provider-bound message list. Chat bodies
// and Anthropic Messages bodies both carry it under "messages"; Anthropic
// keeps the system prompt in a separate field.
func historyResetMessages(body map[string]any) []map[string]any {
	items, _ := body["messages"].([]any)
	messages := make([]map[string]any, 0, len(items))
	for _, item := range items {
		if message, ok := item.(map[string]any); ok {
			messages = append(messages, message)
		}
	}
	return messages
}

// historyResetMessageText flattens one message's text, whether its content is
// a string or a list of blocks.
func historyResetMessageText(message map[string]any) string {
	switch content := message["content"].(type) {
	case string:
		return content
	case []any:
		var text string
		for _, block := range content {
			if part, ok := block.(map[string]any); ok {
				if value, ok := part["text"].(string); ok {
					text += value
				}
			}
		}
		return text
	}
	return ""
}

// historyResetReplayRecord waits for this session's Replay row and returns the
// raw record detail, so a case can both decode its diagnostics and search the
// complete serialized record. An empty token sends no Authorization header.
func historyResetReplayRecord(
	ctx context.Context,
	apiSession *fixtures.ServiceSession,
	sessionID string,
	token string,
) ([]byte, error) {
	var lastErr error
	for attempt := 0; attempt < 15; attempt++ {
		if attempt > 0 {
			time.Sleep(2 * time.Second)
		}
		query := url.Values{"session_id": {sessionID}, "limit": {"5"}}
		list, err := historyResetManagementGET(ctx, apiSession, "/api/v1/observability/replays?"+query.Encode(), token)
		if err != nil {
			lastErr = err
			continue
		}
		var rows struct {
			Data []replayListItem `json:"data"`
		}
		if err := json.Unmarshal(list, &rows); err != nil {
			return nil, fmt.Errorf("decode replay list: %w", err)
		}
		if len(rows.Data) == 0 {
			lastErr = fmt.Errorf("no replay row for session %q yet", sessionID)
			continue
		}
		return historyResetManagementGET(ctx, apiSession, "/api/v1/observability/replays/"+url.PathEscape(rows.Data[0].ID), token)
	}
	return nil, lastErr
}

func historyResetManagementGET(
	ctx context.Context,
	apiSession *fixtures.ServiceSession,
	target string,
	token string,
) ([]byte, error) {
	headers := map[string]string{}
	if token != "" {
		headers["Authorization"] = "Bearer " + token
	}
	response, err := fixtures.DoGETRequestWithHeaders(ctx, apiSession.HTTPClient(30*time.Second), apiSession.URL(target), headers)
	if err != nil {
		return nil, err
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GET %s status %d", target, response.StatusCode)
	}
	return response.Body, nil
}

// historyResetReplayReceipt is the content-free receipt Replay stores.
type historyResetReplayReceipt struct {
	Signal           string `json:"signal"`
	Scope            string `json:"scope"`
	TriggerClass     string `json:"trigger_class"`
	Outcome          string `json:"outcome"`
	Reason           string `json:"reason"`
	ExaminedMessages int    `json:"examined_messages"`
	RetainedMessages int    `json:"retained_messages"`
	RemovedMessages  int    `json:"removed_messages"`
	RemovedTurns     int    `json:"removed_turns"`
	RecoveryStatus   string `json:"recovery_status"`
	RecoveryEntries  int    `json:"recovery_entries"`
}

func historyResetReceiptFrom(record []byte) (historyResetReplayReceipt, error) {
	var detail struct {
		RouteDiagnostics struct {
			HistoryReset *historyResetReplayReceipt `json:"history_reset"`
		} `json:"route_diagnostics"`
	}
	if err := json.Unmarshal(record, &detail); err != nil {
		return historyResetReplayReceipt{}, fmt.Errorf("decode replay detail: %w", err)
	}
	if detail.RouteDiagnostics.HistoryReset == nil {
		return historyResetReplayReceipt{}, fmt.Errorf("replay record has no history_reset receipt")
	}
	return *detail.RouteDiagnostics.HistoryReset, nil
}

// historyResetLeakPaths reports where in a JSON document a value contains
// needle. Only the paths are returned, never the values, so a failing
// privacy assertion can say where it found content without printing it.
func historyResetLeakPaths(document []byte, needle string) []string {
	var decoded any
	if json.Unmarshal(document, &decoded) != nil {
		if bytes.Contains(document, []byte(needle)) {
			return []string{"<undecodable document>"}
		}
		return nil
	}
	var paths []string
	var walk func(path string, value any)
	walk = func(path string, value any) {
		switch typed := value.(type) {
		case map[string]any:
			for key, child := range typed {
				if strings.Contains(key, needle) {
					paths = append(paths, path+"{key}")
				}
				walk(path+"."+key, child)
			}
		case []any:
			for index, child := range typed {
				walk(fmt.Sprintf("%s[%d]", path, index), child)
			}
		case string:
			if strings.Contains(typed, needle) {
				paths = append(paths, path)
			}
		}
	}
	walk("$", decoded)
	return paths
}

// historyResetRawReceipt returns the serialized history_reset receipt alone.
func historyResetRawReceipt(record []byte) ([]byte, error) {
	var detail struct {
		RouteDiagnostics struct {
			HistoryReset json.RawMessage `json:"history_reset"`
		} `json:"route_diagnostics"`
	}
	if err := json.Unmarshal(record, &detail); err != nil {
		return nil, err
	}
	if len(detail.RouteDiagnostics.HistoryReset) == 0 {
		return nil, fmt.Errorf("replay record has no history_reset receipt")
	}
	return detail.RouteDiagnostics.HistoryReset, nil
}
