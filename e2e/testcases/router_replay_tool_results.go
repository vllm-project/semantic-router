package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const (
	replayToolResultsModel  = "vllm-sr/replay-tool-results"
	replayToolResultsRecipe = "replay-tool-results"
	replayToolResultsLimit  = 32
	replayToolResultText    = "Forecast for the test city: sunny, 21C, light wind from the west."
)

func init() {
	pkgtestcases.Register("router-replay-tool-result-capture", pkgtestcases.TestCase{
		Description: "capture_tool_results stores bounded client tool results for replay.detail readers, " +
			"redacts them for other principals, and leaves routes without the option redacted",
		Tags: []string{"router-replay", "functional", "privacy"},
		Fn:   testRouterReplayToolResultCapture,
	})
}

type replayToolResultStep struct {
	Type            string `json:"type"`
	ToolCallID      string `json:"tool_call_id"`
	Status          string `json:"status"`
	Output          string `json:"output"`
	ContentRedacted bool   `json:"content_redacted"`
	Truncated       bool   `json:"truncated"`
}

type replayToolResultRecord struct {
	RequestBody string `json:"request_body"`
	ToolTrace   *struct {
		Steps []replayToolResultStep `json:"steps"`
	} `json:"tool_trace"`
}

func (record replayToolResultRecord) toolResult() (replayToolResultStep, bool) {
	if record.ToolTrace == nil {
		return replayToolResultStep{}, false
	}
	for _, step := range record.ToolTrace.Steps {
		if step.Type == "client_tool_result" && step.ToolCallID == "call_weather" {
			return step, true
		}
	}
	return replayToolResultStep{}, false
}

func testRouterReplayToolResultCapture(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("open session: %w", err)
	}
	defer session.Close()
	apiSession, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("open Router management API session: %w", err)
	}
	defer apiSession.Close()

	base := fmt.Sprintf("e2e_tool_results_%d", time.Now().UnixNano())
	captured, redacted := base+"_captured", base+"_redacted"
	for _, request := range []struct{ model, sessionID string }{
		{replayToolResultsModel, captured},
		{"vllm-sr/replay-first", redacted},
	} {
		if postErr := postReplayToolLoop(ctx, session, request.model, request.sessionID); postErr != nil {
			return postErr
		}
	}

	for _, check := range []struct {
		name, sessionID, token string
		assert                 func(replayToolResultRecord) error
	}{
		{"replay.detail reader", captured, routerReplayDetailToken, assertCapturedToolResult},
		{"reader without replay.detail", captured, routerReplayManagementToken, func(record replayToolResultRecord) error {
			return assertRedactedToolResult(record, false)
		}},
		{"route without capture_tool_results", redacted, routerReplayDetailToken, func(record replayToolResultRecord) error {
			return assertRedactedToolResult(record, true)
		}},
	} {
		record, awaitErr := awaitReplayToolResultRecord(ctx, apiSession, check.sessionID, check.token)
		if awaitErr != nil {
			return fmt.Errorf("%s: %w", check.name, awaitErr)
		}
		if assertErr := check.assert(record); assertErr != nil {
			return fmt.Errorf("%s: %w", check.name, assertErr)
		}
	}
	return assertCapturedToolResultTrajectory(ctx, apiSession, captured)
}

func postReplayToolLoop(ctx context.Context, session *fixtures.ServiceSession, model, sessionID string) error {
	body := map[string]any{
		"model": model,
		"messages": []map[string]any{
			{"role": "user", "content": "What is the weather in the test city?"},
			{"role": "assistant", "content": nil, "tool_calls": []map[string]any{{
				"id": "call_weather", "type": "function",
				"function": map[string]any{"name": "get_weather", "arguments": `{"city":"test"}`},
			}}},
			{"role": "tool", "tool_call_id": "call_weather", "content": replayToolResultText},
		},
		"tools": []map[string]any{{
			"type": "function",
			"function": map[string]any{
				"name":       "get_weather",
				"parameters": map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}},
			},
		}},
	}
	response, err := sendProtocolMatrixRaw(ctx, session, "/v1/chat/completions", body, false, map[string]string{"x-session-id": sessionID})
	if err != nil {
		return fmt.Errorf("%s chat completions: %w", model, err)
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("%s chat completions status %d", model, response.StatusCode)
	}
	return nil
}

// The list omits tool traces, so the record is read from its detail once the
// session's row appears.
func awaitReplayToolResultRecord(
	ctx context.Context,
	apiSession *fixtures.ServiceSession,
	sessionID, token string,
) (replayToolResultRecord, error) {
	var lastErr error
	for attempt := 0; attempt < 15; attempt++ {
		if attempt > 0 {
			time.Sleep(2 * time.Second)
		}
		items, err := fetchReplayListForSession(apiSession, sessionID, 5)
		if err != nil {
			lastErr = err
			continue
		}
		if len(items) == 0 {
			lastErr = fmt.Errorf("no replay row for session %q yet", sessionID)
			continue
		}
		raw, err := doRouterReplayManagementGETAs(ctx, apiSession, "/api/v1/observability/replays/"+url.PathEscape(items[0].ID), token)
		if err != nil {
			lastErr = err
			continue
		}
		if raw.StatusCode != http.StatusOK {
			lastErr = fmt.Errorf("GET replay detail status %d: %s", raw.StatusCode, string(raw.Body))
			continue
		}
		var record replayToolResultRecord
		if err := json.Unmarshal(raw.Body, &record); err != nil {
			return record, fmt.Errorf("decode replay detail: %w", err)
		}
		if _, ok := record.toolResult(); !ok {
			lastErr = fmt.Errorf("session %q record has no client tool result step yet", sessionID)
			continue
		}
		return record, nil
	}
	return replayToolResultRecord{}, lastErr
}

func assertCapturedToolResult(record replayToolResultRecord) error {
	step, _ := record.toolResult()
	if step.ContentRedacted || step.Status != "succeeded" {
		return fmt.Errorf("tool result step = %+v, want captured and succeeded", step)
	}
	if step.Output != replayToolResultText[:replayToolResultsLimit] || !step.Truncated {
		return fmt.Errorf("tool result output = %q truncated=%v, want the first %d bytes marked truncated",
			step.Output, step.Truncated, replayToolResultsLimit)
	}
	if !strings.Contains(record.RequestBody, replayToolResultText) {
		return fmt.Errorf("captured request body lost the tool result")
	}
	return nil
}

func assertRedactedToolResult(record replayToolResultRecord, bodyCaptured bool) error {
	step, _ := record.toolResult()
	if !step.ContentRedacted || step.Output != "" || step.Status != "succeeded" {
		return fmt.Errorf("tool result step = %+v, want redacted with its status", step)
	}
	if strings.Contains(record.RequestBody, replayToolResultText) {
		return fmt.Errorf("request body exposes the tool result")
	}
	if bodyCaptured && record.RequestBody == "" {
		return fmt.Errorf("request body was not captured for a replay.detail reader")
	}
	return nil
}

func assertCapturedToolResultTrajectory(ctx context.Context, apiSession *fixtures.ServiceSession, sessionID string) error {
	target := "/api/v1/observability/replays/trajectory?session_id=" + url.QueryEscape(sessionID) +
		"&recipe=" + url.QueryEscape(replayToolResultsRecipe)
	raw, err := doRouterReplayManagementGETAs(ctx, apiSession, target, routerReplayDetailToken)
	if err != nil {
		return err
	}
	if raw.StatusCode != http.StatusOK {
		return fmt.Errorf("GET trajectory status %d: %s", raw.StatusCode, string(raw.Body))
	}
	var trajectory struct {
		Messages []struct {
			Role             string `json:"role"`
			Content          string `json:"content"`
			ToolCallID       string `json:"tool_call_id"`
			ContentTruncated bool   `json:"content_truncated"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(raw.Body, &trajectory); err != nil {
		return fmt.Errorf("decode trajectory: %w", err)
	}
	for _, message := range trajectory.Messages {
		if message.Role == "tool" && message.ToolCallID == "call_weather" {
			if message.Content != replayToolResultText[:replayToolResultsLimit] || !message.ContentTruncated {
				return fmt.Errorf("trajectory tool message = %+v, want the truncated captured result", message)
			}
			return nil
		}
	}
	return fmt.Errorf("trajectory has no tool message for call_weather: %+v", trajectory.Messages)
}
