package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const (
	topicContinuityModel    = "vllm-sr/topic-continuity"
	topicContinuityDecision = "topic_continuity_decision"
	topicContinuitySignal   = "topic_boundary"
)

func init() {
	pkgtestcases.Register("topic-continuity-replay-receipts", pkgtestcases.TestCase{
		Description: "A declared topic_continuity rule records continuation, full-coverage change, and " +
			"beyond-window unknown receipts in Router Replay without changing routing",
		Tags: []string{"router-replay", "functional", "topic-continuity"},
		Fn:   testTopicContinuityReplayReceipts,
	})
}

type topicContinuityCase struct {
	name     string
	messages []fixtures.ChatMessage
	class    string
	reason   string
	coverage string
}

type topicContinuityReceipt struct {
	Signal        string  `json:"signal"`
	SchemaVersion string  `json:"schema_version"`
	Class         string  `json:"class"`
	Reason        string  `json:"reason"`
	Coverage      string  `json:"coverage"`
	Confidence    float64 `json:"confidence"`
	HistorySource string  `json:"history_source"`
}

type topicContinuityReplayRecord struct {
	ID               string `json:"id"`
	RouteDiagnostics struct {
		Decision        string                   `json:"decision"`
		TopicContinuity []topicContinuityReceipt `json:"topic_continuity"`
	} `json:"route_diagnostics"`
}

func topicContinuityCases() []topicContinuityCase {
	refactor := []fixtures.ChatMessage{
		{Role: "user", Content: "Refactor the routing module so plugins load lazily"},
		{Role: "assistant", Content: "Done. The loader now defers plugin initialization."},
	}
	var longHistory []fixtures.ChatMessage
	for i := 0; i < 3; i++ {
		longHistory = append(longHistory, refactor...)
	}
	change := fixtures.ChatMessage{Role: "user", Content: "Unrelated question: how do I renew a passport?"}
	return []topicContinuityCase{
		{
			name: "continuation",
			messages: []fixtures.ChatMessage{
				{Role: "user", Content: "There is a crash in auth.ts inside validateToken when the header is empty"},
				{Role: "assistant", Content: "The crash comes from validateToken reading a missing header."},
				{Role: "user", Content: "Add a unit test for validateToken in auth.ts"},
			},
			class: "continuation", reason: "continuation_entity_overlap", coverage: "full",
		},
		{
			name:     "explicit change",
			messages: append(append([]fixtures.ChatMessage{}, refactor...), change),
			class:    "change", reason: "change_explicit_marker", coverage: "full",
		},
		{
			// The recipe's window is two prior turns; three exist, so change
			// must not be claimed for history that was not evaluated.
			name:     "history beyond window",
			messages: append(append([]fixtures.ChatMessage{}, longHistory...), change),
			class:    "unknown", reason: "unknown_history_beyond_window", coverage: "window",
		},
	}
}

func testTopicContinuityReplayReceipts(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
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

	base := fmt.Sprintf("e2e_topic_%d", time.Now().UnixNano())
	for i, tc := range topicContinuityCases() {
		sessionID := fmt.Sprintf("%s_%d", base, i)
		if err := postTopicContinuityChat(ctx, session, sessionID, tc.messages); err != nil {
			return fmt.Errorf("%s: %w", tc.name, err)
		}
		record, err := awaitTopicContinuityRecord(ctx, apiSession, sessionID)
		if err != nil {
			return fmt.Errorf("%s: %w", tc.name, err)
		}
		if err := assertTopicContinuityRecord(tc, record); err != nil {
			return fmt.Errorf("%s: %w", tc.name, err)
		}
		if opts.Verbose {
			fmt.Printf("[Test] topic continuity %q: %+v\n", tc.name, record.RouteDiagnostics.TopicContinuity[0])
		}
	}
	return nil
}

func postTopicContinuityChat(
	ctx context.Context,
	session *fixtures.ServiceSession,
	sessionID string,
	messages []fixtures.ChatMessage,
) error {
	chat := fixtures.NewChatCompletionsClient(session, 45*time.Second)
	resp, err := chat.Create(ctx, fixtures.ChatCompletionsRequest{
		Model:    topicContinuityModel,
		Messages: messages,
	}, map[string]string{"x-session-id": sessionID})
	if err != nil {
		return fmt.Errorf("chat completions: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("chat completions status %d: %s", resp.StatusCode, string(resp.Body))
	}
	return nil
}

// The replay list omits route diagnostics, so the receipt is read from the
// record detail once the session's row appears.
func awaitTopicContinuityRecord(
	ctx context.Context,
	apiSession *fixtures.ServiceSession,
	sessionID string,
) (topicContinuityReplayRecord, error) {
	var record topicContinuityReplayRecord
	var lastErr error
	for attempt := 0; attempt < 15; attempt++ {
		if attempt > 0 {
			time.Sleep(2 * time.Second)
		}
		items, err := fetchReplayListForSession(apiSession, sessionID, 5)
		if err != nil {
			lastErr = fmt.Errorf("list replay rows for session %q: %w", sessionID, err)
			continue
		}
		if len(items) == 0 {
			lastErr = fmt.Errorf("no replay row for session %q yet", sessionID)
			continue
		}
		raw, err := doRouterReplayManagementGETAs(ctx, apiSession,
			"/api/v1/observability/replays/"+url.PathEscape(items[0].ID), routerReplayDetailToken)
		if err != nil {
			lastErr = err
			continue
		}
		if raw.StatusCode != http.StatusOK {
			lastErr = fmt.Errorf("GET replay detail status %d: %s", raw.StatusCode, string(raw.Body))
			continue
		}
		if err := json.Unmarshal(raw.Body, &record); err != nil {
			return record, fmt.Errorf("decode replay detail: %w", err)
		}
		return record, nil
	}
	return record, lastErr
}

func assertTopicContinuityRecord(tc topicContinuityCase, record topicContinuityReplayRecord) error {
	diagnostics := record.RouteDiagnostics
	// Topic continuity is context-policy evidence: routing must not change.
	if diagnostics.Decision != topicContinuityDecision {
		return fmt.Errorf("decision = %q, want %q", diagnostics.Decision, topicContinuityDecision)
	}
	if len(diagnostics.TopicContinuity) != 1 {
		return fmt.Errorf("topic_continuity receipts = %+v, want exactly one", diagnostics.TopicContinuity)
	}
	receipt := diagnostics.TopicContinuity[0]
	if receipt.Signal != topicContinuitySignal || receipt.SchemaVersion != "v1" ||
		receipt.HistorySource != "original_snapshot" {
		return fmt.Errorf("receipt identity = %+v", receipt)
	}
	if receipt.Class != tc.class || receipt.Reason != tc.reason || receipt.Coverage != tc.coverage {
		return fmt.Errorf("receipt = %s/%s/%s, want %s/%s/%s",
			receipt.Class, receipt.Reason, receipt.Coverage, tc.class, tc.reason, tc.coverage)
	}
	return nil
}
